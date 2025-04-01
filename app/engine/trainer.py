import os
import math
import csv
import wandb
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime

from app.util.logger import log_validation_samples
from app.util.utils import set_seed, ensure_dir
from app.engine.evaluator import Evaluator

from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, TrainerCallback
from torch.utils.data import Dataset


class ValidationCallback(TrainerCallback):
    """
    自定义回调函数，在训练过程中定期运行验证。
    会调用父级Trainer的_eval_validation方法。
    """

    def __init__(self, trainer):
        self.trainer = trainer
        self.last_validation_step = 0  # Track when we last did validation
        self.validation_interval = self.trainer.sft_config.eval_steps  # How often to validate

    def on_step_end(self, args, state, control, **kwargs):
        """Only run validation if enough steps have passed since the last validation"""
        current_step = state.global_step
        steps_since_last = current_step - self.last_validation_step

        if steps_since_last >= self.validation_interval:
            print(f"[INFO] 在步骤 {state.global_step} 运行验证 (上次验证: {self.last_validation_step})")
            # 更新全局步数用于日志
            self.trainer.global_step = state.global_step
            # 运行验证
            self.trainer._eval_validation()
            # Update last validation step
            self.last_validation_step = current_step


class Trainer:
    """
    This Trainer merges your approach with the TRL SFTTrainer for supervised fine-tuning
    of Qwen2-VL. We keep your prior code's style of logging and shape debugging as is.

    Key Changes for shape mismatch fix:
    - In `my_data_collator`, we add special vision tokens if `images` is present in the batch:
      1) `vision_start_token_id` once.
      2) `image_token_id` repeated as many times as the final image embeddings (e.g. 256).
    - We automatically compute the needed number of image tokens based on
      (temporal_patch_size * (height // patch_size) * (width // patch_size)) / (spatial_merge_size^2).
    - We set the appended tokens in `labels` to `-100` so as not to compute loss on them.

    New enhancement:
    - Custom model saving logic to properly handle shared tensors between model components
    """

    def __init__(
            self,
            model,
            train_dataloader,
            val_dataloader=None,
            lr=1e-4,
            max_epochs=3,
            grad_acc_steps=1,
            scheduler_type="linear",
            warmup_steps=1000,
            early_stop_patience=5,
            clip_grad_norm=1.0,
            output_dir="runs",
            run_name="sam4mllm",
            device="cuda",
            use_amp=True,
            log_interval=10
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # 基本超参数
        self.lr = lr
        self.max_epochs = max_epochs
        self.grad_acc_steps = grad_acc_steps
        self.scheduler_type = scheduler_type
        self.warmup_steps = warmup_steps
        self.early_stop_patience = early_stop_patience
        self.clip_grad_norm = clip_grad_norm
        self.output_dir = output_dir
        self.device = device
        self.use_amp = use_amp
        self.log_interval = log_interval

        dt_string = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = os.path.join(self.output_dir, f"{run_name}_{dt_string}")
        ensure_dir(self.run_dir)

        # 设置wandb
        wandb.init(project="SAM4MLLM", name=run_name, dir=self.run_dir)

        # 日志CSV
        self.train_log_path = os.path.join(self.run_dir, "train_log.csv")
        self.val_log_path = os.path.join(self.run_dir, "val_log.csv")
        with open(self.train_log_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["step", "epoch", "val_loss", "val_point_match", "val_iou", "val_dice"])
        with open(self.val_log_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["step", "epoch", "val_loss"])

        # 创建TRL SFT配置
        self.sft_config = SFTConfig(
            num_train_epochs=max_epochs,
            per_device_train_batch_size=1,  # 我们会通过自定义逻辑传递实际批次
            per_device_eval_batch_size=1,  # Force batch size of 1 for evaluation to avoid collation issues
            gradient_accumulation_steps=grad_acc_steps,
            max_seq_length=1536,
            learning_rate=lr,
            lr_scheduler_type=self.scheduler_type,
            warmup_ratio=0.0,
            output_dir=self.run_dir,
            logging_steps=10,
            eval_steps=200,  # 增加验证间隔，减少验证频率
            save_steps=500,  # 同时增加保存间隔
            dataset_kwargs={"skip_prepare_dataset": True},
            remove_unused_columns=False,  # 保留'text_input'/'image'键
            save_safetensors=False,  # 避免safetensors共享张量问题
        )

        # 准备评估器，并限制最大评估样本数
        self.evaluator = Evaluator(
            self.model,
            self.device,
            output_dir=os.path.join(self.run_dir, "validation_viz"),
            max_eval_samples=10  # Only evaluate a maximum of 10 samples
        ) if self.val_dataloader else None

        # 全局步数跟踪
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0
        self.should_stop = False

        # 将模型移至设备
        self.model.to(self.device)

    def train(self):
        """
        我们创建TRL的SFTTrainer并提供现有的训练和验证数据集。
        我们依赖SFTTrainer的标准train()循环来实现高级SFT逻辑。
        我们保留了您之前代码的日志记录和形状调试风格。

        添加了ValidationCallback以在训练过程中定期运行验证。
        """

        print("[Trainer] 创建TRL SFTTrainer...")

        def _compute_n_img_tokens_for_sample(img_tensor: torch.Tensor, backbone) -> int:
            """
            计算Qwen2-VL为单个图像生成的最终patch标记数量
            基于用户指定的patch_size、temporal_patch_size和spatial_merge_size。

            例如，对于512×512的图像，patch_size=16，spatial_merge_size=2，
            temporal_patch_size=1 => 最终256个标记。
            """
            if img_tensor is None or img_tensor.shape[0] == 0:
                return 0

            # 形状 = [channels, H, W]
            _, h, w = img_tensor.shape
            patch_size = getattr(backbone, 'override_patch_size', 14)
            temporal_patch_size = getattr(backbone, 'override_temporal_patch_size', 1)

            # 尝试从vision_config读取，如果存在
            if hasattr(backbone.qwen_model.config, "vision_config"):
                vc = backbone.qwen_model.config.vision_config
                if hasattr(vc, "spatial_merge_size"):
                    spatial_merge = vc.spatial_merge_size
                else:
                    spatial_merge = 2
            else:
                spatial_merge = 2

            # T x (H//patch_size) x (W//patch_size)，然后 / (spatial_merge^2)
            t_count = temporal_patch_size  # 对于单个图像
            h_count = h // patch_size
            w_count = w // patch_size
            total_patches = t_count * h_count * w_count
            merged_count = total_patches // (spatial_merge * spatial_merge)
            return merged_count

        # 此自定义收集器处理来自数据集的文本+图像
        def my_data_collator(batch):
            """
            批次中的每个元素是具有以下键的dict：
              "text_input", "image", "points_str"
            我们将使用Qwen分词器对'text_input'进行分词，
            然后如果存在images，我们将添加正确数量的视觉标记：
              - 1个vision_start_token_id
              - 重复的image_token_id（匹配最终patch数量）。
            我们还将调整`labels`，使附加标记的标签=-100。

            Enhanced:
            - Added debug prints for shape tracking
            - Added error handling for inconsistent batch items
            - Limit to batch size 1 for validation
            """
            # Print item keys for debugging
            if len(batch) == 0:
                print("[WARNING] Empty batch passed to data collator")
                # Return an empty dict with the expected structure
                return {
                    "input_ids": torch.zeros((0, 0), dtype=torch.long),
                    "attention_mask": torch.zeros((0, 0), dtype=torch.long),
                    "images": torch.zeros((0, 3, 512, 512), dtype=torch.float),
                    "labels": torch.zeros((0, 0), dtype=torch.long),
                }

            # Print debug info
            # print(f"[DEBUG] Batch size: {len(batch)}")
            # print(f"[DEBUG] First batch item keys: {list(batch[0].keys())}")

            try:
                text_list = [item["text_input"] for item in batch]
                images = [item["image"] for item in batch]

                # Qwen的分词器
                tokenizer = self.model.backbone.qwen_tokenizer

                # 1) 基本文本分词
                encoded = tokenizer(
                    text_list,
                    padding=False,  # 我们将在下面手动填充
                    truncation=True,
                    max_length=self.sft_config.max_seq_length,
                    return_tensors="pt",
                )

                # 将图像转换为浮点数
                pixel_values = torch.stack(images, dim=0).float()

                # 从配置中识别特殊标记ID
                vs_id = self.model.backbone.qwen_model.config.vision_start_token_id
                vi_id = self.model.backbone.qwen_model.config.image_token_id
                pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

                batch_input_ids = []
                batch_attention_mask = []
                batch_labels = []

                # 2) 构建最终序列，可能添加视觉标记
                for i in range(len(batch)):
                    input_ids_i = encoded["input_ids"][i]
                    attention_mask_i = encoded["attention_mask"][i]

                    # 我们将克隆用于标签，初始相同
                    labels_i = input_ids_i.clone()

                    # 检查是否有实际图像。如果有，计算预期的patch标记数量
                    if images[i] is not None:
                        n_img_tokens = _compute_n_img_tokens_for_sample(images[i], self.model.backbone)
                    else:
                        n_img_tokens = 0

                    if n_img_tokens > 0:
                        # 插入1个vision_start_token，然后是n_img_tokens个image_token_id
                        # 通常附加在末尾
                        vs_tensor = torch.tensor([vs_id], dtype=torch.long)
                        vi_tensor = torch.tensor([vi_id] * n_img_tokens, dtype=torch.long)
                        # 连接它们
                        input_ids_i = torch.cat([input_ids_i, vs_tensor, vi_tensor], dim=0)

                        # 注意力掩码
                        mask_append = torch.ones(1 + n_img_tokens, dtype=attention_mask_i.dtype)
                        attention_mask_i = torch.cat([attention_mask_i, mask_append], dim=0)

                        # 我们不训练模型生成这些标记，所以我们将它们设置为-100
                        labels_append = torch.full((1 + n_img_tokens,), -100, dtype=labels_i.dtype)
                        labels_i = torch.cat([labels_i, labels_append], dim=0)

                    batch_input_ids.append(input_ids_i)
                    batch_attention_mask.append(attention_mask_i)
                    batch_labels.append(labels_i)

                # 3) 现在将它们填充到此批次的最大长度
                padded_input_ids = torch.nn.utils.rnn.pad_sequence(
                    batch_input_ids,
                    batch_first=True,
                    padding_value=pad_id
                )
                padded_attention_mask = torch.nn.utils.rnn.pad_sequence(
                    batch_attention_mask,
                    batch_first=True,
                    padding_value=0
                )
                padded_labels = torch.nn.utils.rnn.pad_sequence(
                    batch_labels,
                    batch_first=True,
                    padding_value=-100
                )

                # 返回标准批次
                batch_dict = {
                    "input_ids": padded_input_ids,
                    "attention_mask": padded_attention_mask,
                    "images": pixel_values,
                    "labels": padded_labels,
                }
                return batch_dict
            except Exception as e:
                print(f"[ERROR] Error in data collator: {e}")
                # Analyze the issue
                print(f"[DEBUG] Batch length: {len(batch)}")
                for i, item in enumerate(batch):
                    print(f"[DEBUG] Item {i} keys: {list(item.keys())}")
                    if "image" in item:
                        if isinstance(item["image"], torch.Tensor):
                            print(f"[DEBUG] Item {i} image shape: {item['image'].shape}")
                        else:
                            print(f"[DEBUG] Item {i} image type: {type(item['image'])}")

                # Return a default safe batch
                return {
                    "input_ids": torch.zeros((1, 1), dtype=torch.long),
                    "attention_mask": torch.zeros((1, 1), dtype=torch.long),
                    "images": torch.zeros((1, 3, 512, 512), dtype=torch.float),
                    "labels": torch.zeros((1, 1), dtype=torch.long),
                }

        # 自定义保存函数处理共享张量
        def custom_save_model(trainer, output_dir, _internal_call=True):
            """
            处理共享张量的自定义save_model实现。
            这替换了默认SFTTrainer的save_model以避免safetensors错误。
            """
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

            os.makedirs(output_dir, exist_ok=True)
            print(f"[INFO] 保存模型检查点到 {output_dir}")

            # 保存模型状态字典，正确处理共享张量
            # 对于LoRA或PEFT模型，使用PEFT特定保存
            if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'qwen_model'):
                # 分别保存Qwen模型和SAM模型
                if hasattr(self.model.backbone, 'qwen_model'):
                    qwen_output_dir = os.path.join(output_dir, "qwen")
                    os.makedirs(qwen_output_dir, exist_ok=True)
                    print(f"[INFO] 保存Qwen模型到 {qwen_output_dir}")

                    # 检查模型是否为PEFT模型
                    if hasattr(self.model.backbone.qwen_model, 'save_pretrained'):
                        # 对于PEFT模型，仅保存PEFT模型适配器
                        if hasattr(self.model.backbone.qwen_model, 'peft_config'):
                            print(f"[INFO] 检测到PEFT模型，保存适配器")
                            self.model.backbone.qwen_model.save_pretrained(qwen_output_dir)
                        else:
                            # 对于完整模型，使用PyTorch的save以避免safetensors错误
                            print(f"[INFO] 使用PyTorch保存完整模型")
                            torch.save(
                                self.model.backbone.qwen_model.state_dict(),
                                os.path.join(qwen_output_dir, "pytorch_model.bin")
                            )
                            # 同时保存配置
                            if hasattr(self.model.backbone.qwen_model, 'config'):
                                self.model.backbone.qwen_model.config.save_pretrained(qwen_output_dir)

                    # 保存分词器
                    if hasattr(self.model.backbone, 'qwen_tokenizer'):
                        self.model.backbone.qwen_tokenizer.save_pretrained(qwen_output_dir)

                # 如果存在，保存SAM模型
                if hasattr(self.model.backbone, 'sam_model') and self.model.backbone.sam_model is not None:
                    sam_output_dir = os.path.join(output_dir, "sam")
                    os.makedirs(sam_output_dir, exist_ok=True)
                    print(f"[INFO] 保存SAM模型到 {sam_output_dir}")

                    if hasattr(self.model.backbone.sam_model, 'save_pretrained'):
                        self.model.backbone.sam_model.save_pretrained(sam_output_dir)
                    else:
                        # 使用PyTorch的save
                        torch.save(
                            self.model.backbone.sam_model.state_dict(),
                            os.path.join(sam_output_dir, "pytorch_model.bin")
                        )
                        # 如果可用，保存配置
                        if hasattr(self.model.backbone.sam_model, 'config'):
                            self.model.backbone.sam_model.config.save_pretrained(sam_output_dir)

                # 保存概览文件以帮助重新加载
                overview_path = os.path.join(output_dir, "model_overview.txt")
                with open(overview_path, "w") as f:
                    f.write("SAM4MLLM 模型组件:\n")
                    if hasattr(self.model.backbone, 'qwen_model'):
                        f.write(f"- Qwen 模型: {type(self.model.backbone.qwen_model).__name__}\n")
                        if hasattr(self.model.backbone.qwen_model, 'peft_config'):
                            f.write("  - 使用 PEFT/LoRA 适配器\n")
                    if hasattr(self.model.backbone, 'sam_model'):
                        f.write(f"- SAM 模型: {type(self.model.backbone.sam_model).__name__}\n")
            else:
                # 更简单模型的后备方案 - 使用PyTorch保存
                print(f"[INFO] 对于没有识别到的骨干结构的模型，使用PyTorch保存")
                torch.save(
                    self.model.state_dict(),
                    os.path.join(output_dir, "pytorch_model.bin")
                )

            print(f"[INFO] 模型检查点成功保存到 {output_dir}")
            return output_dir

        # 创建并初始化SFT训练器
        sft_trainer = SFTTrainer(
            model=self.model,
            args=self.sft_config,
            train_dataset=self.train_dataloader.dataset,
            eval_dataset=self.val_dataloader.dataset if self.val_dataloader else None,
            data_collator=my_data_collator,
        )

        # 替换save_model方法以正确处理共享张量
        sft_trainer.save_model = lambda output_dir, _internal_call=True: custom_save_model(
            sft_trainer, output_dir, _internal_call
        )

        # 添加验证回调以在训练期间定期运行
        if self.val_dataloader is not None:
            print("[Trainer] 添加ValidationCallback用于训练期间的定期验证")
            sft_trainer.add_callback(ValidationCallback(self))

        print("[Trainer] 开始使用TRL SFTTrainer进行SFT训练...")
        sft_trainer.train()  # TRL的标准训练循环

        # 训练后，如果需要，用您原来的方法进行最终验证
        if self.val_dataloader is not None:
            print("[Trainer] 用您的代码进行最终验证。")
            val_loss = self._eval_validation()
            print(f"[Trainer] 最终验证损失: {val_loss}")

        # 用我们的自定义保存函数保存最终模型
        final_model_path = os.path.join(self.run_dir, "final_model")
        custom_save_model(sft_trainer, final_model_path)
        print(f"[Trainer] SFT训练完成。模型保存在 {final_model_path}")

    def _eval_validation(self):
        """
        We replicate a minimal approach for cross-entropy on val_dataloader.
        Optionally parse points and do SAM-based IoU if needed.
        Also visualize random validation samples and log to wandb.

        Modified to use batch size 1 for evaluation to avoid batch collation issues.
        """
        print("[Trainer] Running validation...")

        # Create a specific output directory for validation visualizations
        val_viz_dir = os.path.join(self.run_dir, "validation_viz")
        ensure_dir(val_viz_dir)

        # Set visualization output directory for evaluator
        if self.evaluator:
            self.evaluator.output_dir = val_viz_dir
            # Perform full evaluation with visualizations
            eval_results = self.evaluator.evaluate(self.val_dataloader)
            val_loss = eval_results.get("loss", 0.0)

            # Log all metrics to wandb
            wandb.log({
                "val_loss": val_loss,
                "val_point_match": eval_results.get("point_match", 0.0),
                "val_iou": eval_results.get("iou", 0.0),
                "val_dice": eval_results.get("dice", 0.0),
            })

            # Log to CSV
            with open(self.val_log_path, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.global_step,
                    self.global_step // len(self.train_dataloader),
                    val_loss,
                    eval_results.get("point_match", 0.0),
                    eval_results.get("iou", 0.0),
                    eval_results.get("dice", 0.0)
                ])
        else:
            # Fallback to simple loss calculation
            self.model.eval()
            losses = []
            for i, batch in enumerate(tqdm(self.val_dataloader, desc="Validation")):
                # Limit validation samples
                if i >= 10:  # Only process up to 10 samples
                    print(f"[INFO] Reached 10 validation samples, stopping validation.")
                    break

                if isinstance(batch["text_input"], list):
                    text_list = batch["text_input"]
                else:
                    text_list = [batch["text_input"]]

                images = batch["image"]
                if isinstance(images, torch.Tensor):
                    pixel_values = images.to(self.device)
                else:
                    pixel_values = None

                encoded = self.model.backbone.qwen_tokenizer(
                    text_list,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.sft_config.max_seq_length
                )
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)

                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        images=pixel_values,
                        labels=input_ids,  # standard cross-entropy
                    )
                    loss_val = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                    losses.append(loss_val.item())

            val_loss = sum(losses) / len(losses) if len(losses) > 0 else 0.0
            wandb.log({"val_loss": val_loss})

            # Log to CSV
            with open(self.val_log_path, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.global_step,
                    self.global_step // len(self.train_dataloader),
                    val_loss
                ])

        return val_loss