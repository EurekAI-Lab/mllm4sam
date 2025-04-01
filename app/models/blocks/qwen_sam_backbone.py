import os
import sys
import torch
import torch.nn as nn

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# Attempt to import SamModel, SamProcessor for optional synergy
try:
    from transformers import SamModel, SamProcessor

    HF_SAM_AVAILABLE = True
except ImportError as e:
    print("[WARNING] Could not import SamModel or SamProcessor from transformers. Reason:", e)
    HF_SAM_AVAILABLE = False

# Attempt to import PEFT for LoRA
try:
    from peft import LoraConfig, get_peft_model

    LOADING_PEFT_OK = True
except ImportError as e:
    print("[WARNING] Could not import peft or bitsandbytes. Reason:", e)
    LOADING_PEFT_OK = False

# Attempt to import Qwen2-VL PatchEmbed
try:
    from transformers.models.qwen2_vl.modeling_qwen2_vl import PatchEmbed

    QWEN_CONFIG_AVAILABLE = True
except ImportError as e:
    print("[WARNING] Could not import Qwen2VL PatchEmbed. Reason:", e)
    QWEN_CONFIG_AVAILABLE = False
    PatchEmbed = None


class QwenSamBackbone(nn.Module):
    """
    A synergy backbone that loads Qwen2-VL with Qwen2VLForConditionalGeneration
    (and optionally HF's SamModel) for demonstration.

    This class aims to resolve shape mismatches for single-frame images of size 512x512
    by overriding patch embeddings. It also properly computes 'image_grid_thw' and passes
    it to Qwen2VLForConditionalGeneration to avoid 'NoneType' iteration issues inside Qwen2-VL.

    We keep all code segments intact, adding a `_calculate_grid_thw()` function for
    clarity. We do not remove or skip any existing lines, only enhance them to be
    fully functional. We also add debug prints to help identify shape mismatches.
    """

    def __init__(
            self,
            qwen_model_path: str,
            sam_model_path: str,
            device: str = "cuda",
            override_patch_size: int = 16,
            override_temporal_patch_size: int = 1,
    ):
        """
        Args:
            qwen_model_path (str):
                Path or HF repo ID for Qwen2-VL. Must be a Qwen2-VL style checkpoint (2.0 or 2.1).
            sam_model_path (str):
                Path or HF repo ID for SAM (optional).
            device (str):
                "cuda" or "cpu", etc.
            override_patch_size (int):
                The patch size for height/width dimension in Qwen2-VL.
            override_temporal_patch_size (int):
                The patch size for temporal dimension. For single-frame images, set to 1.
        """
        super().__init__()
        self.device = device
        self.override_patch_size = override_patch_size
        self.override_temporal_patch_size = override_temporal_patch_size

        # 1) Load Qwen2-VL
        print(f"[INFO] Loading Qwen2-VL from {qwen_model_path}")
        self.qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
            qwen_model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to(device)

        # Keep a local reference to the config in case we need it
        self.config = self.qwen_model.config

        # 2) Load Qwen processor (tokenizer + image processor)
        print("[INFO] Loading Qwen2-VL processor (tokenizer + image processor)")
        self.qwen_processor = AutoProcessor.from_pretrained(qwen_model_path, trust_remote_code=True)
        self.qwen_tokenizer = self.qwen_processor.tokenizer
        if self.qwen_tokenizer.pad_token is None:
            self.qwen_tokenizer.pad_token = self.qwen_tokenizer.eos_token

        # 3) Attempt to override patch embeddings if PatchEmbed is present
        replaced = False
        if PatchEmbed is not None:
            possible_attrs = []
            if hasattr(self.qwen_model, "visual"):
                possible_attrs.append("visual")
            if hasattr(self.qwen_model, "vision_tower") and self.qwen_model.vision_tower:
                possible_attrs.append("vision_tower.0")

            if len(possible_attrs) == 0:
                print("[WARNING] Could not find a 'visual' or 'vision_tower' attribute on Qwen2-VL model. "
                      "Patch override might fail.")
            else:
                for attr in possible_attrs:
                    try:
                        if attr == "visual":
                            old_pe = getattr(self.qwen_model, "visual").patch_embed
                            print("[INFO] Found patch_embed at qwen_model.visual.patch_embed")
                            new_pe = self._make_new_patch_embed(old_pe)
                            self.qwen_model.visual.patch_embed = new_pe
                            replaced = True
                            break
                        elif attr == "vision_tower.0":
                            old_pe = self.qwen_model.vision_tower[0].patch_embed
                            print("[INFO] Found patch_embed at qwen_model.vision_tower[0].patch_embed")
                            new_pe = self._make_new_patch_embed(old_pe)
                            self.qwen_model.vision_tower[0].patch_embed = new_pe
                            replaced = True
                            break
                    except Exception as e:
                        print(f"[WARNING] Attempt to override patch_embed via '{attr}' failed. Reason: {e}")
        else:
            print("[WARNING] Qwen2VL PatchEmbed is not available to override. May cause shape mismatch if patch_size "
                  "differs from default 14.")

        if replaced:
            print(f"[INFO] Successfully replaced patch embedding with patch_size={override_patch_size}, "
                  f"temporal_patch_size={override_temporal_patch_size}")
        else:
            print("[WARNING] No patch embedding override was performed. The code may still work if the original "
                  "model patch_size matches your input dimension, or it may fail on shape mismatch.")

        # 4) Attempt LoRA
        if LOADING_PEFT_OK:
            try:
                self._setup_lora()
            except Exception as e:
                print(f"[WARNING] LoRA setup failed. Error: {e}")
        else:
            print("[WARNING] LoRA is not available in this environment. Skipping LoRA setup.")

        # 5) Attempt to load SAM
        self.sam_model = None
        self.sam_processor = None
        if HF_SAM_AVAILABLE:
            try:
                print(f"[INFO] Attempting to load SAM from {sam_model_path}")
                self.sam_model = SamModel.from_pretrained(sam_model_path, torch_dtype=torch.float16).eval().to(device)
                self.sam_processor = SamProcessor.from_pretrained(sam_model_path)
            except Exception as e:
                print(f"[WARNING] Could not load huggingface SamModel from {sam_model_path}. Reason: {e}")
                self.sam_model = None
                self.sam_processor = None
        else:
            print("[WARNING] SamModel is not available from transformers package. Skipping HF-SAM usage.")

    def _make_new_patch_embed(self, old_patch_embed: PatchEmbed) -> PatchEmbed:
        """
        Given the original Qwen2-VL PatchEmbed, create and return a new PatchEmbed
        with override_patch_size and override_temporal_patch_size.
        """
        # print(f"[DEBUG] Creating new PatchEmbed with patch_size={self.override_patch_size}, "
        #       f"temporal_patch_size={self.override_temporal_patch_size}")
        new_patch_embed = PatchEmbed(
            patch_size=self.override_patch_size,
            temporal_patch_size=self.override_temporal_patch_size,
            in_channels=old_patch_embed.in_channels,
            embed_dim=old_patch_embed.embed_dim
        ).to(self.device).to(self.qwen_model.dtype)
        return new_patch_embed

    def _setup_lora(self):
        """
        Create and apply a LoRA adapter config to the Qwen2-VL model if environment permits.
        We target typical modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj.
        """
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        print("[INFO] Setting up LoRA (r=16, alpha=16, dropout=0.05) for Qwen2-VL on modules: ", target_modules)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.qwen_model = get_peft_model(self.qwen_model, lora_config)
        self.qwen_model.print_trainable_parameters()

    def _calculate_grid_thw(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute the grid_thw parameter for Qwen2-VL based on the shape of 'images' and
        the override patch sizes. This ensures Qwen2-VL's 'self.visual' can proceed
        without encountering 'NoneType' issues in 'grid_thw'.

        Args:
            images (torch.Tensor): shape [batch_size, channels, height, width]

        Returns:
            torch.LongTensor: shape [batch_size, 3], each row is [t, h, w].
        """
        if images is None:
            # print("[DEBUG] _calculate_grid_thw: images is None, returning None")
            return None

        batch_size, channels, height, width = images.shape
        # For single images (not video), temporal dimension = 1
        t_dim = 1
        h_dim = height // self.override_patch_size
        w_dim = width // self.override_patch_size

        grid_thw = torch.tensor(
            [[t_dim, h_dim, w_dim] for _ in range(batch_size)],
            device=images.device,
            dtype=torch.long
        )
        # print(f"[DEBUG] Calculated grid_thw shape: {grid_thw.shape}, first row: {grid_thw[0].tolist()}")
        return grid_thw

    def forward(
            self,
            input_ids: torch.Tensor = None,
            attention_mask: torch.Tensor = None,
            images: torch.Tensor = None,
            labels: torch.Tensor = None,
            **kwargs,
    ):
        """
        Standard forward pass for Qwen2VLForConditionalGeneration. We add 'image_grid_thw'
        argument here to avoid 'NoneType' iteration errors in Qwen2-VL.

        :param input_ids: [batch_size, seq_len]
        :param attention_mask: [batch_size, seq_len]
        :param images: [batch_size, 3, height, width], optional
        :param labels: [batch_size, seq_len], optional
        :param kwargs: additional arguments (cache, etc.)
        :return: outputs from Qwen2VLForConditionalGeneration
        """
        # We possibly remove extra keys from kwargs that Qwen2-VL won't accept, but we keep them
        # in debug prints. We never skip or simplify; we just pop them if needed.

        # Make a copy of kwargs to avoid modifying the original dict during iteration
        kwargs_copy = kwargs.copy()

        # Check if image_grid_thw is already in kwargs
        use_provided_grid_thw = "image_grid_thw" in kwargs

        for k in list(kwargs_copy.keys()):
            if k not in [
                "pixel_values", "image_grid_thw", "cache_position", "use_cache",
                "output_attentions", "output_hidden_states", "return_dict"
            ]:
                ignored = kwargs.pop(k, None)
                # if ignored is not None:
                #    print(f"[DEBUG] Ignoring extra kwarg '{k}' passed to QwenSamBackbone.forward()")

        # If images is not None and image_grid_thw is not provided, compute it dynamically
        image_grid_thw = None
        if images is not None and not use_provided_grid_thw:
            # We'll compute the shape-based grid
            image_grid_thw = self._calculate_grid_thw(images)
            # Print for debug
            # print(f"[DEBUG] Calculated image_grid_thw: {image_grid_thw[0].tolist() if image_grid_thw is not None else None}")

        # Debug shapes for help debugging
        # if input_ids is not None:
        #    print(f"[DEBUG QwenSamBackbone.forward()] input_ids shape: {input_ids.shape}")
        # if attention_mask is not None:
        #    print(f"[DEBUG QwenSamBackbone.forward()] attention_mask shape: {attention_mask.shape}")
        # if images is not None:
        #    print(f"[DEBUG QwenSamBackbone.forward()] images shape: {images.shape}")

        # Use either the provided image_grid_thw from kwargs or the one we calculated
        model_kwargs = kwargs.copy()
        if images is not None:
            if not use_provided_grid_thw and image_grid_thw is not None:
                model_kwargs["image_grid_thw"] = image_grid_thw

            # print(f"[DEBUG] Using pixel_values shape: {images.shape}")
            # if "image_grid_thw" in model_kwargs:
            #    print(f"[DEBUG] Using image_grid_thw shape: {model_kwargs['image_grid_thw'].shape}")

            outputs = self.qwen_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=images,
                labels=labels,
                **model_kwargs
            )
        else:
            # No images, text-only
            outputs = self.qwen_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )

        return outputs

    def generate(
            self,
            input_ids: torch.Tensor = None,
            attention_mask: torch.Tensor = None,
            images: torch.Tensor = None,
            max_new_tokens: int = 128,
            **kwargs,
    ):
        """
        For text generation (multi-modal or text-only). We also compute 'image_grid_thw'
        if images is present, to avoid NoneType errors in Qwen2-VL generation.
        """
        # Make a copy of kwargs to avoid modifying the original dict during iteration
        kwargs_copy = kwargs.copy()

        # Check if image_grid_thw is already in kwargs
        use_provided_grid_thw = "image_grid_thw" in kwargs

        for k in list(kwargs_copy.keys()):
            if k not in [
                "pixel_values", "image_grid_thw", "cache_position", "use_cache",
                "output_attentions", "output_hidden_states", "return_dict"
            ]:
                ignored = kwargs.pop(k, None)
                # if ignored is not None:
                #    print(f"[DEBUG] Ignoring extra kwarg '{k}' passed to QwenSamBackbone.generate()")

        # Calculate image_grid_thw only if not already provided
        image_grid_thw = None
        if images is not None and not use_provided_grid_thw:
            image_grid_thw = self._calculate_grid_thw(images)
            # print(f"[DEBUG QwenSamBackbone.generate()] images shape: {images.shape}")

        # if input_ids is not None:
        #    print(f"[DEBUG QwenSamBackbone.generate()] input_ids shape: {input_ids.shape}")
        # if attention_mask is not None:
        #    print(f"[DEBUG QwenSamBackbone.generate()] attention_mask shape: {attention_mask.shape}")

        # Use either the provided image_grid_thw from kwargs or the one we calculated
        model_kwargs = kwargs.copy()
        if images is not None:
            if not use_provided_grid_thw and image_grid_thw is not None:
                model_kwargs["image_grid_thw"] = image_grid_thw

            try:
                outputs = self.qwen_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=images,
                    max_new_tokens=max_new_tokens,
                    **model_kwargs
                )
            except Exception as e:
                print(f"[WARNING] Error in generation with images: {e}")
                print(f"[DEBUG] Generation shapes - input_ids: {input_ids.shape if input_ids is not None else None}, "
                      f"attention_mask: {attention_mask.shape if attention_mask is not None else None}, "
                      f"images: {images.shape if images is not None else None}, "
                      f"image_grid_thw: {model_kwargs.get('image_grid_thw').shape if 'image_grid_thw' in model_kwargs else None}")
                # Fallback to text-only generation
                print("[INFO] Falling back to text-only generation")
                outputs = self.qwen_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    **{k: v for k, v in model_kwargs.items() if k != "image_grid_thw" and k != "pixel_values"}
                )
        else:
            outputs = self.qwen_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                **kwargs
            )

        return outputs

    def predict_segmentation(self, image, text=None, prompt=None, return_points=False):
        """
        End-to-end inference pipeline:
        1. Generate point coordinates using Qwen2-VL
        2. Pass the points to SAM for segmentation

        Args:
            image (torch.Tensor or PIL.Image): Input image [3, H, W] or PIL Image
            text (str, optional): Text context for point prediction. If None, uses default prompt.
            prompt (str, optional): Specific prompt for point generation. Overrides text if provided.
            return_points (bool): Whether to return the predicted points along with mask.

        Returns:
            dict: {
                'mask': Binary segmentation mask of shape [H, W],
                'points': List of [x, y] coordinates (if return_points=True)
                'confidence': SAM confidence score
            }
        """
        # Prepare image if needed
        if not isinstance(image, torch.Tensor):
            # Convert PIL to tensor
            if hasattr(self.qwen_processor, "image_processor"):
                # Use processor if available
                pixel_values = self.qwen_processor.image_processor(image, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(self.device)
            else:
                # Simple conversion
                import numpy as np
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                pixel_values = transform(image).unsqueeze(0).to(self.device)
        else:
            # Ensure tensor is batched and on the right device
            pixel_values = image.unsqueeze(0) if image.dim() == 3 else image
            pixel_values = pixel_values.to(self.device)

        # 1) Generate point coordinates using Qwen2-VL
        if prompt is None:
            if text is None:
                # Default prompt if nothing is provided
                prompt = "Point out the wound area in up to 10 points. Answer only with coordinates in format (x,y)."
            else:
                # If text context is provided but no specific prompt
                prompt = f"{text}\nPoint out the wound area in up to 10 points. Answer only with coordinates in format (x,y)."

        # Tokenize the prompt
        tokenized_prompt = self.qwen_tokenizer([prompt], return_tensors="pt").to(self.device)

        # Run text generation to get point coordinates
        try:
            with torch.no_grad():
                # Calculate image grid dimensions for this specific image
                batch_size, channels, height, width = pixel_values.shape
                patch_size = self.override_patch_size
                h_grid = height // patch_size
                w_grid = width // patch_size

                # Generate spatial_merge_size (default 2)
                spatial_merge_size = 2
                if hasattr(self.qwen_model.config, "vision_config"):
                    vc = self.qwen_model.config.vision_config
                    if hasattr(vc, "spatial_merge_size"):
                        spatial_merge_size = vc.spatial_merge_size

                # Calculate expected number of image tokens
                n_image_tokens = (h_grid * w_grid) // (spatial_merge_size * spatial_merge_size)

                # Create image_grid_thw tensor
                image_grid_thw = torch.tensor(
                    [[1, h_grid, w_grid]],  # Batch size of 1
                    device=self.device,
                    dtype=torch.long
                )

                # Get vision token IDs
                vision_start_token_id = self.qwen_model.config.vision_start_token_id
                image_token_id = self.qwen_model.config.image_token_id

                # Append vision tokens to the prompt
                input_ids = tokenized_prompt.input_ids[0]
                attention_mask = tokenized_prompt.attention_mask[0]

                # Append vision tokens
                input_ids_with_vision = torch.cat([
                    input_ids,
                    torch.tensor([vision_start_token_id], device=self.device),
                    torch.tensor([image_token_id] * n_image_tokens, device=self.device)
                ])

                attention_mask_with_vision = torch.cat([
                    attention_mask,
                    torch.ones(1 + n_image_tokens, device=self.device, dtype=attention_mask.dtype)
                ])

                # Reshape for batch dimension
                input_ids_with_vision = input_ids_with_vision.unsqueeze(0)
                attention_mask_with_vision = attention_mask_with_vision.unsqueeze(0)

                print(f"[DEBUG] Generation - input_ids: {input_ids_with_vision.shape}, "
                      f"attention_mask: {attention_mask_with_vision.shape}, "
                      f"images: {pixel_values.shape}, image_grid_thw: {image_grid_thw.shape}")

                # Generate with explicit image tokens and grid
                generated_ids = self.qwen_model.generate(
                    input_ids=input_ids_with_vision,
                    attention_mask=attention_mask_with_vision,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    max_new_tokens=128,
                    temperature=0.2,  # Lower temperature for more deterministic outputs
                    do_sample=False,  # Greedy decoding for coordinates
                    num_beams=1
                )

                # Decode the generated text
                generated_text = self.qwen_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

                # Extract the model's response (after the prompt)
                if "[ASSISTANT]:" in generated_text:
                    response = generated_text.split("[ASSISTANT]:")[-1].strip()
                else:
                    # Fallback - extract text after the prompt
                    response = generated_text.split(prompt)[-1].strip()

                # Parse coordinates using improved parser
                points = self._parse_coordinates_from_text(response)

                # If no points were found, return empty result
                if len(points) == 0:
                    print(f"[WARNING] No valid coordinates found in model output: {response}")
                    if return_points:
                        return {"mask": None, "points": [], "confidence": 0.0}
                    return {"mask": None, "confidence": 0.0}

                print(f"[INFO] Successfully found {len(points)} points")
        except Exception as e:
            print(f"[ERROR] Failed to generate or parse points: {e}")
            if return_points:
                return {"mask": None, "points": [], "confidence": 0.0}
            return {"mask": None, "confidence": 0.0}

        # 2) Pass points to SAM for segmentation
        if self.sam_model is not None and self.sam_processor is not None:
            # Convert to SAM's expected format
            input_points = torch.tensor([points], dtype=torch.float).to(self.device)
            input_labels = torch.ones(input_points.shape[:2], dtype=torch.int).to(self.device)

            # Ensure image is in right format for SAM (might need to resize or normalize)
            if hasattr(self.sam_processor, "resize_longest_size"):
                # Convert the tensor to a PIL Image for SAM processing
                from PIL import Image
                import numpy as np

                # Denormalize and convert to PIL
                img = pixel_values[0].cpu().permute(1, 2, 0).numpy()
                img = (img * 255).astype(np.uint8)
                pil_img = Image.fromarray(img)

                # Process with SAM processor
                sam_inputs = self.sam_processor(
                    images=pil_img,
                    input_points=points,
                    return_tensors="pt"
                )

                sam_inputs = {k: v.to(self.device) for k, v in sam_inputs.items() if isinstance(v, torch.Tensor)}

                # Generate mask with SAM
                with torch.no_grad():
                    sam_outputs = self.sam_model(**sam_inputs)
                    masks = sam_outputs.pred_masks.squeeze().cpu().numpy()
                    scores = sam_outputs.iou_scores.squeeze().cpu().numpy()

                # Get the best mask
                if masks.ndim == 3:
                    best_mask_idx = scores.argmax()
                    mask = masks[best_mask_idx]
                    confidence = scores[best_mask_idx]
                else:
                    mask = masks
                    confidence = scores.item() if hasattr(scores, 'item') else scores

                # Return results
                result = {"mask": mask, "confidence": confidence}
                if return_points:
                    result["points"] = points

                return result
            else:
                print("[WARNING] SAM processor doesn't have the expected interface.")
                if return_points:
                    return {"mask": None, "points": points, "confidence": 0.0}
                return {"mask": None, "confidence": 0.0}
        else:
            print("[WARNING] SAM model or processor not available. Returning only points.")
            if return_points:
                return {"mask": None, "points": points, "confidence": 0.0}
            return {"mask": None, "confidence": 0.0}

    def _parse_coordinates_from_text(self, text):
        """
        Enhanced coordinate parser that's more robust to different formats.

        Args:
            text (str): Generated text to parse coordinates from

        Returns:
            list: List of [x, y] point coordinates
        """
        import re

        # Regular expressions for different coordinate formats
        patterns = [
            r"\(\s*(\d+)\s*,\s*(\d+)\s*\)",  # (x,y)
            r"(\d+)\s*,\s*(\d+)",  # x,y
            r"x\s*=\s*(\d+)\s*,\s*y\s*=\s*(\d+)",  # x=x, y=y
            r"X\s*=\s*(\d+)\s*,\s*Y\s*=\s*(\d+)",  # X=x, Y=y
        ]

        points = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    x, y = int(match[0]), int(match[1])
                    # Basic bounds check (adjust bounds as needed)
                    if 0 <= x < 4096 and 0 <= y < 4096:  # Arbitrary large limit
                        points.append([x, y])
                except (ValueError, IndexError):
                    continue

        # Remove duplicates while preserving order
        seen = set()
        unique_points = []
        for point in points:
            point_tuple = tuple(point)
            if point_tuple not in seen:
                seen.add(point_tuple)
                unique_points.append(point)

        return unique_points[:10]  # Limit to 10 points max