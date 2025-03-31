source .venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
pip install -U huggingface_hub
huggingface-cli download --resume-download Qwen/Qwen2.5-VL-3B-Instruct --local-dir Qwen25-VL-3B-Instruct
huggingface-cli download --resume-download facebook/sam-vit-base --local-dir sam-vit-base