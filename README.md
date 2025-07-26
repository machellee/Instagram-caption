# Instagram Captioning with BLIP 🎆

This repository contains code and models for generating Instagram-style image captions using a fine-tuned BLIP model hosted on Hugging Face. 📷

## About This Project

This repository contains the final project for my Large Language Models (LLM) course at Yonsei University. The goal was to fine-tune a BLIP model for Instagram-style caption generation.

Course: Large Language Models (LLM)  
Term: Spring 2025 at Yonsei Univeristy

---

## Setup Instructions

### 1. Clone the repository

git clone https://github.com/machellee/Instagram-caption.git
cd Instagram-caption

### 2. Install dependencies

Install all required packages, including PyTorch with CUDA support:

pip install -r requirements.txt
pip install torch==2.7.0+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

### 3. (Optional) Install WandB for experiment tracking

pip install wandb==0.19.9

---

## Usage

### Load the pre-trained BLIP model from Hugging Face

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests

processor = BlipProcessor.from_pretrained("mle1/instagram_caption_blip")
model = BlipForConditionalGeneration.from_pretrained("mle1/instagram_caption_blip")

image_url = "https://example.com/your-image.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt")
outputs = model.generate(**inputs)
caption = processor.decode(outputs[0], skip_special_tokens=True)

print("Generated caption:", caption)

---

## Training

The BLIP model was fine-tuned on a huggingface datasets of Instagram image-caption pairs.

If you want to reproduce the training, please refer to the `training_notebook_cleaned.ipynb` or the `train_model.py` script in this repo.

---
## Requirements

See `requirements.txt` for the full list of packages. Some key dependencies include:

- transformers (4.51.3)
- timm
- Pillow
- datasets
---

