

# Layered Image Generation with Lightweight Diffusion

This project explores **layered/compositional image generation using diffusion models**, focusing on efficient, modular implementations suitable for academic or resource-constrained environments. Instead of training models from scratch, we leverage **pretrained diffusion backbones (e.g., Stable Diffusion 1.5)** and implement **low-cost conditioning strategies** to control individual semantic layers such as backgrounds, objects, textures, and layouts.

---

## Project Overview

This project aims to develop a **layer-wise controllable generation pipeline** using techniques like **masking, inpainting, and ControlNet-based conditioning**. The workflow allows users to construct images in **stages (e.g., background → midground → foreground)** for greater interpretability and creative flexibility.

### Core Objectives

* Build a **compositional generation pipeline** using pretrained diffusion models.
* Implement **layer-wise control** via masks, semantic maps, or sketches with minimal/no finetuning.
* Use **plug-and-play modules** such as ControlNet, LoRA adapters, or inpainting-based strategies.
* Provide **qualitative examples and ablation studies** demonstrating effectiveness.

---

## 🧩 Key Features

* **Layered Generation**: Stage-wise control of image composition.
* **ControlNet Integration**: For edge maps, segmentation, or pose conditioning.
* **Inpainting Support**: Modify or replace individual layers easily.
* **Lightweight Adaptation**: Optional LoRA/adapters for custom styles.
* **Reproducible Workflow**: Works on Google Colab or standard lab GPUs.

---




## 🛠 Installation & Setup

### Prerequisites

* Python 3.9+
* GPU with 12–24 GB VRAM (T4, A100, RTX 3090 recommended)
* [Hugging Face account](https://huggingface.co) (for model access)

### Install Dependencies

```bash
git clone https://github.com/your-username/Layered-Image-Generation-with-Lightweight-Diffusion.git
cd Layered-Image-Generation-with-Lightweight-Diffusion
pip install -r requirements.txt
```

### Required Packages

* `diffusers`
* `transformers`
* `torch`
* `controlnet_aux`
* `gradio` (optional for demo)
* `xformers` or `bitsandbytes` (optional for memory efficiency)

---

## 📊 Usage

### Example: Layered Image Generation

```python
from src.pipeline import LayeredDiffusionPipeline

pipeline = LayeredDiffusionPipeline(model="stabilityai/stable-diffusion-1-5")

# Stage 1: Generate background
bg = pipeline.generate_layer(prompt="sunset over the mountains")

# Stage 2: Add midground forest
mid = pipeline.add_layer(image=bg, prompt="dense pine forest")

# Stage 3: Add hiker in foreground
final = pipeline.add_layer(image=mid, prompt="hiker with backpack standing on a cliff")
final.save("examples/final_image.png")
```



## 🧠 Methods & Components

* **Base Models**: Stable Diffusion 1.5, ControlNet-compatible versions.
* **Conditioning**: Masks, semantic maps, edge/sketch guides.
* **Layer Composition**: Inpainting-based compositional control.
* **Optional Fine-tuning**: LoRA/adapters for style or custom layers.


