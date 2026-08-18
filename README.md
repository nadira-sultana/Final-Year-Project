# 🩻 Multimodal AI Medical X-Ray Assistant

An AI-powered multimodal diagnostic assistant that analyzes chest radiographs (X-rays) and answers clinical questions using Vision-Language Models (VLMs) and parameter-efficient fine-tuning (LoRA / QLoRA).

## 🌟 Key Features
- **Multimodal Understanding:** Combines computer vision (ViT) and language generation to interpret X-ray plates.
- **Efficient Fine-Tuning:** Uses 4-bit quantization and LoRA to train on Google Kaggle GPUs.
- **Interactive UI:** Web interface powered by Gradio for real-time radiograph upload and Q&A.

## 🛠️ Tech Stack
- **Framework:** PyTorch, Hugging Face Transformers
- **Training Optimization:** PEFT, TRL (Transformer Reinforcement Learning), BitsAndBytes
- **Dataset:** VQA-RAD / Chest X-Ray Images
- **UI:** Gradio

## 📋 Project Structure
