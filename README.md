# DI725 Project: Quantized PaliGemma with Adapters

This repository contains code and experiments for training and evaluating **quantized PaliGemma models with Adapters** with the PaliGemma model. 
## Project Purpose

- Train and evaluate PaliGemma with and without adapters
- Develop and benchmark **Quantized Vision Language Adapters (QVLA)**
- Enable efficient, memory-friendly multimodal model adaptation

## Repository Structure

```
base_paligemma.ipynb           # Train & evaluate the base PaliGemma model
vision_adapter_paligemma.ipynb # Train & evaluate PaliGemma with Adapters
qvla_paligemma.ipynb           # Train & evaluate QVLA (Quantized Vision Language Adapters)
process_dataset.py             # Prepare RISCM for training
dataset_visualizations.ipynb   # Visualize dataset samples and statistics
model_addons.py                # Utilities to make PaliGemma compatible with the adapters library
RISCM/data_readme.txt          # Instructions for downloading the RISCM dataset
...
```
