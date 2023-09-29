# Model Fine-Tuning Lab with Google Colab & Local Integration

## Goal

This repository houses various notebooks and scripts aimed at exploring, integrating, and fine-tuning state-of-the-art models such as BNB 4-bit, Falcon-7B, GPT-NeoX 20B, Bloom560M, and GPT-2. The overarching goal is to demonstrate the capabilities of these models, experiment with different techniques, and instantiate models locally and on Google Colab.

## Overview

The repository contains a diverse set of notebooks each focusing on a specific model or technique, such as 4-bit quantization, fine-tuning with English quotes, language model tagger fine-tuning, integration with Falcon-7B, and local instantiation of GPT-2. Additionally, the repository contains Python scripts for loading and utilizing GPT-2 models locally.

## Features

- **4-bit Model Integration**: Integration of \`transformers\` with \`bitsandbytes\` for 4-bit quantization, showcasing model loading and conversion.
- **Falcon-7B Exploration**: Comprehensive exploration and usage of the Falcon-7B model, including instantiation, pipeline setup, and inference.
- **Fine-Tuning GPT-NeoX 20B**: Fine-tuning a GPT-NeoX 20B model with English quotes using techniques from the QLoRA paper.
- **PEFT Fine-Tune Bloom560M Tagger**: Utilization of PEFT & bitsandbytes to fine-tune a LoRa checkpoint named "Bloom560M" for tagging purposes.
- **GPT-2 Local Instantiation**: Python script for loading a pre-trained GPT-2 model and tokenizer from Hugging Face locally.

## Notebooks & Scripts

1. **BNB_4bit_integration.ipynb**
   - Focuses on integrating 4-bit quantization techniques introduced in the [QLoRA paper](https://arxiv.org/abs/2305.14314) with \`transformers\` and \`bitsandbytes\`.
   - Demonstrates how to load and convert models in 4-bit for inference.

2. **falcon_with_langchain.ipynb**
   - Explores the functionalities of Falcon-7B, providing a practical guide on how to get started with Falcon-7B for tasks like inference, fine-tuning, and quantization.
   - Details the model's architecture, training data, and procedures, along with considerations related to bias, risks, and limitations.

3. **finetuning_QLORA_gpt_neox_20b_with_english_quotes.ipynb**
   - Fine-tunes a GPT-NeoX 20B model with English quotes, guiding through the process of loading quotes, running training, and model evaluation.

4. **PEFT_Finetune_Bloom560M_tagger.ipynb**
   - Demonstrates the fine-tuning of the Bloom560M model using PEFT & bitsandbytes.
   - The fine-tuned model is available on Hugging Face: [Bloom560M LoRa Tagger](https://huggingface.co/Omid-sar/bloom-560M-lora-tagger).

5. **gpt2_model_local.py**
   - Python script for instantiating GPT-2 models from Hugging Face locally, including loading, tokenizing, generating sequences, and saving models.

## Prerequisites

- Python 3.x
- Hugging Face Transformers
- bitsandbytes
- PyTorch 2.0 (for Falcon LLMs)

## Setup and Installation

- Clone the repository: \`git clone https://github.com/your-username/odel_Finetuning_Lab_GoogleColab_Local.git\`
- Navigate to the repository: \`cd odel_Finetuning_Lab_GoogleColab_Local\`
- Install the required libraries: \`pip install -r requirements.txt\`

## License

This project is licensed under the MIT License - see the LICENSE file for details.
