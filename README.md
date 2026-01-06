# LoRaMed: QLoRA Fine Tuning for Medical QA

Training repository for fine-tuning Gemma-2B on PubMedQA using QLoRA.

## Setup

```bash
pip install transformers datasets accelerate peft evaluate bitsandbytes sentencepiece torch numpy
```

Authenticate with Hugging Face:
```python
from huggingface_hub import login
login()
```

## Training

Run `loramed.ipynb` sequentially. The notebook:
1. Loads and quantizes Gemma-2B (4-bit NF4)
2. Configures LoRA adapters (r=16, alpha=32)
3. Loads PubMedQA dataset (5k train, 1k val)
4. Trains with QLoRA
5. Evaluates and saves model

## Configuration

- **Model**: `google/gemma-2b-it`
- **Quantization**: 4-bit NF4 with double quantization
- **LoRA**: r=16, alpha=32, dropout=0.05
- **Training**: 2 epochs, lr=2e-4, batch=16 (effective)
- **Dataset**: PubMedQA (yes/no/maybe classification)

## Outputs

- Checkpoints: `gemma2-pubmedqa-qlora/`
- Final model: `gemma2-pubmedqa-qlora-final/`

## Resources

- Dataset: [qiaojin/PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA)
- Model: [google/gemma-2b-it](https://huggingface.co/google/gemma-2b-it)
