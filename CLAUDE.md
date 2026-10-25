# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands
- **Run App**: `source venv/bin/activate && streamlit run app.py`
- **Run App (Headless)**: `source venv/bin/activate && streamlit run app.py --server.headless true`
- **Install Dependencies**: `source venv/bin/activate && pip install -r requirements.txt`

## Architecture
The project implements a specialized decision-making model called **QWEN_JEV Decision Model**, based on a calibrated RL (RLCB) approach.

### Components
- **Backbone**: Uses `Qwen/Qwen3-0.6B` as the base language model.
- **LoRA Adapters**: Specialized adapters are applied to the backbone (specifically blocks 16–19) to adapt the model for decision tasks.
- **Custom Decision Head**: A specialized `Head` class (implemented in `app.py`) that takes representations from the backbone and outputs probabilities for different task types.
- **Task Types**: Supports three primary tasks: `choice`, `noul` (Yes/No), and `score`.

### Model Loading Flow
1. **Meta-data**: Loads `final/meta.json` to determine model name and head configuration.
2. **Base Model**: Loads the Qwen backbone with specified precision (typically float16/float32).
3. **Adapters**: Loads LoRA weights from `final/adapter_model.safetensors`.
4. **Head**: Initializes the `Head` class and loads weights from `final/head.pt`.

## Constraints & Stability
- **Hardware**: On macOS, the model may experience segmentation faults or connection errors when using `device_map="auto"` or MPS. For maximum stability, the model should be forced to `DEVICE = "cpu"`.
- **Dependencies**: Requires `torch`, `transformers`, `peft`, and `streamlit`.
