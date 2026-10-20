# QWEN_JEV: System-One Calibrated Decision Model (Qwen3-0.6B + Custom Head + RLCB)

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-QWEN__JEV-yellow)](https://huggingface.co/Piyush-5541/QWEN_JEV/tree/main)
[![Hugging Face SFT](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-bev--decision--lora-orange)](https://huggingface.co/Piyush-5541/bev-decision-lora/tree/main)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-QWEN__JEV__Dataset-blue)](https://huggingface.co/datasets/Piyush-5541/QWEN_JEV_Dataset)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**QWEN_JEV** is a fast, single-pass decision, verification, and rubric-scoring architecture built on a truncated **`Qwen/Qwen3-0.6B`** backbone with a custom 2-layer self-attention decision head (`head.pt`), rank-8 LoRA adapters (`lora/`), and **RLCB (Reinforcement Learning for Calibrated Decisions)** post-training.

Instead of relying on slow, uncalibrated autoregressive token generation, **QWEN_JEV** encodes the state, question, and all candidate options in a single forward pass using **parallel position IDs** and a **2D block-causal attention mask**, outputting calibrated probability distributions across three task contracts:
- **`choice` (0):** Multi-option selection (`Choose the best option.`)
- **`noul` (1):** Binary Yes/No verification (`Answer yes or no.`)
- **`score` (2):** Ordinal rubric evaluation (`Rate it on the given scale.`)

---

## 🔗 Open-Source Releases

- **Final RLCB-Calibrated Model:** [`Piyush-5541/QWEN_JEV`](https://huggingface.co/Piyush-5541/QWEN_JEV/tree/main)
- **Stage-1 SFT Model:** [`Piyush-5541/bev-decision-lora`](https://huggingface.co/Piyush-5541/bev-decision-lora/tree/main)
- **Curated Dataset:** [`Piyush-5541/QWEN_JEV_Dataset`](https://huggingface.co/datasets/Piyush-5541/QWEN_JEV_Dataset)
- **Author GitHub:** [piyushofficialpatel](https://github.com/piyushofficialpatel?tab=repositories)

---

## 🧠 Model Architecture & Parameter Breakdown

Standard `Qwen/Qwen3-0.6B` contains 28 transformer layers (~596.05M parameters). **QWEN_JEV** truncates the backbone to the first **20 transformer blocks** (`layers.0` through `layers.19`), freezes the base weights, attaches **LoRA (`r=8, alpha=16`)** to the last 4 blocks (`layers.16`–`19`), and routes extracted option and answer representations into a custom **2-layer Decision Head**.

### 1. 2D Block-Causal Attention Mask & Position Reset
When scoring $K$ candidate options `<option> opt_0 </option> ... <option> opt_K </option> The answer is:`:
1. **Order-Invariant Position IDs:** Each option resets its position counter to start immediately after the question/state tokens (`q_end + 1`), eliminating positional bias between early and late options.
2. **Option Isolation Mask:** Each option attends causally to the question/state and to its own tokens, but is masked (`-inf`) from seeing other options.
3. **Global Readout:** The trailing `"The answer is:"` tokens attend across the question and all options. Summary vectors are extracted at the final token of each option (`choice_read_idx`) and the final answer token (`answer_end_idx`).

### 2. Exact Parameter Count (Before & After RLCB)

| Component | Configuration | Exact Parameters | Status |
| :--- | :--- | :--- | :--- |
| **Token Embeddings** (`embed_tokens`) | $151,936 \times 1,024$ | `155,582,464` | Frozen |
| **Qwen3 Transformer Blocks** (`layers.0` – `19`) | 20 blocks $\times 15,730,944$ | `314,618,880` | Frozen |
| **Backbone Final RMSNorm** (`norm`) | $1,024$ | `1,024` | Frozen |
| **LoRA Adapters** (`lora/` on Blocks `16`–`19`) | $r=8, \alpha=16$ on `q,k,v,o_proj` | **`327,680`** | **Trainable** |
| **Custom Decision Head** (`head.pt`) | 2 SA Blocks ($h=1024, d=512, \text{tasks}=3$) | **`7,361,536`** | **Trainable** |
| **Total Active Parameters** | **20-Layer Backbone + LoRA + Head** | **`477,891,584` (~477.89M)** | **100%** |
| **Total Trainable Parameters** | **LoRA + Custom Decision Head** | **`7,689,216` (~7.69M)** | **1.61%** |

*(Note: Both Stage-1 SFT and Stage-2 RLCB have the exact same **477,891,584** total active parameter count and **7,689,216** trainable parameter footprint.)*

---

## 🎯 Two-Stage Training Pipeline & Evaluation Results

| Stage | Training Samples | Test Samples | Batch Size | Optimization Objective |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1: SFT** | **30,000** | **6,302** | `4` | Cross-Entropy Loss |
| **Stage 2: RLCB** | **12,000** | **4,000** | `4` | Brier Score + LOO Bandit + KL Anchor ($\beta = 0.04$) |

### Stage 1: Supervised Fine-Tuning (SFT)
Trained on **30,000** structured decision examples and evaluated on **6,302** held-out test questions (`batch_size=4`, `lr_lora=1e-4`, `lr_head=1e-3`, cosine schedule with 5% warmup).

| Metric / Split | Score |
| :--- | :--- |
| **Overall Test Accuracy** | **58.46%** |
| **Binary Verification Accuracy (`accuracy_noul`)** | **66.81%** |
| **Multi-Choice Accuracy (`accuracy_choice`)** | **59.01%** |
| **Multi-Point Rubric Accuracy (`accuracy_score`)** | **46.21%** |

### Stage 2: RLCB (Reinforcement Learning for Calibrated Decisions)
Standard Cross-Entropy loss drives winning logits toward $+\infty$, resulting in overconfident probabilities on ambiguous queries. In Stage 2, we post-train the SFT checkpoint on **12,000** training decisions and evaluate on **4,000** test decisions (`batch_size=4`) using **RLCB**:
1. **Strictly Proper Brier Score Objective:**
   $$R(u_\theta(x), y) = 2u_{\theta, y}(x) - \sum_{k=1}^{K} u_{\theta, k}(x)^2 = 1 - \Vert{}u_\theta(x) - e_y\Vert{}_2^2$$
2. **KL-Divergence Reference Anchor ($\beta = 0.04$):** Penalizes drift from the Stage-1 SFT distribution $u_{\text{ref}}(x)$ using memory-efficient dual PEFT adapter toggling (`sft_ref` vs. `default`).
3. **Contextual Bandit Policy Gradient (Leave-One-Out Baseline):** Samples $M=4$ actions per state and applies a variance-reduced Leave-One-Out (LOO) baseline without standard-deviation normalization, preserving proper score calibration.

---

## 🚀 Quickstart Inference (`Piyush-5541/QWEN_JEV`)

```bash
pip install torch transformers peft huggingface_hub
```

```python
import json, math, torch
import torch.nn as nn
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, Qwen3Model
from peft import PeftModel

REPO_ID = "Piyush-5541/QWEN_JEV"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
TASK_TYPES = {"choice": 0, "noul": 1, "score": 2}
PROMPTS = {"choice": "Choose the best option.", "score": "Rate it on the given scale.", "noul": "Answer yes or no."}

class SA(nn.Module):
    def __init__(self, h, d):
        super().__init__(); self.d = d; self.q = nn.Linear(h, d); self.k = nn.Linear(h, d); self.v = nn.Linear(h, d); self.o = nn.Linear(d, h)
    def forward(self, x, valid):
        w = torch.softmax((torch.einsum("bqd,bkd->bqk", self.q(x), self.k(x)) / math.sqrt(self.d)).masked_fill(~valid[:, None, :], -float("inf")), dim=-1)
        return self.o(torch.einsum("bqk,bkd->bqd", w, self.v(x)))

class Block(nn.Module):
    def __init__(self, h, d):
        super().__init__(); self.n1 = nn.LayerNorm(h); self.a = SA(h, d); self.n2 = nn.LayerNorm(h); self.m = nn.Sequential(nn.Linear(h, d), nn.GELU(), nn.Linear(d, h))
    def forward(self, x, v):
        x = x + self.a(self.n1(x), v); return x + self.m(self.n2(x))

class Head(nn.Module):
    def __init__(self, h, new_dim, num_layers, num_task_types):
        super().__init__(); self.d = new_dim; self.task_embedding = nn.Embedding(num_task_types, h); self.norm = nn.LayerNorm(h)
        self.blocks = nn.ModuleList([Block(h, new_dim) for _ in range(num_layers)]); self.final = nn.LayerNorm(h); self.ap = nn.Linear(h, new_dim); self.cp = nn.Linear(h, new_dim)
    def forward(self, choices, answer, choice_mask, task):
        x = self.norm(torch.cat([self.task_embedding(task).unsqueeze(1), choices, answer.unsqueeze(1)], 1))
        valid = torch.ones((len(x), x.shape[1]), dtype=torch.bool, device=x.device)
        for b in self.blocks: x = b(x, valid)
        x = self.final(x)
        return torch.einsum("bd,bcd->bc", self.ap(x[:, -1]), self.cp(x[:, 1:-1])) / math.sqrt(self.d)

folder = Path(snapshot_download(repo_id=REPO_ID))
meta = json.loads((folder / "meta.json").read_text())
tok = AutoTokenizer.from_pretrained(meta["model_name"])

base = Qwen3Model.from_pretrained(meta["model_name"], dtype=torch.float32, num_hidden_layers=meta["num_layers"])
if (folder / "lora").exists():
    base = PeftModel.from_pretrained(base, folder / "lora")
head = Head(base.config.hidden_size, **meta["head_config"])
head.load_state_dict(torch.load(folder / "head.pt", map_location="cpu"))
base.to(DEVICE).eval(); head.to(DEVICE).eval()
```

---

## 🙏 Acknowledgements

Special thanks to **Avik Biswas (`avbiswas`)** for creating and open-sourcing the foundational **`avbiswas/bev-decision`(https://huggingface.co/datasets/avbiswas/bev-decision/viewer/default/train)** dataset and BEV training framework, which served as the foundation for curating [`Piyush-5541/QWEN_JEV_Dataset`](https://huggingface.co/datasets/Piyush-5541/QWEN_JEV_Dataset) (30,000 train / 6,302 test SFT split and 12,000 train / 4,000 test RLCB split) and developing the **QWEN_JEV** pipelines.
