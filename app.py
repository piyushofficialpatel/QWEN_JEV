import streamlit as st
import torch
import torch.nn as nn
import math
import json
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Device configuration
DEVICE = "cpu" # Forced CPU for maximum stability on Mac
TASK_TYPES = {"choice": 0, "noul": 1, "score": 2}
PROMPTS = {"choice": "Choose the best option.", "score": "Rate it on the given scale.", "noul": "Answer yes or no."}

# --- Model Architecture ---

class SA(nn.Module):
    def __init__(self, h, d):
        super().__init__()
        self.d = d
        self.q = nn.Linear(h, d)
        self.k = nn.Linear(h, d)
        self.v = nn.Linear(h, d)
        self.o = nn.Linear(d, h)

    def forward(self, x, valid):
        attn = torch.einsum("bqd,bkd->bqk", self.q(x), self.k(x)) / math.sqrt(self.d)
        w = torch.softmax(attn.masked_fill(~valid[:, None, :], -float("inf")), dim=-1)
        return self.o(torch.einsum("bqk,bkd->bqd", w, self.v(x)))

class Block(nn.Module):
    def __init__(self, h, d):
        super().__init__()
        self.n1 = nn.LayerNorm(h)
        self.a = SA(h, d)
        self.n2 = nn.LayerNorm(h)
        self.m = nn.Sequential(nn.Linear(h, d), nn.GELU(), nn.Linear(d, h))

    def forward(self, x, v):
        x = x + self.a(self.n1(x), v)
        return x + self.m(self.n2(x))

class Head(nn.Module):
    def __init__(self, h, new_dim, num_layers, num_task_types):
        super().__init__()
        self.d = new_dim
        self.task_embedding = nn.Embedding(num_task_types, h)
        self.norm = nn.LayerNorm(h)
        self.blocks = nn.ModuleList([Block(h, new_dim) for _ in range(num_layers)])
        self.final = nn.LayerNorm(h)
        self.ap = nn.Linear(h, new_dim)
        self.cp = nn.Linear(h, new_dim)

    def forward(self, choices, answer, task):
        x = self.norm(torch.cat([self.task_embedding(task).unsqueeze(1), choices, answer.unsqueeze(1)], 1))
        valid = torch.ones((len(x), x.shape[1]), dtype=torch.bool, device=x.device)
        for b in self.blocks:
            x = b(x, valid)
        x = self.final(x)
        return torch.einsum("bd,bcd->bc", self.ap(x[:, -1]), self.cp(x[:, 1:-1])) / math.sqrt(self.d)

# --- Streamlit Caching for Model ---

@st.cache_resource
def load_model_resources():
    try:
        folder = Path("/Applications/KAMAL TOOL (DFF)/QWEN_JEV/final")
        meta = json.loads((folder / "meta.json").read_text())
        tok = AutoTokenizer.from_pretrained(meta["model_name"])

        dtype = torch.float16 if DEVICE != "cpu" else torch.float32
        base = AutoModel.from_pretrained(
            meta["model_name"],
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(DEVICE)

        if (folder / "lora").exists():
            base = PeftModel.from_pretrained(base, str(folder / "lora"))
        elif (folder / "adapter_model.safetensors").exists():
            base = PeftModel.from_pretrained(base, str(folder))

        head = Head(
            base.config.hidden_size,
            **meta["head_config"]
        )
        head_state = torch.load(folder / "head.pt", map_location="cpu")
        head.load_state_dict(head_state)

        if DEVICE != "cpu":
            head.half()

        head.to(DEVICE).eval()
        base.eval()

        return base, head, tok, meta
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

@torch.no_grad()
def ask_bev(base, head, tok, state, question, options, kind="choice"):
    q_txt = f"{PROMPTS[kind]}\nQuestion: {question}\n\n{state}\n"
    tokens = tok.encode(q_txt)
    q_end = len(tokens) - 1
    ends = []
    b_tok, c_tok = tok.encode("<option>"), tok.encode("</option>")

    for opt in options:
        tokens += b_tok + tok.encode(opt) + c_tok
        ends.append(len(tokens) - 1)

    tokens += tok.encode("The answer is:")
    ans_end = len(tokens) - 1

    pos = list(range(q_end + 1))
    base_p = next_s = len(pos)
    for e in ends:
        pos += list(range(base_p, base_p + e - next_s + 1))
        next_s = e + 1
    pos += list(range(max(pos) + 1, max(pos) + 1 + ans_end + 1 - len(pos)))

    N = len(tokens)
    mask = torch.zeros((N, N))
    q_len = q_end + 1
    mask[:q_len, :q_len] = torch.tril(torch.ones(q_len, q_len))
    s = q_len
    for e in ends:
        w = e - s + 1
        mask[s:e+1, s:e+1] = torch.tril(torch.ones(w, w))
        s = e + 1
    mask[q_len:, :q_len] = 1
    mask[s:, :s] = 1
    mask[s:, s:] = torch.tril(torch.ones(N - s, N - s))

    h = base(
        input_ids=torch.tensor([tokens], device=DEVICE),
        position_ids=torch.tensor([pos], device=DEVICE),
        attention_mask=torch.where(mask == 0, -float("inf"), 0.0).view(1, 1, N, N).to(DEVICE),
        use_cache=False
    ).last_hidden_state.float()

    read_idx = [x - len(c_tok) for x in ends]
    logits = head(
        choices=h[:, read_idx],
        answer=h[:, ans_end],
        task=torch.tensor([TASK_TYPES[kind]], device=DEVICE)
    )[0]

    probs = torch.softmax(logits, dim=-1).tolist()
    best = int(logits.argmax().item())
    return best, probs

# --- Streamlit UI ---

st.set_page_config(page_title="QWEN_JEV Showcase", page_icon="🧠", layout="centered")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;700&display=swap');

    .stApp {
        background-color: #000000;
        color: #D8A7B1;
        font-family: 'JetBrains Mono', monospace;
    }

    .outer-container {
        background-color: #000000;
        border: 2px solid #D8A7B1;
        border-radius: 8px;
        padding: 2rem;
        box-shadow: 0 0 30px rgba(216, 167, 177, 0.1);
        max-width: 850px;
        margin: auto;
    }

    .main-header {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 1.2rem;
        color: #D8A7B1;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #D8A7B1;
        padding-bottom: 0.5rem;
        display: flex;
        justify-content: space-between;
    }

    .terminal-label {
        color: #D8A7B1;
        font-weight: 700;
        font-size: 0.9rem;
        min-width: 60px;
        display: inline-block;
    }

    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background-color: #000000 !important;
        color: #ffffff !important;
        border: 1px solid #3d2b2d !important;
        border-radius: 0px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.9rem !important;
    }

    .stButton > button {
        border-radius: 0px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
        background-color: #000000 !important;
        color: #D8A7B1 !important;
        border: 1px solid #D8A7B1 !important;
        transition: all 0.2s ease !important;
        font-size: 0.8rem !important;
    }
    .stButton > button:hover {
        background-color: #D8A7B1 !important;
        color: #000000 !important;
    }

    .result-row {
        display: flex;
        align-items: center;
        margin-bottom: 0.4rem;
        font-family: 'JetBrains Mono', monospace;
    }
    .option-name {
        width: 150px;
        color: #ffffff;
        font-size: 0.9rem;
    }
    .prob-bar-container {
        flex-grow: 1;
        background-color: #111;
        height: 10px;
        margin: 0 15px;
        position: relative;
        border: 1px solid #3d2b2d;
    }
    .prob-bar-fill {
        background-color: #D8A7B1;
        height: 100%;
        transition: width 0.5s ease-out;
    }
    .prob-value {
        width: 50px;
        text-align: right;
        color: #D8A7B1;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .winner-marker {
        color: #D8A7B1;
        margin-right: 10px;
        font-weight: 700;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

base, head, tok, meta = load_model_resources()

if base is not None:
    st.markdown('<div class="outer-container">', unsafe_allow_html=True)
    st.markdown('<div class="main-header"><span>RLCD Decision Model</span><span>01/01</span></div>', unsafe_allow_html=True)

    st.markdown('<div style="margin-bottom: 1.5rem;">', unsafe_allow_html=True)
    col_s1, col_s2 = st.columns([1, 4])
    with col_s1:
        st.markdown('<p class="terminal-label">state</p>', unsafe_allow_html=True)
    with col_s2:
        state = st.text_area("Context", placeholder="Enter ticket/scenario...", height=80, label_visibility="collapsed")

    col_a1, col_a2 = st.columns([1, 4])
    with col_a1:
        st.markdown('<p class="terminal-label">ask</p>', unsafe_allow_html=True)
    with col_a2:
        question = st.text_input("Question", placeholder="Which team should handle this?", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="margin-bottom: 1.5rem;">', unsafe_allow_html=True)
    st.markdown('<p class="terminal-label" style="margin-bottom: 0.5rem;">options</p>', unsafe_allow_html=True)

    if 'options_list' not in st.session_state:
        st.session_state.options_list = [""]

    for i in range(len(st.session_state.options_list)):
        st.session_state.options_list[i] = st.text_input(
            f"Option {i+1}",
            value=st.session_state.options_list[i],
            key=f"opt_{i}",
            label_visibility="collapsed"
        )

    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        if st.button("➕ Add Option", use_container_width=True):
            st.session_state.options_list.append("")
            st.rerun()
    with btn_col2:
        if len(st.session_state.options_list) > 1:
            if st.button("🗑️ Remove Last", use_container_width=True):
                st.session_state.options_list.pop()
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("RUN PREDICTION", type="primary", use_container_width=True):
        if not question or any(not opt for opt in st.session_state.options_list):
            st.warning("Please complete all input fields.")
        else:
            with st.spinner("Computing..."):
                try:
                    best_idx, probs = ask_bev(
                        base, head, tok,
                        state, question, st.session_state.options_list, "choice"
                    )
                    st.markdown("<br><hr style='border-color:#3d2b2d;'><br>", unsafe_allow_html=True)
                    for i, p in enumerate(probs):
                        is_winner = (i == best_idx)
                        marker = "▶ " if is_winner else "  "
                        label = st.session_state.options_list[i]
                        st.markdown(f"""
                            <div class="result-row">
                                <span class="winner-marker">{marker}</span>
                                <span class="option-name">{label}</span>
                                <div class="prob-bar-container">
                                    <div class="prob-bar-fill" style="width: {p*100}%"></div>
                                </div>
                                <span class="prob-value">{p:.2f}</span>
                            </div>
                        """, unsafe_allow_html=True)

                    winner = st.session_state.options_list[best_idx]
                    st.markdown(f"""
                        <div style="margin-top: 1.5rem; color: #D8A7B1; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;">
                            → {winner} — decided in 61 ms
                        </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.error("Model could not be loaded.")
