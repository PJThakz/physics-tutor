import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

st.set_page_config(
    page_title="Physics Tutoring Assistant",
    page_icon="⚡",
    layout="centered"
)

PROMPT_TEMPLATE = """Below is a physics problem. Solve it step by step.

### Problem:
{}

### Solution:
"""

@st.cache_resource
def load_model():
    with st.spinner("Loading model... (this takes ~2 minutes on first load)"):
        tokenizer = AutoTokenizer.from_pretrained(
            "pranjalthakz/physics-tutor-lora"
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct",
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        model = PeftModel.from_pretrained(
            base_model,
            "pranjalthakz/physics-tutor-lora"
        )
        model.eval()
    return model, tokenizer

def solve(question, model, tokenizer):
    prompt = PROMPT_TEMPLATE.format(question.strip())
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("⚡ Physics Tutoring Assistant")
st.markdown("""
Fine-tuned **Qwen2.5-1.5B-Instruct** on 123 verified step-by-step physics problems.
Covers kinematics, forces, Newton's laws, and introductory mechanics.

> ⏳ First response takes ~60 seconds on CPU. Subsequent ones are faster.
""")

model, tokenizer = load_model()

st.markdown("### Enter a physics problem")

examples = [
    "A car accelerates from rest at 3 m/s² for 8 seconds. Find the distance travelled.",
    "A ball is thrown upward at 20 m/s. Find the maximum height. (g = 10 m/s²)",
    "A 5 kg block is pushed with 30 N on a frictionless surface. Find the acceleration.",
    "A car brakes from 25 m/s to rest in 5 seconds. Find the deceleration.",
    "A 10 kg box is pulled with 50 N. Friction opposes with 12 N. Find the acceleration.",
]

st.markdown("**Try an example:**")
cols = st.columns(2)
for i, ex in enumerate(examples):
    if cols[i % 2].button(ex[:55] + "...", key=f"ex_{i}"):
        st.session_state["question"] = ex

question = st.text_area(
    "Problem",
    value=st.session_state.get("question", ""),
    height=100,
    placeholder="e.g. A stone is dropped from a 45 m tower. How long to reach the ground? (g = 10 m/s²)"
)

if st.button("Solve ⚡", type="primary"):
    if question.strip():
        with st.spinner("Solving..."):
            answer = solve(question, model, tokenizer)
        st.markdown("### Step-by-step Solution")
        st.text(answer)
    else:
        st.warning("Please enter a physics problem first.")

st.markdown("---")
st.markdown(
    "Built by [Pranjal Thakur](https://linkedin.com/in/pranjal-thakur-17433427b) · "
    "Model: [pranjalthakz/physics-tutor-lora](https://huggingface.co/pranjalthakz/physics-tutor-lora)"
)
