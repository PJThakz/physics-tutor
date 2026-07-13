import streamlit as st
import requests
import os

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

API_URL = "https://api-inference.huggingface.co/models/pranjalthakz/physics-tutor-merged"
HF_TOKEN = st.secrets.get("HF_TOKEN", "")

def solve(question):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    prompt = PROMPT_TEMPLATE.format(question.strip())
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 500,
            "do_sample": False,
            "return_full_text": False,
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "No response generated.")
        return str(result)
    elif response.status_code == 503:
        return "⏳ Model is loading on HF servers (first request takes ~30s). Please try again in 30 seconds."
    else:
        return f"Error {response.status_code}: {response.text}"

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("⚡ Physics Tutoring Assistant")
st.markdown("""
Fine-tuned **Qwen2.5-1.5B-Instruct** on 123 verified step-by-step physics problems.
Covers kinematics, forces, Newton's laws, and introductory mechanics.
""")

examples = [
    "A car accelerates from rest at 3 m/s² for 8 seconds. Find the distance travelled.",
    "A ball is thrown upward at 20 m/s. Find the maximum height. (g = 10 m/s²)",
    "A 5 kg block is pushed with 30 N on a frictionless surface. Find the acceleration.",
    "A car brakes from 25 m/s to rest in 5 seconds. Find the deceleration.",
]

st.markdown("**Try an example:**")
cols = st.columns(2)
for i, ex in enumerate(examples):
    if cols[i % 2].button(ex[:50] + "...", key=f"ex_{i}"):
        st.session_state["question"] = ex

question = st.text_area(
    "Physics Problem",
    value=st.session_state.get("question", ""),
    height=100,
    placeholder="e.g. A stone is dropped from 45 m. How long to reach the ground? (g = 10 m/s²)"
)

if st.button("Solve ⚡", type="primary"):
    if question.strip():
        with st.spinner("Solving..."):
            answer = solve(question)
        st.markdown("### Step-by-step Solution")
        st.text(answer)
    else:
        st.warning("Please enter a physics problem first.")

st.markdown("---")
st.markdown(
    "Built by [Pranjal Thakur](https://linkedin.com/in/pranjal-thakur-17433427b) · "
    "Model: [pranjalthakz/physics-tutor-lora](https://huggingface.co/pranjalthakz/physics-tutor-lora)"
)
