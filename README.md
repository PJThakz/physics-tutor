# Physics Tutoring Assistant

Fine-tuned **Qwen2.5-1.5B-Instruct** using LoRA on a self-built dataset of
123 verified step-by-step physics problems. Increased fully correct structured
solutions from 3% to 97% on a held-out test set of 30 problems compared to
the base model.

## Live Demo
👉 [Try it here](https://your-app.streamlit.app)

## Project Structure
physics-tutor/
├── app.py                  # Streamlit demo app
├── requirements.txt        # Dependencies
├── dataset/
│   ├── physics_train.json  # 93 training examples
│   └── physics_test.json   # 30 held-out test examples
├── training/
│   └── train.ipynb         # Unsloth fine-tuning notebook
└── README.md

## Model
- **Base model:** Qwen/Qwen2.5-1.5B-Instruct
- **Fine-tuning method:** LoRA (r=16, alpha=16)
- **Training data:** 123 verified physics problems (Kinematics + Forces)
- **Training time:** ~18 minutes on Google Colab T4 GPU
- **Adapter:** [pranjalthakz/physics-tutor-lora](https://huggingface.co/pranjalthakz/physics-tutor-lora)

## Dataset
123 problems built from scratch across two topics:

| Topic | Count |
|---|---|
| Kinematics | 64 |
| Forces (Newton's Laws, friction, pulleys, momentum) | 59 |

Difficulty split: 41 Easy / 45 Medium / 37 Hard

Problems adapted from multiple sources (OpenStax, Cambridge IGCSE,
IB Physics, AP Physics) and rewritten into a unified 4-step tutoring format:

1. Identify known values
2. Choose the correct equation
3. Substitute values
4. Solve → Final Answer

## Evaluation Results

| Criterion | Base Model | Fine-tuned |
|---|---|---|
| Equation selected | 18/30 | 30/30 |
| Substitutions shown | 16/30 | 30/30 |
| Arithmetic shown | 19/30 | 30/30 |
| Units present | 18/30 | 29/30 |
| Final answer line | 6/30 | 30/30 |
| **Perfect solutions (5/5)** | **1/30 (3%)** | **29/30 (97%)** |

## How to Run Locally
pip install -r requirements.txt
streamlit run app.py

## Tech Stack
- Unsloth (2x faster LoRA fine-tuning)
- Hugging Face Transformers + PEFT
- Streamlit (demo UI)
- Google Colab T4 GPU (free tier)
