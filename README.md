# CDT Dementia Screening

AI-powered Clock Drawing Test (CDT) analysis
for dementia screening using CCSS scoring standard.

## 🌐 Demo

[![Streamlit App]](https://cdt-dementia-screening-as7vhbgavmfsmc9np9o5r7.streamlit.app/)


## Dataset
NHATS Round 14 — not included
Request access: [nhats.org](https://www.nhats.org/data-access/nhats)

## 📊 Results

| Model | Sensitivity | Specificity | AUC |
|---|---|---|---|
| LLM Baseline | 0.600 | 0.967 | 0.873 |
| Vision Only (ViT) | 0.751 | 0.844 | 0.882 |
| Hybrid VQA | 0.760 | 0.850 | 0.887 |
| **Focal Loss (Best)** | **0.884** | **0.701** | **0.893** |

Target: Sensitivity ≥ 0.88, Specificity ≥ 0.82, AUC ≥ 0.91

## 🏗️ Architecture
Input Image (224×224)

↓

ViT Backbone (Frozen)

↓

Shared Layer (512 dims)

↓

5 Domain Heads (A/B/C/D/E)

↓

Total Score → if < 6 → Dementia Risk

## 📁 Structure
cdt-dementia-screening/

├── app.py                            ← Streamlit web app

├── requirements.txt                  ← Python dependencies

├── src/

│   ├── models.py                     ← CDTModel, CDTHybridModel

│   └── features.py                   ← VQA feature functions

├── results/

│   ├── 08_roc_curve.png

│   ├── 08_model_comparison.png

│   ├── 08_distribution_all.png

│   ├── 08_domain_accuracy_heatmap.png

│   ├── 08_final_summary.json

│   └── eda_*.png

└── notebooks/

├── 01_labeling_v1.ipynb          ← LLM labeling

├── 02_prompt_test.ipynb          ← ทดสอบ prompt

├── 03_check.ipynb                ← ตรวจสอบ labels

├── 04_EDA.ipynb                  ← EDA

├── 05_data_prep.ipynb            ← เตรียมข้อมูล

├── 06b_baseline_llm_human.ipynb  ← LLM + human labels

├── human_review_tool.ipynb       ← Human review interface

├── 07j_human_labels.ipynb        ← Human label review

├── 07k_undersample.ipynb         ← Undersample + ViT baseline

├── 07l_vqa_features.ipynb        ← Hybrid VQA features

├── 07m_hand_features.ipynb       ← Hybrid + Hand features

├── 07n_downsample_train.ipynb    ← Downsample class 2

├── 07o_binary_label.ipynb        ← Binary label experiment

├── 07p_generate_images.ipynb     ← SD Inpainting (failed)

├── 07q_finetune_sd.ipynb         ← LoRA Fine-tune (failed)

├── 07r_focal_loss.ipynb          ← Focal Loss

├── 07s_improved_features.ipynb   ← Improved VQA features v2

└── 08_evaluation.ipynb           ← Final evaluation


## Setup

### Colab
1. Open notebook in Google Colab
2. Set `ANTHROPIC_API_KEY` in Colab Secrets
3. Mount Google Drive
4. Update `BASE` path

### Local
```bash
pip install -r requirements.txt
```

## 📦 Dataset

NHATS Round 14 Clock Drawing Test
- 6,602 images labeled with CCSS scoring
- 5 domains: digit_count, worst_quadrant,
  spatial, hands_present, hands_placement
- Not included due to data use agreement

## Experiments

| Notebook | Description | AUC |
|---|---|---|
| 07j | Human label review | 0.872 |
| 07k | Undersample baseline | 0.882 |
| 07l | Hybrid VQA features | 0.887 |
| 07m | Hybrid + Hand features | 0.881 |
| 07n | Downsample class 2 | 0.880 |
| 07o | Binary label | 0.866 |
| 07p | SD Inpainting (failed) | - |
| 07q | LoRA Fine-tune (failed) | - |
| 07r | **Focal Loss (Best)** | **0.893** |
| 07s | Improved features v2 | 0.892 |

## Citation

Built for AI Builders 2026 course project.
