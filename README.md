# CDT Dementia Screening

AI-powered Clock Drawing Test (CDT) analysis
for dementia screening using CCSS scoring standard.

## 🌐 Demo

[![Streamlit App]](https://cdt-dementia-screening-as7vhbgavmfsmc9np9o5r7.streamlit.app/)

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

├── app.py                  ← Streamlit app

├── requirements.txt

├── src/

│   ├── models.py           ← CDTModel, CDTHybridModel

│   └── features.py         ← VQA feature functions

└── notebooks/

├── 07j_human_labels.ipynb

├── 07k_undersample.ipynb

├── 07l_vqa_features.ipynb

├── 07m_hand_features.ipynb

├── 07n_downsample_train.ipynb

├── 07o_binary_label.ipynb

├── 07p_generate_images.ipynb

├── 07q_finetune_sd.ipynb

├── 07r_focal_loss.ipynb

├── 07s_improved_features.ipynb

└── 08_evaluation.ipynb

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
