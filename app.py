
import streamlit as st
import torch
import torch.nn as nn
import timm
import numpy as np
from PIL import Image
from torchvision import transforms, models
import requests
import os
import io
import gdown

# ── Page Config ──
st.set_page_config(
    page_title="CDT Dementia Screening",
    page_icon="🧠",
    layout="centered"
)

# ── Constants ──
CUTOFF     = 6
MODEL_URL  = "https://drive.google.com/uc?id=1cTh__P48xT8DE0TY6au0I_-kjwyAVyMI"
# ← จะใส่ URL หลัง upload model ขึ้น Drive

domain_names = [
    "A: Digit Count",
    "B: Worst Quadrant",
    "C: Spatial Arrangement",
    "D: Hands Present",
    "E: Hands Placement",
]
domain_desc = {
    "A: Digit Count"         : "จำนวนตัวเลขที่ถูกต้อง",
    "B: Worst Quadrant"      : "ความสมดุลของตัวเลขใน 4 ส่วน",
    "C: Spatial Arrangement" : "การจัดวางตัวเลขในวงกลม",
    "D: Hands Present"       : "การวาดเข็มนาฬิกา",
    "E: Hands Placement"     : "ตำแหน่งเข็มนาฬิกา (11:10)",
}

# ── Model Definition ──
class CDTModel(nn.Module):
    def __init__(self, backbone_name="vit",
                 num_domains=5, num_classes=3):
        super().__init__()
        if backbone_name == "vit":
            self.backbone = timm.create_model(
                "vit_base_patch16_224",
                pretrained=False, num_classes=0
            )
            self.feature_dim = 768
        else:
            backbone = models.resnet50(pretrained=False)
            self.feature_dim = 2048
            self.backbone = nn.Sequential(
                *list(backbone.children())[:-1]
            )
        self.backbone_name = backbone_name
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(), nn.Dropout(0.3)
        )
        self.heads_reg = nn.ModuleList([
            nn.Linear(512, 1)
            for _ in range(num_domains)
        ])
        self.heads_cls = nn.ModuleList([
            nn.Linear(512, num_classes)
            for _ in range(num_domains)
        ])

    def forward(self, x, mode="cls"):
        if self.backbone_name == "vit":
            feat = self.backbone(x)
            feat = feat.unsqueeze(-1).unsqueeze(-1)
        else:
            feat = self.backbone(x)
        feat = self.shared(feat)
        if mode == "reg":
            return [h(feat).squeeze(-1)
                    for h in self.heads_reg]
        return [h(feat) for h in self.heads_cls]

# ── Load Model ──
@st.cache_resource
def load_model():
    model_path = "model.pth"

    if not os.path.exists(model_path):
        with st.spinner("กำลังโหลด model..."):
            gdown.download(
                MODEL_URL, model_path, quiet=False
            )

    device = torch.device("cpu")
    model  = CDTModel(backbone_name="vit").to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    model.eval()
    return model, device

# ── Preprocess ──
def preprocess(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])
    return transform(img).unsqueeze(0)

# ── Predict ──
def predict(img, model, device):
    tensor  = preprocess(img).to(device)
    with torch.no_grad():
        outputs = model(tensor, mode="cls")

    scores = []
    probs  = []
    for output in outputs:
        prob  = torch.softmax(output, dim=1)[0]
        score = prob.argmax().item()
        scores.append(score)
        probs.append(prob.cpu().numpy())

    total        = sum(scores)
    is_dementia  = total < CUTOFF
    confidence   = total / 10.0

    return scores, probs, total, is_dementia

# ── UI ──
st.title("🧠 CDT Dementia Screening")
st.markdown("""
ระบบวิเคราะห์ Clock Drawing Test (CDT)
สำหรับคัดกรองความเสี่ยง Dementia

**วิธีใช้:** อัปโหลดหรือวาดรูปนาฬิกาด้านล่าง
""")

st.warning("""
**คำเตือน:** ระบบนี้เป็นเพียง screening tool
ไม่ใช่การวินิจฉัยทางการแพทย์
ควรปรึกษาแพทย์ผู้เชี่ยวชาญเสมอ
""")

st.divider()

# ── Input Mode ──
mode = st.radio(
    "เลือกวิธีใส่รูป",
    ["📁 Upload รูป", "✏️ วาดรูป"],
    horizontal=True
)

img = None

if mode == "📁 Upload รูป":
    uploaded = st.file_uploader(
        "อัปโหลดรูป Clock Drawing Test",
        type=["jpg", "jpeg", "png", "tif"],
        help="รองรับ JPG, PNG, TIF"
    )
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="รูปที่อัปโหลด",
                 width=300)

else:
    try:
        from streamlit_drawable_canvas import (
            st_canvas
        )

        st.markdown("**วาดรูปนาฬิกาในกล่องด้านล่าง**")
        st.caption(
            "วาดวงกลม ตัวเลข 1-12 และเข็มชี้ 11:10"
        )

        canvas = st_canvas(
            fill_color   = "rgba(255,255,255,1)",
            stroke_width = 3,
            stroke_color = "#000000",
            background_color = "#FFFFFF",
            width  = 400,
            height = 400,
            drawing_mode = "freedraw",
            key = "canvas",
        )

        if (canvas.image_data is not None and
                canvas.image_data.sum() > 0):
            img = Image.fromarray(
                canvas.image_data.astype("uint8"),
                "RGBA"
            ).convert("RGB")

    except ImportError:
        st.error(
            "ไม่พบ streamlit-drawable-canvas "
            "กรุณา upload รูปแทน"
        )

# ── Analyze ──
if img is not None:
    st.divider()

    if st.button("🔍 วิเคราะห์", type="primary",
                 use_container_width=True):
        with st.spinner("กำลังวิเคราะห์..."):
            model, device = load_model()
            scores, probs, total, is_dementia =                 predict(img, model, device)

        # ── Result ──
        st.divider()
        st.subheader("📊 ผลการวิเคราะห์")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "คะแนนรวม",
                f"{total}/10",
                help="คะแนนเต็ม 10"
            )
        with col2:
            st.metric(
                "Cutoff",
                "< 6 = เสี่ยง",
            )
        with col3:
            if is_dementia:
                st.error("🔴 เสี่ยง Dementia")
            else:
                st.success("🟢 ปกติ")

        st.divider()

        # ── Per-Domain ──
        st.subheader("📋 คะแนนรายด้าน")

        score_labels = {
            0: "❌ ต้องปรับปรุง",
            1: "⚠️ พอใช้",
            2: "✅ ดี",
        }
        score_colors = {0: "red", 1: "orange", 2: "green"}

        for i, (domain, score) in enumerate(
            zip(domain_names, scores)
        ):
            col1, col2, col3 = st.columns([3, 1, 2])
            with col1:
                st.write(f"**{domain}**")
                st.caption(domain_desc[domain])
            with col2:
                st.markdown(
                    f"<h3 style='color:"
                    f"{score_colors[score]}'>"
                    f"{score}/2</h3>",
                    unsafe_allow_html=True
                )
            with col3:
                st.write(score_labels[score])
                # Progress bar
                st.progress(score / 2)

            st.divider()

        # ── Interpretation ──
        st.subheader("การแปลผล")

        if total >= 8:
            st.success(
                "**คะแนนดีมาก** "
                "ไม่พบความเสี่ยงในการทดสอบนี้"
            )
        elif total >= 6:
            st.info(
                "**คะแนนปกติ** "
                "แต่ควรติดตามอาการอย่างต่อเนื่อง"
            )
        elif total >= 4:
            st.warning(
                "**คะแนนต่ำกว่าเกณฑ์เล็กน้อย** "
                "แนะนำให้พบแพทย์เพื่อตรวจเพิ่มเติม"
            )
        else:
            st.error(
                "**คะแนนต่ำมาก** "
                "ควรพบแพทย์ผู้เชี่ยวชาญโดยเร็ว"
            )

        st.divider()
        st.caption(
            "วิเคราะห์โดย AI Model (ViT + Focal Loss) "
            "| AUC = 0.893 | Sensitivity = 0.884"
        )

# ── Sidebar ──
with st.sidebar:
    st.header("ℹ️ เกี่ยวกับระบบ")
    st.markdown("""
    **CDT Dementia Screening**

    ระบบวิเคราะห์ Clock Drawing Test
    โดยใช้ Deep Learning

    **Model:** ViT + Focal Loss
    **AUC:** 0.893
    **Sensitivity:** 0.884

    **CCSS Scoring:**
    - Domain A: จำนวนตัวเลข
    - Domain B: ตำแหน่ง quadrant
    - Domain C: การจัดวาง
    - Domain D: เข็มนาฬิกา
    - Domain E: ตำแหน่งเข็ม

    **Cutoff:** < 6/10 = เสี่ยง Dementia
    """)

    st.divider()
    st.caption(
        "ใช้เพื่อ screening เท่านั้น "
        "ไม่ใช่การวินิจฉัย"
    )
