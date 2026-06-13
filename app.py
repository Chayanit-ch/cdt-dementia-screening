
import streamlit as st
import torch
import torch.nn as nn
import timm
import numpy as np
from PIL import Image, ImageEnhance
from torchvision import transforms, models
import os
import gdown
import json

st.set_page_config(
    page_title="CDT Dementia Screening",
    page_icon="🧠",
    layout="centered"
)

CUTOFF = 6

# FILE IDs
MODEL_07L_URL = "https://drive.google.com/uc?id=17hHmRT0XT7RAK7yWNbEYiWTp4sDc_MbZ"
MODEL_07R_URL = "https://drive.google.com/uc?id=1cTh__P48xT8DE0TY6au0I_-kjwyAVyMI"

# Domain selection (from val set)
# A=07l, B=07l, C=07r, D=07r, E=07r
USE_07R = [False, False, True, True, True]

domain_names = [
    "A: Digit Count",
    "B: Worst Quadrant",
    "C: Spatial Arrangement",
    "D: Hands Present",
    "E: Hands Placement",
]
domain_desc = {
    "A: Digit Count"         : "ตัวเลข 1-12 ครบถ้วนถูกต้อง",
    "B: Worst Quadrant"      : "ตัวเลขกระจายสม่ำเสมอใน 4 ส่วน",
    "C: Spatial Arrangement" : "การจัดวางตัวเลขในวงกลม",
    "D: Hands Present"       : "มีเข็มนาฬิกาครบ 2 เข็ม",
    "E: Hands Placement"     : "เข็มชี้ถูกตำแหน่ง 11:10",
}

# ── Model Definitions ──
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
            backbone = models.resnet50(weights=None)
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

class CDTHybridModel(nn.Module):
    def __init__(self, backbone_name="vit",
                 vqa_dim=12, num_domains=5):
        super().__init__()
        if backbone_name == "vit":
            self.backbone = timm.create_model(
                "vit_base_patch16_224",
                pretrained=False, num_classes=0
            )
            vis_dim = 768
        else:
            backbone = models.resnet50(weights=None)
            self.backbone = nn.Sequential(
                *list(backbone.children())[:-1]
            )
            vis_dim = 2048
        self.backbone_name = backbone_name
        self.vis_shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(vis_dim, 512),
            nn.ReLU(), nn.Dropout(0.3)
        )
        self.vqa_encoder = nn.Sequential(
            nn.Linear(vqa_dim, 64),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(512+32, 256),
            nn.ReLU(), nn.Dropout(0.3)
        )
        self.heads_reg = nn.ModuleList([
            nn.Linear(256, 1)
            for _ in range(num_domains)
        ])
        self.heads_cls = nn.ModuleList([
            nn.Linear(256, 3)
            for _ in range(num_domains)
        ])

    def forward(self, img, vqa_feat=None,
                mode="reg"):
        if self.backbone_name == "vit":
            vis = self.backbone(img)
            vis = vis.unsqueeze(-1).unsqueeze(-1)
        else:
            vis = self.backbone(img)
        vis = self.vis_shared(vis)
        if vqa_feat is not None:
            vqa   = self.vqa_encoder(vqa_feat)
            fused = self.fusion(
                torch.cat([vis, vqa], dim=1)
            )
        else:
            fused = vis[:, :256]
        if mode == "reg":
            return [h(fused).squeeze(-1)
                    for h in self.heads_reg]
        return [h(fused) for h in self.heads_cls]

# ── Load Models ──
@st.cache_resource
def load_models():
    device = torch.device("cpu")

    # โหลด 07l
    if not os.path.exists("model_07l.pth"):
        with st.spinner(
            "กำลังดาวน์โหลด Model 1/2..."
        ):
            gdown.download(
                MODEL_07L_URL,
                "model_07l.pth", quiet=False
            )

    # โหลด 07r
    if not os.path.exists("model_07r.pth"):
        with st.spinner(
            "กำลังดาวน์โหลด Model 2/2..."
        ):
            gdown.download(
                MODEL_07R_URL,
                "model_07r.pth", quiet=False
            )

    model_07l = CDTHybridModel(
        backbone_name="vit", vqa_dim=12
    ).to(device)
    model_07l.load_state_dict(
        torch.load("model_07l.pth",
                   map_location=device)
    )
    model_07l.eval()

    model_07r = CDTModel(
        backbone_name="vit"
    ).to(device)
    model_07r.load_state_dict(
        torch.load("model_07r.pth",
                   map_location=device)
    )
    model_07r.eval()

    return model_07l, model_07r, device

# ── Preprocess ──
def preprocess(img):
    gray = img.convert("L")
    from PIL import ImageEnhance
    gray     = ImageEnhance.Contrast(
        gray
    ).enhance(1.5)
    img_proc = gray.convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])
    return transform(img_proc).unsqueeze(0)

# ── Predict ──
def predict(img, model_07l, model_07r, device):
    tensor = preprocess(img).to(device)

    scores = []
    with torch.no_grad():
        # 07l predict A, B (index 0,1)
        out_07l = model_07l(tensor, mode="reg")
        # 07r predict C, D, E (index 2,3,4)
        out_07r = model_07r(tensor, mode="cls")

    for i in range(5):
        if USE_07R[i]:
            # จาก 07r
            prob  = torch.softmax(
                out_07r[i], dim=1
            )[0]
            score = prob.argmax().item()
        else:
            # จาก 07l
            score = int(
                out_07l[i].round().clamp(0,2).item()
            )
        scores.append(score)

    total       = sum(scores)
    is_dementia = total < CUTOFF
    return scores, total, is_dementia

st.title("🧠 CDT Dementia Screening")
st.markdown(
    "ระบบวิเคราะห์ **Clock Drawing Test (CDT)** "
    "เพื่อคัดกรองความเสี่ยง Dementia"
)

st.warning(
    "ระบบนี้เป็นเพียง **screening tool** "
    "ไม่ใช่การวินิจฉัยทางการแพทย์"
)

with st.expander("📋 วิธีวาด Clock Drawing Test"):
    st.markdown("""
    1. **วาดวงกลม** เป็นหน้าปัดนาฬิกา
    2. **ใส่ตัวเลข 1-12** ให้ครบและถูกตำแหน่ง
    3. **วาดเข็ม 2 เข็ม** ชี้ที่ **11:10**
       - เข็มสั้น (ชั่วโมง) → ชี้เลข **11**
       - เข็มยาว (นาที) → ชี้เลข **2**
    """)

st.divider()

mode = st.radio(
    "เลือกวิธีใส่รูป",
    ["📁 Upload รูป", "✏️ วาดรูป"],
    horizontal=True
)

img = None

if mode == "📁 Upload รูป":
    uploaded = st.file_uploader(
        "อัปโหลดรูป CDT",
        type=["jpg","jpeg","png","tif"],
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
        col1, col2 = st.columns([3,1])
        with col1:
            st.markdown("**วาดรูปนาฬิกาด้านล่าง**")
        with col2:
            stroke = st.slider("ขนาดปากกา",1,8,3)

        canvas = st_canvas(
            fill_color       = "rgba(255,255,255,1)",
            stroke_width     = stroke,
            stroke_color     = "#000000",
            background_color = "#FFFFFF",
            width=400, height=400,
            drawing_mode="freedraw",
            key="canvas",
        )
        if (canvas.image_data is not None and
                canvas.image_data[:,:,3].sum()>1000):
            img = Image.fromarray(
                canvas.image_data.astype("uint8"),
                "RGBA"
            ).convert("RGB")
            st.caption("พร้อมวิเคราะห์แล้ว")
    except ImportError:
        st.error("กรุณา upload รูปแทน")

if img is not None:
    st.divider()
    if st.button(
        "🔍 วิเคราะห์", type="primary",
        use_container_width=True
    ):
        with st.spinner("กำลังวิเคราะห์..."):
            model_07l, model_07r, device =                 load_models()
            scores, total, is_dementia = predict(
                img, model_07l, model_07r, device
            )

        with st.expander("🖼️ รูปที่ใช้วิเคราะห์"):
            gray = ImageEnhance.Contrast(
                img.convert("L")
            ).enhance(1.5)
            st.image(
                gray.convert("RGB"),
                caption="หลัง preprocessing",
                width=250
            )

        st.divider()
        st.subheader("📊 ผลการวิเคราะห์")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("คะแนนรวม", f"{total}/10")
        with col2:
            st.metric("เกณฑ์", "< 6 = เสี่ยง")
        with col3:
            if is_dementia:
                st.error("🔴 เสี่ยง Dementia")
            else:
                st.success("🟢 ปกติ")

        st.progress(total / 10)
        st.divider()

        st.subheader("📋 คะแนนรายด้าน")
        colors = {0:"🔴", 1:"🟡", 2:"🟢"}
        models_used = {
            0:"07l", 1:"07l",
            2:"07r", 3:"07r", 4:"07r"
        }

        for i, (domain, score) in enumerate(
            zip(domain_names, scores)
        ):
            col1, col2 = st.columns([4,1])
            with col1:
                st.write(f"**{domain}**")
                st.caption(domain_desc[domain])
                st.progress(score / 2)
            with col2:
                st.markdown(
                    f"<div style='text-align:center;"
                    f"font-size:2em'>"
                    f"{colors[score]}</div>"
                    f"<div style='text-align:center'>"
                    f"{score}/2</div>",
                    unsafe_allow_html=True
                )

        st.divider()

        if total >= 8:
            st.success("**คะแนนดีมาก** ไม่พบสัญญาณเสี่ยง")
        elif total >= 6:
            st.info("**คะแนนปกติ** แนะนำติดตามต่อเนื่อง")
        elif total >= 4:
            st.warning("**ต่ำกว่าเกณฑ์** ควรพบแพทย์")
        else:
            st.error("**คะแนนต่ำมาก** ควรพบแพทย์โดยด่วน")

        st.caption(
            "Domain-wise Ensemble "
            "(07l + 07r) | AUC=0.901"
        )

with st.sidebar:
    st.header("ℹ️ เกี่ยวกับระบบ")
    st.markdown("""
    **CDT Dementia Screening**

    วิเคราะห์ Clock Drawing Test
    ด้วย Domain-wise Ensemble

    ---
    **Model Performance:**
    - AUC: 0.901
    - Sensitivity: 0.871
    - Specificity: 0.754

    ---
    **Domain Selection:**
    - A, B → Hybrid VQA (07l)
    - C, D, E → Focal Loss (07r)

    ---
    **CCSS Domains:**
    - A: จำนวนตัวเลข
    - B: ตำแหน่ง quadrant
    - C: การจัดวาง
    - D: เข็มนาฬิกา
    - E: ตำแหน่งเข็ม

    **Cutoff:** < 6/10 = เสี่ยง
    """)
    st.warning("ใช้เพื่อ screening เท่านั้น")
    st.caption("Built for AI Builders 2026")
