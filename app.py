
import streamlit as st
import torch
import torch.nn as nn
import timm
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms, models
import os
import gdown

# ── Page Config ──
st.set_page_config(
    page_title="CDT Dementia Screening",
    page_icon="🧠",
    layout="centered"
)

# ── Constants ──
CUTOFF    = 6
MODEL_URL = "https://drive.google.com/uc?id=YOUR_FILE_ID"

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

# ── Model ──
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
            backbone = models.resnet50(
                weights=None
            )
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
        with st.spinner(
            "กำลังดาวน์โหลด model (~330MB)..."
        ):
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
    """
    แปลงรูปให้ใกล้เคียง CDT dataset
    1. แปลง grayscale → RGB
    2. เพิ่ม contrast
    3. resize + normalize
    """
    # แปลงเป็น grayscale ก่อน
    # เพราะ dataset เป็นรูปขาวดำ
    gray = img.convert("L")

    # เพิ่ม contrast
    from PIL import ImageEnhance
    gray = ImageEnhance.Contrast(gray).enhance(1.5)

    # กลับเป็น RGB
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
def predict(img, model, device):
    tensor = preprocess(img).to(device)
    with torch.no_grad():
        outputs = model(tensor, mode="cls")

    scores = []
    probs  = []
    for output in outputs:
        prob  = torch.softmax(output, dim=1)[0]
        score = prob.argmax().item()
        scores.append(score)
        probs.append(prob.cpu().numpy())

    total       = sum(scores)
    is_dementia = total < CUTOFF
    return scores, probs, total, is_dementia

st.title("🧠 CDT Dementia Screening")
st.markdown(
    "ระบบวิเคราะห์ **Clock Drawing Test (CDT)** "
    "เพื่อคัดกรองความเสี่ยง Dementia"
)

st.warning(
    "⚠️ ระบบนี้เป็นเพียง **screening tool** เท่านั้น "
    "ไม่ใช่การวินิจฉัยทางการแพทย์ "
    "ควรปรึกษาแพทย์ผู้เชี่ยวชาญเสมอ"
)

# ── คำแนะนำวาดรูป ──
with st.expander("📋 วิธีวาด Clock Drawing Test"):
    st.markdown("""
    1. **วาดวงกลม** เป็นหน้าปัดนาฬิกา
    2. **ใส่ตัวเลข 1-12** ให้ครบและถูกตำแหน่ง
    3. **วาดเข็มนาฬิกา 2 เข็ม** ชี้ที่ **11:10**
       - เข็มสั้น (ชั่วโมง) → ชี้ที่เลข **11**
       - เข็มยาว (นาที) → ชี้ที่เลข **2**
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
    )
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="รูปที่อัปโหลด",
                 width=300)

else:
    try:
        from streamlit_drawable_canvas import st_canvas

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("**วาดรูปนาฬิกาในกล่องด้านล่าง**")
        with col2:
            stroke_width = st.slider(
                "ขนาดปากกา", 1, 8, 3
            )

        canvas = st_canvas(
            fill_color       = "rgba(255,255,255,1)",
            stroke_width     = stroke_width,
            stroke_color     = "#000000",
            background_color = "#FFFFFF",
            width  = 400,
            height = 400,
            drawing_mode = "freedraw",
            key = "canvas",
        )

        col1, col2 = st.columns(2)
        with col2:
            if st.button("ล้างกระดาน"):
                st.rerun()

        if (canvas.image_data is not None and
                canvas.image_data[:,:,3].sum() > 1000):
            img = Image.fromarray(
                canvas.image_data.astype("uint8"),
                "RGBA"
            ).convert("RGB")
            st.caption("พร้อมวิเคราะห์แล้ว")

    except ImportError:
        st.error(
            "ไม่พบ streamlit-drawable-canvas "
            "กรุณา upload รูปแทน"
        )

# ── Analyze Button ──
if img is not None:
    st.divider()

    if st.button(
        "🔍 วิเคราะห์", type="primary",
        use_container_width=True
    ):
        with st.spinner("กำลังวิเคราะห์รูป..."):
            model, device = load_model()
            scores, probs, total, is_dementia =                 predict(img, model, device)

        # ── แสดงรูปที่ process แล้ว ──
        with st.expander("รูปที่ใช้วิเคราะห์"):
            gray = img.convert("L")
            from PIL import ImageEnhance
            gray = ImageEnhance.Contrast(
                gray
            ).enhance(1.5)
            st.image(
                gray.convert("RGB"),
                caption="รูปหลัง preprocessing",
                width=250
            )

        st.divider()
        st.subheader("📊 ผลการวิเคราะห์")

        # ── Overall Result ──
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

        # ── Score Bar ──
        st.progress(total / 10)
        st.caption(
            f"คะแนน {total}/10 "
            f"({'ต่ำกว่าเกณฑ์' if is_dementia else 'อยู่ในเกณฑ์ปกติ'})"
        )

        st.divider()

        # ── Per-Domain ──
        st.subheader("📋 คะแนนรายด้าน")

        score_colors = {
            0: "🔴", 1: "🟡", 2: "🟢"
        }
        score_labels = {
            0: "ต้องปรับปรุง (0/2)",
            1: "พอใช้ (1/2)",
            2: "ดี (2/2)",
        }

        for i, (domain, score) in enumerate(
            zip(domain_names, scores)
        ):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"**{domain}**")
                st.caption(domain_desc[domain])
                st.progress(score / 2)
            with col2:
                st.markdown(
                    f"<div style='text-align:center;"
                    f"font-size:2em'>"
                    f"{score_colors[score]}</div>"
                    f"<div style='text-align:center'>"
                    f"{score}/2</div>",
                    unsafe_allow_html=True
                )

        st.divider()

        # ── Interpretation ──
        st.subheader("📖 การแปลผล")
        if total >= 8:
            st.success(
                "**คะแนนดีมาก** "
                "ไม่พบสัญญาณเสี่ยงใน CDT นี้"
            )
        elif total >= 6:
            st.info(
                "**คะแนนปกติ** "
                "แนะนำติดตามอาการต่อเนื่อง"
            )
        elif total >= 4:
            st.warning(
                "**คะแนนต่ำกว่าเกณฑ์** "
                "แนะนำพบแพทย์เพื่อตรวจเพิ่มเติม"
            )
        else:
            st.error(
                "**คะแนนต่ำมาก** "
                "ควรพบแพทย์ผู้เชี่ยวชาญโดยเร็ว"
            )

        st.divider()
        st.caption(
            "Model: ViT + Focal Loss | "
            "AUC = 0.893 | Sensitivity = 0.884 | "
            "Train on NHATS Round 14"
        )

# ── Sidebar ──
with st.sidebar:
    st.header("ℹ️ เกี่ยวกับระบบ")
    st.markdown("""
    **CDT Dementia Screening**

    วิเคราะห์ Clock Drawing Test
    ด้วย Deep Learning (ViT)

    ---
    **Model Performance:**
    - AUC: 0.893
    - Sensitivity: 0.884
    - Specificity: 0.701

    ---
    **CCSS Domains:**
    - A: จำนวนตัวเลข
    - B: ตำแหน่ง quadrant
    - C: การจัดวาง
    - D: เข็มนาฬิกา
    - E: ตำแหน่งเข็ม

    **Cutoff:** < 6/10 = เสี่ยง

    ---
    **Dataset:**
    NHATS Round 14
    6,602 รูป

    ---
    """)
    st.warning(
        "ใช้เพื่อ screening เท่านั้น"
    )
    st.caption(
        "Built for AI Builders 2026"
    )
