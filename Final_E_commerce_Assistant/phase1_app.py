import base64
import json
import os
from typing import Dict, Any, List, Tuple

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# -------------------------
# 0. Setup
# -------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set. Add it to your .env file.")

# Vision + text models (Phase 1)
vision_llm = ChatOpenAI(
    model="gpt-4o-mini",
    max_tokens=200,
    temperature=0.2,
)

text_llm = ChatOpenAI(
    model="gpt-4o-mini",
    max_tokens=300,
    temperature=0.2,
)


# -------------------------
# 1. Helper functions
# -------------------------

def image_bytes_to_data_url(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    """Convert raw image bytes to a base64 data URL for the vision model."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def analyze_image(image_bytes: bytes, mime_type: str) -> Dict[str, Any]:
    """
    Call GPT-4o-mini Vision to:
      - generate a canonical caption
      - extract features (color, material, product_type, visible_features)
    """
    data_url = image_bytes_to_data_url(image_bytes, mime_type)

    system_msg = {
        "role": "system",
        "content": (
            "You are an e-commerce product vision model. "
            "Look at the image and: "
            "1) Write ONE short canonical caption (10–18 words) "
            "   including color, material, and product type.\n"
            "2) Extract structured product features as JSON with keys:\n"
            "   - caption: string\n"
            "   - color: string or null\n"
            "   - material: string or null\n"
            "   - product_type: string or null\n"
            "   - style: string or null\n"
            "   - visible_features: list of short phrases\n"
            "Return ONLY JSON."
        ),
    }

    human_msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Analyze this product image."},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    }

    resp = vision_llm.invoke([system_msg, human_msg])
    raw_content = resp.content

    # LangChain sometimes wraps text as a list
    if isinstance(raw_content, list):
        text = "".join(str(c) for c in raw_content)
    else:
        text = str(raw_content)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: not perfect, but avoids crash
        data = {
            "caption": text.strip(),
            "color": None,
            "material": None,
            "product_type": None,
            "style": None,
            "visible_features": [],
        }

    return data


def analyze_description(description: str, price: float, category: str) -> Dict[str, Any]:
    """
    Analyze user description (no vision).
    Returns JSON with:
      - keywords
      - claims
      - implied_benefits
      - tone
      - category_guess
    """
    system_msg = {
        "role": "system",
        "content": (
            "You are an e-commerce description analyzer. "
            "Given a product description, price, and category, "
            "extract structured information and return ONLY valid JSON "
            "with keys:\n"
            "  keywords: list of important product words\n"
            "  claims: list of factual claims (e.g., waterproof, shockproof)\n"
            "  implied_benefits: list of benefits implied by the text\n"
            "  tone: one of ['minimal', 'detailed', 'exaggerated', 'vague']\n"
            "  category_guess: short string for guessed category."
        ),
    }

    human_msg = {
        "role": "user",
        "content": (
            f"Description: {description}\n"
            f"Price: {price}\n"
            f"Category: {category}"
        ),
    }

    resp = text_llm.invoke([system_msg, human_msg])
    raw_content = resp.content

    if isinstance(raw_content, list):
        text = "".join(str(c) for c in raw_content)
    else:
        text = str(raw_content)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = {
            "keywords": [],
            "claims": [],
            "implied_benefits": [],
            "tone": "unknown",
            "category_guess": "",
            "raw": text.strip(),
        }
    return data


def simple_text_embedding(text: str) -> np.ndarray:
    """
    Phase 1: SUPER simple 'embedding' using bag-of-words.
    (So you don't need extra API calls. Later we can replace this with
    OpenAI embeddings.)
    """
    # normalize
    tokens = text.lower().split()
    # count occurrences
    vocab = {}
    for t in tokens:
        vocab[t] = vocab.get(t, 0) + 1
    # sort tokens for deterministic order
    items = sorted(vocab.items())
    # convert counts to vector
    vec = np.array([count for _, count in items], dtype=float)
    if vec.size == 0:
        return np.zeros(1)
    # L2 normalize
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    # pad shorter to longer
    if a.size < b.size:
        a = np.pad(a, (0, b.size - a.size))
    elif b.size < a.size:
        b = np.pad(b, (0, a.size - b.size))

    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def compare_image_and_description(
    img_data: Dict[str, Any],
    desc_data: Dict[str, Any],
    description: str,
) -> Dict[str, Any]:
    """
    Compare image features vs description:
      - similarity score (using simple embeddings)
      - missing features (color/material/product_type not in description)
      - contradictions (e.g., color mismatch)
      - risk score (0-100)
    """
    img_caption = img_data.get("caption", "") or ""
    desc_text = description or ""

    # 1) Similarity
    img_vec = simple_text_embedding(img_caption)
    desc_vec = simple_text_embedding(desc_text)
    similarity = cosine_similarity(img_vec, desc_vec)

    # 2) Missing features
    missing_features: List[str] = []
    contradictions: List[str] = []

    desc_lower = desc_text.lower()

    color = (img_data.get("color") or "").lower()
    material = (img_data.get("material") or "").lower()
    product_type = (img_data.get("product_type") or "").lower()

    # Color check (simple heuristic)
    if color:
        if color not in desc_lower:
            missing_features.append("color")
        # naive contradiction example:
        color_words = ["black", "blue", "red", "green", "white", "yellow", "pink", "grey", "gray"]
        other_colors = [c for c in color_words if c != color]
        if any(c in desc_lower for c in other_colors):
            contradictions.append(
                f"Image looks {color}, but description mentions a different color."
            )

    if material:
        if material not in desc_lower:
            missing_features.append("material")

    if product_type:
        if product_type not in desc_lower:
            missing_features.append("product_type")

    # 3) Risk scoring (heuristic)
    risk = 100.0

    if similarity < 0.7:
        risk -= 25
    if similarity < 0.5:
        risk -= 15  # additional penalty

    if contradictions:
        risk -= 30

    risk -= 10 * len(missing_features)

    # Clamp
    risk = max(0.0, min(100.0, risk))

    return {
        "similarity": round(float(similarity), 3),
        "missing_features": missing_features,
        "contradictions": contradictions,
        "risk_score": round(float(risk), 1),
        "img_caption": img_caption,
    }


# -------------------------
# 2. Streamlit UI
# -------------------------
st.set_page_config(page_title="Smart E-Com Assistant – Phase 1", layout="wide")

st.title("🛒 Smart E-Commerce Assistant – Phase 1 (MVP)")
st.write(
    "Upload a product image and its description. "
    "The assistant will generate an AI caption from the image, "
    "analyze your description, and highlight mismatches + risk."
)

col_left, col_right = st.columns(2)

with col_left:
    uploaded_file = st.file_uploader(
        "Upload product image",
        type=["jpg", "jpeg", "png", "webp"],
    )

    description = st.text_area(
        "Product description (what the seller would write)",
        height=120,
        placeholder="e.g., 'Nice blue phone case with great protection.'",
    )

    price = st.number_input(
        "Price",
        min_value=0.0,
        step=1.0,
        value=19.99,
    )

    category = st.text_input(
        "Category",
        value="Phone Case",
        help="e.g., 'Phone Case', 'Headphones', 'Sneakers'",
    )

    run_btn = st.button("Run Phase 1 Analysis")

with col_right:
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded image", use_column_width=True)

if run_btn:
    if not uploaded_file:
        st.error("Please upload an image first.")
    elif not description.strip():
        st.error("Please enter a product description.")
    else:
        with st.spinner("Analyzing image and description..."):
            img_bytes = uploaded_file.getvalue()
            mime_type = uploaded_file.type or "image/jpeg"

            # 1) Image analysis
            img_data = analyze_image(img_bytes, mime_type)

            # 2) Description analysis
            desc_data = analyze_description(description, price, category)

            # 3) Comparison
            comparison = compare_image_and_description(img_data, desc_data, description)

        # -------------------------
        # Display results
        # -------------------------
        st.subheader("📸 AI View of the Image")
        st.markdown(f"**Canonical Caption:** {comparison['img_caption']}")
        st.json({
            "color": img_data.get("color"),
            "material": img_data.get("material"),
            "product_type": img_data.get("product_type"),
            "style": img_data.get("style"),
            "visible_features": img_data.get("visible_features"),
        })

        st.subheader("📝 Description Analysis")
        st.json(desc_data)

        st.subheader("✅ Validation & Risk")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Similarity (0–1)", comparison["similarity"])
        with col2:
            st.metric("Risk Score (0–100)", comparison["risk_score"])
        with col3:
            st.write(" ")

        st.write("**Missing features in description:**", comparison["missing_features"] or "None")
        if comparison["contradictions"]:
            st.error("Contradictions detected:")
            for c in comparison["contradictions"]:
                st.write(f"- {c}")
        else:
            st.success("No contradictions detected between image and description.")
