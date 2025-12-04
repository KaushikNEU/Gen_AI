import os
import json
import base64
import io
from typing import Dict, Any, List, Optional

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------------------------------
# Load environment variables + Initialize OpenAI
# -----------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. "
        "Create a .env file in your project root with:\n"
        "OPENAI_API_KEY=sk-..."
    )

client = OpenAI(api_key=OPENAI_API_KEY)


# -----------------------------------------------------
# Helper: Convert image bytes → Base64
# -----------------------------------------------------
def encode_image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def _strip_json_fences(text: str) -> str:
    """Remove ```json ... ``` wrappers if present."""
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.replace("```json", "").replace("```", "").strip()
    return clean


# -----------------------------------------------------
# 1. Vision Model: Analyze Product Image (caption JSON)
# -----------------------------------------------------
def analyze_image(image_bytes: bytes, mime_type: str) -> Dict[str, Any]:
    """
    Uses GPT-4o-mini Vision to extract:
      - caption
      - color
      - material
      - product_type
      - style
      - visible_features[]
    Returns a JSON-like dict.
    """
    b64 = encode_image_to_base64(image_bytes)

    system_prompt = """
You are an e-commerce vision assistant.

Analyze the product image and return ONLY valid JSON with this structure:

{
  "caption": "short sentence including color and product type",
  "color": "pink | blue | black | etc, or null",
  "material": "plastic | silicone | leather | etc, or null",
  "product_type": "phone case | t-shirt | shoes | etc, or null",
  "style": "cute | minimalist | sporty | etc, or null",
  "visible_features": ["short phrase 1", "short phrase 2"]
}
""".strip()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this product image and return JSON only."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{b64}"
                        },
                    },
                ],
            },
        ],
        max_tokens=400,
    )

    raw = response.choices[0].message.content or ""
    clean = _strip_json_fences(raw)

    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        # Fallback: keep whatever caption text we have
        data = {
            "caption": clean,
            "color": None,
            "material": None,
            "product_type": None,
            "style": None,
            "visible_features": [],
        }

    # Ensure keys exist
    data.setdefault("caption", "")
    data.setdefault("color", None)
    data.setdefault("material", None)
    data.setdefault("product_type", None)
    data.setdefault("style", None)
    data.setdefault("visible_features", [])

    return data


# -----------------------------------------------------
# 2. Description Analysis (LLM → JSON)
# -----------------------------------------------------
def analyze_description(description: str, price: float, category: str) -> Dict[str, Any]:
    """
    Extracts keywords, claims, tone, category_guess from text description.
    """

    system_prompt = """
You are an e-commerce description analyzer.

Given a product description, price, and category, extract structured info
and return ONLY valid JSON with this structure:

{
  "keywords": ["word1", "word2"],
  "claims": ["stated factual claim 1", "claim 2"],
  "implied_benefits": ["benefit 1", "benefit 2"],
  "tone": "minimal | detailed | exaggerated | vague",
  "category_guess": "short category name"
}
""".strip()

    user_content = f"""
Description: {description}
Price: {price}
Category: {category}
""".strip()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_tokens=400,
    )

    raw = response.choices[0].message.content or ""
    clean = _strip_json_fences(raw)

    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        data = {
            "keywords": [],
            "claims": [],
            "implied_benefits": [],
            "tone": "unknown",
            "category_guess": "",
            "raw": clean,
        }

    # Ensure keys exist
    data.setdefault("keywords", [])
    data.setdefault("claims", [])
    data.setdefault("implied_benefits", [])
    data.setdefault("tone", "unknown")
    data.setdefault("category_guess", "")

    return data


# -----------------------------------------------------
# 3. Embeddings + Similarity (OpenAI)
# -----------------------------------------------------
def embed_text(text: str) -> List[float]:
    """
    Generate embedding for text using OpenAI embedding model.
    Handles empty text safely.
    """
    text = text or ""
    if not text.strip():
        # Return a zero vector (size 1536 for text-embedding-3-small)
        return [0.0] * 1536

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


def cosine_similarity(v1, v2) -> float:
    v1, v2 = np.array(v1, dtype=float), np.array(v2, dtype=float)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


# -----------------------------------------------------
# 4. Compare Image vs Description + Risk Score
# -----------------------------------------------------
def compare_image_and_description(
    img_data: Dict[str, Any],
    description: str,
) -> Dict[str, Any]:
    """
    Compare the image-derived caption/features against the user description.
    Returns similarity, missing_features, contradictions, and risk_score.
    """

    img_caption = img_data.get("caption") or ""
    desc_text = description or ""

    # Similarity
    img_vec = embed_text(img_caption)
    desc_vec = embed_text(desc_text)
    similarity = cosine_similarity(img_vec, desc_vec)

    missing_features: List[str] = []
    contradictions: List[str] = []

    desc_lower = desc_text.lower()

    color = (img_data.get("color") or "").lower()
    material = (img_data.get("material") or "").lower()
    product_type = (img_data.get("product_type") or "").lower()

    known_colors = [
        "black", "blue", "red", "green", "white",
        "yellow", "pink", "grey", "gray", "purple",
    ]

    # Color mismatch
    if color:
        if color not in desc_lower:
            missing_features.append("color")
        other_colors = [c for c in known_colors if c != color]
        if any(c in desc_lower for c in other_colors):
            contradictions.append(
                f"Image looks {color}, but description mentions a different color."
            )

    # Material mismatch
    if material and material not in desc_lower:
        missing_features.append("material")

    # Product type mismatch
    if product_type and product_type not in desc_lower:
        missing_features.append("product_type")

    # -------------------------------
    # Risk: 0 (safe) → 100 (very risky)
    # -------------------------------
    risk = 0.0

    # Similarity penalties
    if similarity < 0.7:
        risk += 25
    if similarity < 0.5:
        risk += 15

    # Contradictions penalty
    if contradictions:
        risk += 30

    # Missing features penalty
    risk += 10 * len(missing_features)

    # Clamp to [0, 100]
    risk = max(0.0, min(100.0, risk))

    return {
        "img_caption": img_caption,
        "similarity": round(similarity, 3),
        "missing_features": missing_features,
        "contradictions": contradictions,
        "risk_score": round(risk, 1),
    }


# -----------------------------------------------------
# 5. Simple Review Generator (for ratings)
# -----------------------------------------------------
def generate_reviews_for_product(
    caption: str,
    description: str,
    price: float,
    category: str,
) -> List[Dict[str, Any]]:
    """
    Generate realistic reviews for a product:
    - at least one 5★, 4★, 3★, and 1★ review.
    All in text (title + body).
    """

    system_prompt = """
You are an e-commerce review generator.

Given product information, generate realistic customer reviews across ratings:
- At least one 5-star review (very positive).
- At least one 4-star review (positive, minor issues).
- At least one 3-star review (mixed).
- At least one 1-star review (clearly negative).

Return ONLY valid JSON as a list, like:

[
  {"rating": 5, "title": "string", "body": "50-80 words"},
  {"rating": 4, "title": "string", "body": "50-80 words"},
  {"rating": 3, "title": "string", "body": "50-80 words"},
  {"rating": 1, "title": "string", "body": "50-80 words"}
]

Rules:
- Each body should be 50-80 words.
- Use natural, human-like language.
- Focus on design, comfort, durability, and value for money.
""".strip()

    user_content = f"""
Product caption (from image): {caption}
Seller description: {description}
Category: {category}
Price: {price}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_tokens=700,
    )

    raw = response.choices[0].message.content or ""
    clean = _strip_json_fences(raw)

    try:
        data = json.loads(clean)
        if isinstance(data, list):
            return data
        return [data]
    except json.JSONDecodeError:
        # fallback: return a single generic 5-star review
        return [
            {
                "rating": 5,
                "title": "Great product",
                "body": clean[:300],
            }
        ]


# -----------------------------------------------------
# 6. Build a Text-Only Product Report
# -----------------------------------------------------
def build_report_text(
    image_analysis: Dict[str, Any],
    description: str,
    description_analysis: Dict[str, Any],
    comparison: Dict[str, Any],
    reviews: List[Dict[str, Any]],
    price: float,
    category: str,
) -> str:
    """
    Build a concise, plain-text report that can be:
    - shown in the UI
    - saved as .txt
    - later turned into a PDF/email

    No unnecessary info, just clear, formatted sections.
    """

    caption = image_analysis.get("caption", "")
    color = image_analysis.get("color") or "N/A"
    material = image_analysis.get("material") or "N/A"
    style = image_analysis.get("style") or "N/A"
    visible_features = image_analysis.get("visible_features") or []

    keywords = description_analysis.get("keywords") or []
    claims = description_analysis.get("claims") or []
    implied_benefits = description_analysis.get("implied_benefits") or []
    tone = description_analysis.get("tone") or "unknown"
    category_guess = description_analysis.get("category_guess") or ""

    similarity = comparison.get("similarity")
    risk_score = comparison.get("risk_score")
    missing_features = comparison.get("missing_features") or []
    contradictions = comparison.get("contradictions") or []

    lines: List[str] = []

    lines.append("=== PRODUCT REPORT ===")
    lines.append("")
    lines.append("1. Basic Product Details")
    lines.append(f"   • Category: {category}")
    lines.append(f"   • Price: ${price:.2f}")
    lines.append(f"   • AI Caption (from image): {caption}")
    lines.append("")

    lines.append("2. Image Analysis")
    lines.append(f"   • Color: {color}")
    lines.append(f"   • Material: {material}")
    lines.append(f"   • Style: {style}")
    if visible_features:
        lines.append(f"   • Visible Features: {', '.join(visible_features)}")
    lines.append("")

    lines.append("3. Seller Description")
    lines.append(f"   • Original Text: {description}")
    lines.append(f"   • Tone: {tone}")
    if keywords:
        lines.append(f"   • Keywords: {', '.join(keywords)}")
    if claims:
        lines.append(f"   • Claims: {', '.join(claims)}")
    if implied_benefits:
        lines.append(f"   • Implied Benefits: {', '.join(implied_benefits)}")
    if category_guess:
        lines.append(f"   • AI Category Guess: {category_guess}")
    lines.append("")

    lines.append("4. Consistency & Risk")
    lines.append(f"   • Similarity between image and description: {similarity}")
    lines.append(f"   • Risk Score (0 = safe, 100 = risky): {risk_score}")
    if missing_features:
        lines.append(f"   • Missing Info: {', '.join(missing_features)}")
    else:
        lines.append("   • Missing Info: None detected")
    if contradictions:
        lines.append(f"   • Contradictions: {', '.join(contradictions)}")
    else:
        lines.append("   • Contradictions: None detected")
    lines.append("")

    lines.append("5. Simulated Customer Reviews (by Rating)")
    if not reviews:
        lines.append("   No reviews generated.")
    else:
        # Sort by rating descending
        sorted_reviews = sorted(reviews, key=lambda r: r.get("rating", 0), reverse=True)
        for r in sorted_reviews:
            rating = r.get("rating", "N/A")
            title = r.get("title", "").strip()
            body = r.get("body", "").strip()
            lines.append(f"   --- {rating}-Star Review ---")
            if title:
                lines.append(f"   Title: {title}")
            lines.append(f"   Review: {body}")
            lines.append("")

    report_text = "\n".join(lines).strip()
    return report_text


# -----------------------------------------------------
# 7. Build PDF from report text + image (word wrapping + multipage)
# -----------------------------------------------------
def build_report_pdf(report_text: str, image_bytes: Optional[bytes] = None) -> bytes:
    """
    Build a PDF containing:
    - the product image at the top (if provided)
    - the plain-text report below (word-wrapped, multi-page if needed)

    Returns PDF bytes, ready to send as a file response.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader
    except ImportError:
        raise RuntimeError(
            "reportlab is not installed. Install it with: pip install reportlab"
        )

    import textwrap

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    left_margin = 50
    top_margin = height - 50
    bottom_margin = 50

    y = top_margin

    # 1) Draw image (optional)
    if image_bytes:
        try:
            img_reader = ImageReader(io.BytesIO(image_bytes))
            img_width, img_height = img_reader.getSize()

            max_width = width - 100  # margins
            max_height = height / 3  # top third of page

            scale = min(max_width / img_width, max_height / img_height, 1.0)
            draw_w = img_width * scale
            draw_h = img_height * scale

            x = (width - draw_w) / 2
            c.drawImage(img_reader, x, y - draw_h, width=draw_w, height=draw_h)

            y = y - draw_h - 30  # move cursor below image
        except Exception:
            # If image fails, just ignore and continue with text
            y = top_margin

    # 2) Draw text (wrap by words, multi-page)
    def new_text_obj(start_y: float):
        t = c.beginText(left_margin, start_y)
        t.setFont("Helvetica", 11)
        return t

    text_obj = new_text_obj(y)
    max_chars = 90  # approximate device-independent width

    for line in report_text.splitlines():
        if not line.strip():
            # blank line
            text_obj.textLine("")
            # check page break
            if text_obj.getY() <= bottom_margin:
                c.drawText(text_obj)
                c.showPage()
                text_obj = new_text_obj(top_margin)
            continue

        wrapped_lines = textwrap.wrap(line, width=max_chars) or [line]
        for seg in wrapped_lines:
            text_obj.textLine(seg)
            if text_obj.getY() <= bottom_margin:
                # finish this page and start a new one
                c.drawText(text_obj)
                c.showPage()
                text_obj = new_text_obj(top_margin)

    # Draw remaining text
    c.drawText(text_obj)
    c.showPage()
    c.save()

    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


# -----------------------------------------------------
# 8. Main function used by /analyze
# -----------------------------------------------------
def full_analysis(
    image_bytes: bytes,
    mime_type: str,
    description: str,
    price: float,
    category: str,
) -> Dict[str, Any]:
    """
    Main function the /analyze endpoint calls:
    - image_bytes + mime_type → image_analysis (JSON)
    - description + price + category → description_analysis (JSON)
    - both → comparison (similarity, risk, etc.)
    """

    image_analysis = analyze_image(image_bytes, mime_type)
    description_analysis = analyze_description(description, price, category)
    comparison = compare_image_and_description(image_analysis, description)

    return {
        "image_analysis": image_analysis,
        "description_analysis": description_analysis,
        "comparison": comparison,
    }
