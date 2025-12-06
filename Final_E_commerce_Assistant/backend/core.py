import io
import os
import json
import math
import base64
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from html import escape as esc


# ----------------------------------------------------------------------
# Load environment (.env)
# ----------------------------------------------------------------------
load_dotenv()


# ----------------------------------------------------------------------
# Helper: Return OpenAI client using API key
# ----------------------------------------------------------------------
def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please export it or add it to your `.env` file."
        )
    return OpenAI(api_key=api_key)


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
VISION_MODEL = "gpt-4o-mini"
TEXT_MODEL = "gpt-4o-mini"


# ----------------------------------------------------------------------
# 1. IMAGE CAPTIONER (robust JSON parsing)
# ----------------------------------------------------------------------
def analyze_image(image_bytes: bytes) -> Dict[str, Any]:
    """
    Given raw image bytes, use GPT-4o Vision to extract:
    - caption
    - color
    - material
    - product-type guess
    - visible features

    We try to be robust to the model accidentally wrapping JSON in ```json``` fences
    or adding a short explanation.
    """
    client = get_openai_client()

    b64_img = base64.b64encode(image_bytes).decode("utf-8")

    system_prompt = """
You are an AI assistant analyzing an e-commerce product image.

Return STRICT JSON ONLY, no markdown, no code fences, no explanation.

Format EXACTLY:

{
  "caption": "...",
  "color": "... or null",
  "material": "... or null",
  "product_type": "... or null",
  "style": "... or null",
  "visible_features": []
}
"""

    resp = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                    }
                ],
            },
        ],
        temperature=0.2,
    )

    raw = resp.choices[0].message.content.strip()

    # 1) First attempt: direct JSON
    try:
        return json.loads(raw)
    except Exception:
        pass

    # 2) Second attempt: strip ```json ... ``` fences if present
    if raw.startswith("```"):
        # remove leading/trailing backticks
        raw = raw.strip("`")
        # sometimes starts with 'json\n'
        if raw.lower().startswith("json"):
            raw = raw[4:].lstrip()

    # 3) Third attempt: extract the first {...} block
    if "{" in raw and "}" in raw:
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            candidate = raw[start:end]
            return json.loads(candidate)
        except Exception:
            pass

    # 4) Final fallback
    return {"caption": "Unrecognized image."}


# ----------------------------------------------------------------------
# 2. SELLER DESCRIPTION ANALYZER
# ----------------------------------------------------------------------
def analyze_description(description: str, category: str) -> Dict[str, Any]:
    """
    Extract keywords, claims, tone, category guess, etc.
    """
    client = get_openai_client()

    system_prompt = """
Extract structured information from the seller's description.
Respond with JSON ONLY:
{
  "keywords": [],
  "claims": [],
  "implied_benefits": [],
  "tone": "",
  "category_guess": ""
}
    """

    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": description},
        ],
        temperature=0.2,
    )

    try:
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {
            "keywords": [],
            "claims": [],
            "implied_benefits": [],
            "tone": "unknown",
            "category_guess": category,
        }


# ----------------------------------------------------------------------
# 3. SIMILARITY CHECK + RISK SCORE
# ----------------------------------------------------------------------
def semantic_similarity(text1: str, text2: str) -> float:
    """
    Convert both texts to embeddings and compute cosine similarity.
    """
    client = get_openai_client()

    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text1, text2],
    )

    a = emb.data[0].embedding
    b = emb.data[1].embedding

    # cosine similarity
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b + 1e-8)


def compute_risk_score(
    similarity: float, missing_feats: List[str], contradictions: List[str]
) -> int:
    """
    Weighted scoring:
    - low similarity → more risk
    - missing features → moderate risk
    - contradictions → high risk
    """
    score = 0

    # Similarity impact
    if similarity < 0.4:
        score += 30
    elif similarity < 0.6:
        score += 10

    # Missing features
    score += len(missing_feats) * 5

    # Contradictions
    score += len(contradictions) * 25

    return min(score, 100)


# ----------------------------------------------------------------------
# 4. FULL ANALYSIS (Pipeline)
# ----------------------------------------------------------------------
def full_analysis(
    image_bytes: bytes, description: str, price: float, category: str
) -> Dict[str, Any]:
    image_analysis = analyze_image(image_bytes)
    description_analysis = analyze_description(description, category)

    # Compute similarity between AI caption and description
    caption = image_analysis.get("caption", "")
    similarity = semantic_similarity(caption, description)

    # Simple heuristics
    missing = []
    if not image_analysis.get("color"):
        missing.append("color")
    if not image_analysis.get("product_type"):
        missing.append("product_type")

    contradictions: List[str] = []

    risk = compute_risk_score(similarity, missing, contradictions)

    comparison = {
        "img_caption": caption,
        "similarity": similarity,
        "missing_features": missing,
        "contradictions": contradictions,
        "risk_score": risk,
    }

    return {
        "image_analysis": image_analysis,
        "description_analysis": description_analysis,
        "comparison": comparison,
    }


# ----------------------------------------------------------------------
# 5. REVIEW GENERATION
# ----------------------------------------------------------------------
def generate_reviews_for_product(
    caption: str, description: str, price: float, category: str
) -> List[Dict[str, Any]]:
    client = get_openai_client()

    prompt = f"""
Generate 4 realistic customer reviews for the product described below.
Return STRICT JSON ONLY as a list of review objects:
[
  {{"rating": 5, "title": "...", "body": "..."}},
  {{"rating": 4, "title": "...", "body": "..."}},
  {{"rating": 3, "title": "...", "body": "..."}},
  {{"rating": 1, "title": "...", "body": "..."}}
]

Product Caption: {caption}
Seller Description: {description}
Category: {category}
Price: {price}
"""

    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[{"role": "system", "content": prompt}],
        temperature=0.4,
    )

    try:
        return json.loads(response.choices[0].message.content)
    except Exception:
        return []


# ----------------------------------------------------------------------
# 6. TEXT REPORT
# ----------------------------------------------------------------------
def build_report_text(
    image_analysis: Dict[str, Any],
    description: str,
    description_analysis: Dict[str, Any],
    comparison: Dict[str, Any],
    reviews: List[Dict[str, Any]],
    price: float,
    category: str,
) -> str:

    caption = image_analysis.get("caption", "")
    color = image_analysis.get("color") or "N/A"
    material = image_analysis.get("material") or "N/A"
    style = image_analysis.get("style") or "N/A"
    visible = ", ".join(image_analysis.get("visible_features") or [])

    keywords = ", ".join(description_analysis.get("keywords") or [])
    claims = ", ".join(description_analysis.get("claims") or [])
    benefits = ", ".join(description_analysis.get("implied_benefits") or [])
    tone = description_analysis.get("tone", "unknown")
    cat_guess = description_analysis.get("category_guess", "")

    similarity = comparison.get("similarity", 0.0)
    risk = comparison.get("risk_score", 0)
    missing = ", ".join(comparison.get("missing_features") or [])
    contradictions = ", ".join(comparison.get("contradictions") or [])

    # Build text report
    text: List[str] = []
    text.append("=== PRODUCT REPORT ===\n")

    text.append("1. Basic Product Info")
    text.append(f"- Category: {category}")
    text.append(f"- Price: ${price:.2f}")
    text.append(f"- AI Caption: {caption}\n")

    text.append("2. Image Analysis")
    text.append(f"- Color: {color}")
    text.append(f"- Material: {material}")
    text.append(f"- Style: {style}")
    text.append(f"- Visible Features: {visible}\n")

    text.append("3. Seller Description")
    text.append(f"- Description: {description}")
    text.append(f"- Tone: {tone}")
    text.append(f"- Keywords: {keywords}")
    text.append(f"- Claims: {claims}")
    text.append(f"- Implied Benefits: {benefits}")
    text.append(f"- Category Guess: {cat_guess}\n")

    text.append("4. Consistency Check")
    text.append(f"- Similarity: {similarity:.3f}")
    text.append(f"- Risk Score: {risk}")
    text.append(f"- Missing Features: {missing or 'None'}")
    text.append(f"- Contradictions: {contradictions or 'None'}\n")

    text.append("5. Customer Reviews")
    if not reviews:
        text.append("No reviews generated.\n")
    else:
        for r in sorted(reviews, key=lambda r: r.get("rating", 0), reverse=True):
            text.append(f"--- {r['rating']}-Star Review ---")
            text.append(f"Title: {r['title']}")
            text.append(f"{r['body']}\n")

    return "\n".join(text)


# ----------------------------------------------------------------------
# 7. HTML REPORT BUILDER
# ----------------------------------------------------------------------
def build_report_html(
    image_analysis: Dict[str, Any],
    description: str,
    description_analysis: Dict[str, Any],
    comparison: Dict[str, Any],
    reviews: List[Dict[str, Any]],
    price: float,
    category: str,
) -> str:
    """
    Beautiful HTML version of the report.
    """

    caption = image_analysis.get("caption", "")
    color = image_analysis.get("color") or "N/A"
    material = image_analysis.get("material") or "N/A"
    style = image_analysis.get("style") or "N/A"
    visible = image_analysis.get("visible_features") or []

    keywords = description_analysis.get("keywords") or []
    claims = description_analysis.get("claims") or []
    benefits = description_analysis.get("implied_benefits") or []
    tone = description_analysis.get("tone") or "unknown"
    cat_guess = description_analysis.get("category_guess") or ""

    similarity = comparison.get("similarity", 0.0)
    risk = comparison.get("risk_score", 0)
    missing = comparison.get("missing_features") or []
    contradictions = comparison.get("contradictions") or []

    # Risk color
    if risk < 40:
        risk_class = "risk-low"
    elif risk < 70:
        risk_class = "risk-medium"
    else:
        risk_class = "risk-high"

    styles = """
    <style>
      body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: #f8fafc;
        padding: 20px;
      }
      .container {
        background: #fff;
        padding: 24px;
        border-radius: 10px;
        max-width: 900px;
        margin: auto;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
      }
      h1 { margin-top: 0; }
      h2 { margin-top: 24px; }
      .pill {
        background: #e2e8f0;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 12px;
        margin-right: 6px;
        display: inline-block;
        margin-bottom: 4px;
      }
      .review {
        border: 1px solid #e2e8f0;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 10px;
        background: #f8fafc;
      }
      .review-rating {
        font-weight: bold;
        color: #eab308;
        margin-bottom: 4px;
      }
      .risk-low   { color: #16a34a; }
      .risk-medium{ color: #ca8a04; }
      .risk-high  { color: #dc2626; }
    </style>
    """

    html_parts: List[str] = []
    html_parts.append("<html><head>")
    html_parts.append(styles)
    html_parts.append("</head><body>")
    html_parts.append('<div class="container">')

    html_parts.append("<h1>Product Report</h1>")

    html_parts.append("<h2>1. Basic Product Info</h2>")
    html_parts.append(f"<p><b>Category:</b> {esc(category)}</p>")
    html_parts.append(f"<p><b>Price:</b> ${price:.2f}</p>")
    html_parts.append(f"<p><b>AI Caption:</b> {esc(caption)}</p>")

    html_parts.append("<h2>2. Image Analysis</h2>")
    html_parts.append(f"<p><b>Color:</b> {esc(color)}</p>")
    html_parts.append(f"<p><b>Material:</b> {esc(material)}</p>")
    html_parts.append(f"<p><b>Style:</b> {esc(style)}</p>")

    if visible:
        html_parts.append("<p><b>Visible Features:</b></p>")
        html_parts.append("<div>")
        for feat in visible:
            html_parts.append(f'<span class="pill">{esc(str(feat))}</span>')
        html_parts.append("</div>")

    html_parts.append("<h2>3. Seller Description</h2>")
    html_parts.append(f"<p>{esc(description)}</p>")
    html_parts.append(f"<p><b>Tone:</b> {esc(tone)}</p>")
    html_parts.append(f"<p><b>Category Guess:</b> {esc(cat_guess)}</p>")

    if keywords:
        html_parts.append("<p><b>Keywords:</b></p>")
        html_parts.append("<div>")
        for k in keywords:
            html_parts.append(f'<span class="pill">{esc(str(k))}</span>')
        html_parts.append("</div>")

    if claims:
        html_parts.append("<p><b>Claims:</b></p><ul>")
        for c in claims:
            html_parts.append(f"<li>{esc(str(c))}</li>")
        html_parts.append("</ul>")

    if benefits:
        html_parts.append("<p><b>Implied Benefits:</b></p><ul>")
        for b in benefits:
            html_parts.append(f"<li>{esc(str(b))}</li>")
        html_parts.append("</ul>")

    html_parts.append("<h2>4. Consistency Check</h2>")
    html_parts.append(f"<p><b>Similarity:</b> {similarity:.3f}</p>")
    html_parts.append(
        f'<p><b>Risk Score:</b> <span class="{risk_class}">{risk}/100</span></p>'
    )

    if missing:
        html_parts.append("<p><b>Missing Features:</b></p><ul>")
        for m in missing:
            html_parts.append(f"<li>{esc(str(m))}</li>")
        html_parts.append("</ul>")

    if contradictions:
        html_parts.append("<p><b>Contradictions:</b></p><ul>")
        for c in contradictions:
            html_parts.append(f"<li>{esc(str(c))}</li>")
        html_parts.append("</ul>")

    html_parts.append("<h2>5. Customer Reviews</h2>")
    if not reviews:
        html_parts.append("<p>No reviews generated.</p>")
    else:
        for r in sorted(reviews, key=lambda x: x.get("rating", 0), reverse=True):
            html_parts.append('<div class="review">')
            html_parts.append(
                f"<div class='review-rating'>{'★' * r['rating']} "
                f"({r['rating']}-Star Review)</div>"
            )
            html_parts.append(f"<b>{esc(r['title'])}</b><br>")
            html_parts.append(f"<p>{esc(r['body'])}</p>")
            html_parts.append("</div>")

    html_parts.append("</div></body></html>")

    return "".join(html_parts)


# ----------------------------------------------------------------------
# 8. PDF REPORT BUILDER
# ----------------------------------------------------------------------
def build_report_pdf(report_text: str, image_bytes: Optional[bytes] = None) -> bytes:
    """
    Takes the text report + image, generates a PDF.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)

    width, height = A4
    y = height - 40

    # Draw image if available
    if image_bytes:
        try:
            img = ImageReader(io.BytesIO(image_bytes))
            c.drawImage(
                img,
                40,
                y - 260,
                width=200,
                height=200,
                preserveAspectRatio=True,
            )
            y -= 280
        except Exception:
            # If image fails, ignore and just keep writing text
            pass

    # Write text
    lines = report_text.split("\n")
    for line in lines:
        if y < 50:
            c.showPage()
            y = height - 40
        c.drawString(40, y, line[:120])
        y -= 16

    c.save()
    return buffer.getvalue()
