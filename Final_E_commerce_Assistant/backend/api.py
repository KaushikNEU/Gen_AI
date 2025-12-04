from typing import Any, Dict

import io
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .core import (
    full_analysis,
    generate_reviews_for_product,
    build_report_text,
    build_report_pdf,
)

app = FastAPI(title="Smart E-Commerce Assistant – Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "status": "ok",
        "message": "Smart E-Commerce Assistant backend is running.",
        "endpoints": ["/analyze", "/reviews", "/report", "/report_pdf", "/docs"],
    }


# 1) Image + Description → JSON Analysis
@app.post("/analyze")
async def analyze_endpoint(
    file: UploadFile = File(...),
    description: str = Form(...),
    price: float = Form(...),
    category: str = Form(...),
) -> Dict[str, Any]:
    """
    Core endpoint:
    - receives image + description + price + category
    - returns:
        - image_analysis (JSON caption + features)
        - description_analysis
        - comparison (similarity, missing, contradictions, risk_score)
    """
    img_bytes = await file.read()
    mime_type = file.content_type or "image/jpeg"

    result = full_analysis(
        image_bytes=img_bytes,
        mime_type=mime_type,
        description=description,
        price=price,
        category=category,
    )
    return result


# 2) Reviews endpoint (no image, uses caption + text info)
@app.post("/reviews")
async def reviews_endpoint(
    caption: str = Form(...),
    description: str = Form(...),
    price: float = Form(...),
    category: str = Form(...),
) -> Dict[str, Any]:
    """
    Reviews endpoint:
    - receives caption (from /analyze), description, price, category
    - returns a list of simulated reviews for 5★, 4★, 3★, and 1★.
    All in text form (title + body).
    """

    reviews = generate_reviews_for_product(
        caption=caption,
        description=description,
        price=price,
        category=category,
    )

    return {
        "caption": caption,
        "description": description,
        "price": price,
        "category": category,
        "reviews": reviews,
    }


# 3) Full text report (image + description → report text)
@app.post("/report")
async def report_endpoint(
    file: UploadFile = File(...),
    description: str = Form(...),
    price: float = Form(...),
    category: str = Form(...),
) -> Dict[str, Any]:
    """
    Report endpoint:
    - receives image + description + price + category
    - internally:
        - runs full_analysis()
        - generates reviews
        - builds a concise text-only report
    - returns:
        - 'report' (plain text)
        - 'analysis' (JSON like /analyze)
        - 'reviews' (same structure as /reviews)
    """

    img_bytes = await file.read()
    mime_type = file.content_type or "image/jpeg"

    analysis = full_analysis(
        image_bytes=img_bytes,
        mime_type=mime_type,
        description=description,
        price=price,
        category=category,
    )

    image_analysis = analysis["image_analysis"]
    description_analysis = analysis["description_analysis"]
    comparison = analysis["comparison"]

    # Use caption from image analysis as the main product hook
    caption = image_analysis.get("caption", "")

    reviews = generate_reviews_for_product(
        caption=caption,
        description=description,
        price=price,
        category=category,
    )

    report_text = build_report_text(
        image_analysis=image_analysis,
        description=description,
        description_analysis=description_analysis,
        comparison=comparison,
        reviews=reviews,
        price=price,
        category=category,
    )

    return {
        "report": report_text,
        "analysis": analysis,
        "reviews": reviews,
    }


# 4) PDF report endpoint (image + description → downloadable PDF)
@app.post("/report_pdf")
async def report_pdf_endpoint(
    file: UploadFile = File(...),
    description: str = Form(...),
    price: float = Form(...),
    category: str = Form(...),
):
    """
    PDF report endpoint:
    - receives image + description + price + category
    - internally:
        - runs full_analysis()
        - generates reviews
        - builds a text report
        - converts it into a PDF (with image at top)
    - returns the PDF as a downloadable file.
    """

    img_bytes = await file.read()
    mime_type = file.content_type or "image/jpeg"

    analysis = full_analysis(
        image_bytes=img_bytes,
        mime_type=mime_type,
        description=description,
        price=price,
        category=category,
    )

    image_analysis = analysis["image_analysis"]
    description_analysis = analysis["description_analysis"]
    comparison = analysis["comparison"]

    caption = image_analysis.get("caption", "")

    reviews = generate_reviews_for_product(
        caption=caption,
        description=description,
        price=price,
        category=category,
    )

    report_text = build_report_text(
        image_analysis=image_analysis,
        description=description,
        description_analysis=description_analysis,
        comparison=comparison,
        reviews=reviews,
        price=price,
        category=category,
    )

    pdf_bytes = build_report_pdf(report_text=report_text, image_bytes=img_bytes)

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="product_report.pdf"'},
    )
