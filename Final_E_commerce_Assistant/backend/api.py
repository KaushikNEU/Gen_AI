# backend/api.py

import io
from typing import Any, Dict, List
import json

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from .core import (
    full_analysis,
    generate_reviews_for_product,
    build_report_text,
    build_report_html,
    build_report_pdf,
    generate_improved_description,
)


app = FastAPI(title="Smart E-Commerce Assistant API")

# CORS so Streamlit frontend can talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> Dict[str, str]:
    return {"message": "Smart E-Commerce Assistant backend is running."}


# -----------------------------------------------------------------------------
# /analyze – image + description → analysis JSON (ENHANCED)
# -----------------------------------------------------------------------------
@app.post("/analyze")
async def analyze_endpoint(
    file: UploadFile = File(...),
    description: str = Form(...),
    price: float = Form(...),
    category: str = Form(...),
) -> Dict[str, Any]:
    """
    ENHANCED: Now returns multiple captions and suggestions.
    """
    try:
        image_bytes = await file.read()
        analysis = full_analysis(
            image_bytes=image_bytes,
            description=description,
            price=price,
            category=category,
            generate_captions=True,  # NEW: Generate multiple caption options
        )
        return analysis
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


# -----------------------------------------------------------------------------
# NEW: /improve_description – generate improved description
# -----------------------------------------------------------------------------
@app.post("/improve_description")
async def improve_description_endpoint(
    description: str = Form(...),
    image_analysis: str = Form(...),  # JSON string
    price: float = Form(...),
    category: str = Form(...),
) -> Dict[str, Any]:
    """
    Generate an AI-improved version of the product description.
    """
    try:
        # Parse image_analysis JSON string
        image_analysis_dict = json.loads(image_analysis)
        
        improved = generate_improved_description(
            image_analysis=image_analysis_dict,
            original_description=description,
            price=price,
            category=category,
        )
        
        return {
            "original": description,
            "improved": improved,
        }
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid image_analysis JSON")
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Description improvement failed: {e}")


# -----------------------------------------------------------------------------
# /reviews – caption + description → reviews
# -----------------------------------------------------------------------------
@app.post("/reviews")
async def reviews_endpoint(
    caption: str = Form(...),
    description: str = Form(...),
    price: float = Form(...),
    category: str = Form(...),
) -> Dict[str, Any]:
    try:
        reviews = generate_reviews_for_product(
            caption=caption,
            description=description,
            price=price,
            category=category,
        )
        return {"reviews": reviews}
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Review generation failed: {e}")


# -----------------------------------------------------------------------------
# /report – image + description → text & HTML report + reviews + analysis
# -----------------------------------------------------------------------------
@app.post("/report")
async def report_endpoint(
    file: UploadFile = File(...),
    description: str = Form(...),
    price: float = Form(...),
    category: str = Form(...),
) -> Dict[str, Any]:
    try:
        image_bytes = await file.read()

        analysis = full_analysis(
            image_bytes=image_bytes,
            description=description,
            price=price,
            category=category,
            generate_captions=True,
        )
        image_analysis = analysis["image_analysis"]
        description_analysis = analysis["description_analysis"]
        comparison = analysis["comparison"]
        captions = analysis.get("captions", {})
        suggestions = analysis.get("suggestions", [])

        # Use standard caption for review generation
        standard_caption = captions.get("standard", comparison.get("img_caption", ""))
        reviews = generate_reviews_for_product(
            caption=standard_caption,
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
            captions=captions,
            suggestions=suggestions,
        )

        report_html = build_report_html(
            image_analysis=image_analysis,
            description=description,
            description_analysis=description_analysis,
            comparison=comparison,
            reviews=reviews,
            price=price,
            category=category,
            captions=captions,
            suggestions=suggestions,
        )

        return {
            "report": report_text,
            "report_html": report_html,
            "analysis": analysis,
            "reviews": reviews,
        }

    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")


# -----------------------------------------------------------------------------
# /report_pdf – image + description → PDF bytes
# -----------------------------------------------------------------------------
@app.post("/report_pdf")
async def report_pdf_endpoint(
    file: UploadFile = File(...),
    description: str = Form(...),
    price: float = Form(...),
    category: str = Form(...),
):
    try:
        image_bytes = await file.read()

        analysis = full_analysis(
            image_bytes=image_bytes,
            description=description,
            price=price,
            category=category,
            generate_captions=True,
        )
        image_analysis = analysis["image_analysis"]
        description_analysis = analysis["description_analysis"]
        comparison = analysis["comparison"]
        captions = analysis.get("captions", {})
        suggestions = analysis.get("suggestions", [])

        standard_caption = captions.get("standard", comparison.get("img_caption", ""))
        reviews = generate_reviews_for_product(
            caption=standard_caption,
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
            captions=captions,
            suggestions=suggestions,
        )

        pdf_bytes = build_report_pdf(report_text=report_text, image_bytes=image_bytes)

        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": 'attachment; filename="product_report.pdf"'},
        )

    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {e}")