import io
from typing import Any, Dict, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from .core import (
    full_analysis,
    generate_reviews_for_product,
    build_report_text,
    build_report_html,
    build_report_pdf,
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
# /analyze – image + description → analysis JSON
# -----------------------------------------------------------------------------
@app.post("/analyze")
async def analyze_endpoint(
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
        )
        return analysis
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


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
        )
        image_analysis = analysis["image_analysis"]
        description_analysis = analysis["description_analysis"]
        comparison = analysis["comparison"]

        reviews = generate_reviews_for_product(
            caption=image_analysis.get("caption", ""),
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

        report_html = build_report_html(
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
        )
        image_analysis = analysis["image_analysis"]
        description_analysis = analysis["description_analysis"]
        comparison = analysis["comparison"]

        reviews = generate_reviews_for_product(
            caption=image_analysis.get("caption", ""),
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
