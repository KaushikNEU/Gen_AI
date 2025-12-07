# tests/test_reports.py

import os
import glob
from fastapi.testclient import TestClient
from backend.api import app

client = TestClient(app)


def _get_sample_image_bytes():
    """
    Returns bytes of the first image in tests/data/phone_cases.
    Adjust the folder if you prefer another category.
    """
    image_paths = glob.glob("tests/data/phone_cases/*.*")
    assert image_paths, "No test images found in tests/data/phone_cases"
    img_path = image_paths[0]
    with open(img_path, "rb") as f:
        return os.path.basename(img_path), f.read()


def test_report_endpoint_text_and_structure():
    filename, img_bytes = _get_sample_image_bytes()

    files = {
        "file": (filename, img_bytes, "image/jpeg"),
    }
    data = {
        "description": "Cute pink phone case with a panda design for testing.",
        "price": "19.99",
        "category": "Phone Case",
    }

    resp = client.post("/report", files=files, data=data)
    assert resp.status_code == 200, "/report endpoint failed"

    result = resp.json()

    # Basic keys
    assert "report" in result, "Response missing 'report' key"
    assert "reviews" in result, "Response missing 'reviews' key"
    assert "analysis" in result, "Response missing 'analysis' key"

    # Report text checks
    report_text = result["report"]
    assert isinstance(report_text, str)
    assert len(report_text.strip()) > 50, "Report text is too short"

    # Reviews checks
    reviews = result["reviews"]
    assert isinstance(reviews, list), "reviews should be a list"
    assert len(reviews) >= 1, "Expected at least 1 review-like entry in the report"

    for rev in reviews:
        # We don't enforce all star buckets here (that’s covered in test_reviews.py)
        assert rev.get("title"), "Review title is empty in report output"
        body = rev.get("body", "")
        assert isinstance(body, str)
        assert len(body.strip()) > 40, "Review body in report is too short"

    # Analysis structure checks
    analysis = result["analysis"]
    assert "image_analysis" in analysis
    assert "description_analysis" in analysis
    assert "comparison" in analysis

    img_analysis = analysis["image_analysis"]
    assert img_analysis.get("caption"), "Image caption missing in analysis"


def test_report_pdf_endpoint_generates_valid_pdf():
    filename, img_bytes = _get_sample_image_bytes()

    files = {
        "file": (filename, img_bytes, "image/jpeg"),
    }
    data = {
        "description": "Cute pink phone case with a panda design for testing.",
        "price": "19.99",
        "category": "Phone Case",
    }

    resp = client.post("/report_pdf", files=files, data=data)
    assert resp.status_code == 200, "/report_pdf endpoint failed"

    # content-type header
    content_type = resp.headers.get("content-type", "")
    assert "application/pdf" in content_type.lower(), f"Unexpected content-type: {content_type}"

    pdf_bytes = resp.content
    assert isinstance(pdf_bytes, (bytes, bytearray))
    assert len(pdf_bytes) > 1000, f"PDF seems too small (len={len(pdf_bytes)})"
