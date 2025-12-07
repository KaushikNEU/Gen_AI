# tests/test_pdf_generation.py

import os
import glob
from backend.core import build_report_pdf


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


def test_build_report_pdf_without_image():
    """
    Unit test: build_report_pdf should generate a valid-looking PDF
    when given only text (no image).
    """
    report_text = (
        "=== PRODUCT REPORT ===\n\n"
        "This is a short test report to verify PDF generation.\n"
        "It should produce a valid PDF file even without an image.\n"
        "The content can span multiple lines and should be wrapped correctly."
    )

    pdf_bytes = build_report_pdf(report_text=report_text, image_bytes=None)

    # Basic type + size checks
    assert isinstance(pdf_bytes, (bytes, bytearray)), "PDF output should be bytes"
    assert len(pdf_bytes) > 500, "PDF seems too small; generation might have failed"

    # Check PDF signature
    assert pdf_bytes[:4] == b"%PDF", "PDF does not start with %PDF header"


def test_build_report_pdf_with_image():
    """
    Unit test: build_report_pdf should also generate a valid PDF
    when an image is provided along with the text report.
    """
    filename, img_bytes = _get_sample_image_bytes()

    report_text = (
        "=== PRODUCT REPORT (WITH IMAGE) ===\n\n"
        f"Image file used: {filename}\n"
        "This report includes both an image and some body text to verify\n"
        "that PDF generation, layout, and word wrapping work correctly."
    )

    pdf_bytes = build_report_pdf(report_text=report_text, image_bytes=img_bytes)

    # Basic type + size checks
    assert isinstance(pdf_bytes, (bytes, bytearray)), "PDF output should be bytes"
    assert len(pdf_bytes) > 1000, "PDF with image seems too small; generation might have failed"

    # Check PDF signature
    assert pdf_bytes[:4] == b"%PDF", "PDF does not start with %PDF header"
