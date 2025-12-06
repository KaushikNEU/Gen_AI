# tests/test_image_categories.py

import os
import glob
import pytest
from fastapi.testclient import TestClient
from backend.api import app  # works because of conftest.py

client = TestClient(app)

TEST_CATEGORIES = {
    "phone case": "phone_cases",
    "shoes": "shoes",
    "furniture": "furniture",
}

@pytest.mark.parametrize("category,folder", TEST_CATEGORIES.items())
def test_analyze_images_in_category(category, folder):
    image_paths = glob.glob(f"tests/data/{folder}/*.*")
    assert len(image_paths) > 0, f"No images found in {folder}"

    for img_path in image_paths:
        with open(img_path, "rb") as f:
            img_bytes = f.read()

        files = {
            "file": (os.path.basename(img_path), img_bytes, "image/jpeg")
        }
        data = {
            "description": f"Test description for {category}",
            "price": "20.0",
            "category": category,
        }

        resp = client.post("/analyze", files=files, data=data)
        assert resp.status_code == 200, f"Backend failed for {img_path}"

        result = resp.json()
        img = result["image_analysis"]

        # basic sanity checks
        assert img["caption"], f"Missing caption for {img_path}"
        assert isinstance(img["visible_features"], list)
