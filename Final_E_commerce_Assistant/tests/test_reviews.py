# tests/test_reviews.py

import pytest
from fastapi.testclient import TestClient
from backend.api import app

client = TestClient(app)


@pytest.mark.parametrize(
    "caption,description,price,category",
    [
        (
            "pink panda phone case",
            "Cute pink panda phone case with slim fit and soft edges.",
            19.99,
            "Phone Case",
        ),
        (
            "black running shoes",
            "Lightweight black running shoes with breathable mesh and cushioned sole.",
            59.99,
            "Shoes",
        ),
        (
            "wooden dining chair",
            "Modern wooden dining chair with comfortable back support.",
            89.99,
            "Furniture",
        ),
    ],
)
def test_reviews_endpoint_basic(caption, description, price, category):
    """
    For several product captions, ensure the /reviews endpoint:
    - returns 200
    - returns a list of reviews
    - includes multiple star ratings (ideally 5,4,3,1)
    - each review has a non-empty title and a reasonably long body
    """

    data = {
        "caption": caption,
        "description": description,
        "price": str(price),
        "category": category,
    }

    resp = client.post("/reviews", data=data)
    assert resp.status_code == 200, f"/reviews failed for caption: {caption}"

    result = resp.json()
    assert "reviews" in result, "Response missing 'reviews' key"

    reviews = result["reviews"]
    assert isinstance(reviews, list), "reviews should be a list"
    assert len(reviews) >= 4, "Expected at least 4 reviews (5★,4★,3★,1★)"

    # Collect star ratings present
    ratings = {rev.get("rating") for rev in reviews}
    # Your prompt is designed to produce 5,4,3,1 – enforce that here
    assert {5, 4, 3, 1}.issubset(
        ratings
    ), f"Expected ratings 5,4,3,1 but got {ratings}"

    # Basic quality checks on each review
    for rev in reviews:
        assert rev.get("title"), "Review title is empty"
        body = rev.get("body", "")
        assert isinstance(body, str)
        assert len(body) > 40, "Review body is too short to be meaningful"
