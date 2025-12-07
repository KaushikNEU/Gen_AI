# tests/test_reviews_1.py

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now imports will work
from backend.reviews import run_review_workflow, simulate_reviews_and_responses

print("=" * 70)
print("TEST 1: Positive Review (5 stars)")
print("=" * 70)

result1 = run_review_workflow(
    review_text="Absolutely love this phone case! The panda design is adorable and the quality is outstanding. Fits perfectly and feels protective. Fast shipping too!",
    rating=5,
    product_id="phone-case-001"
)

print(f"Sentiment: {result1['sentiment']}")
print(f"Themes: {', '.join(result1['themes'])}")
print(f"Quality Score: {result1['quality_score']}/100")
print(f"Regeneration Count: {result1['regeneration_count']}")
print(f"\n📝 Aurora's Response:\n{result1['response']}")
print(f"\n✓ Validation Errors: {result1['validation_errors'] if result1['validation_errors'] else 'None!'}")
print()

print("=" * 70)
print("TEST 2: Mixed Review (3 stars)")
print("=" * 70)

result2 = run_review_workflow(
    review_text="The case looks nice but it's bulkier than expected. Took longer to arrive too.",
    rating=3,
    product_id="phone-case-001"
)

print(f"Sentiment: {result2['sentiment']}")
print(f"Themes: {', '.join(result2['themes'])}")
print(f"Quality Score: {result2['quality_score']}/100")
print(f"\n📝 Aurora's Response:\n{result2['response']}")
print(f"\n✓ Validation Errors: {result2['validation_errors'] if result2['validation_errors'] else 'None!'}")
print()

print("=" * 70)
print("TEST 3: Negative Review (1 star)")
print("=" * 70)

result3 = run_review_workflow(
    review_text="Terrible quality. The case cracked after two weeks. Complete waste of money.",
    rating=1,
    product_id="phone-case-001"
)

print(f"Sentiment: {result3['sentiment']}")
print(f"Themes: {', '.join(result3['themes'])}")
print(f"Quality Score: {result3['quality_score']}/100")
print(f"Regeneration Count: {result3['regeneration_count']}")
print(f"\n📝 Aurora's Response:\n{result3['response']}")
print(f"\n✓ Has support email: {'Yes ✓' if 'support@' in result3['response'] else 'No ✗'}")
print(f"✓ Has apology: {'Yes ✓' if any(word in result3['response'].lower() for word in ['apolog', 'sorry']) else 'No ✗'}")
print(f"✓ Validation Errors: {result3['validation_errors'] if result3['validation_errors'] else 'None!'}")
print()

print("=" * 70)
print("TEST 4: Simulated Product Reviews")
print("=" * 70)

simulation = simulate_reviews_and_responses(
    description="Premium pink phone case with cute panda illustration",
    ai_caption="Pink phone case featuring playful panda design",
    price=19.99,
    category="Phone Case",
    missing_features=["waterproof", "wireless charging compatible"]
)

print(f"\n📦 Product: {simulation['product_context']['category']}")
print(f"💰 Price: ${simulation['product_context']['price']}")
print(f"⚠️  Missing Features: {', '.join(simulation['product_context']['missing_features'])}")
print()

for idx, review in enumerate(simulation['predicted_reviews'], 1):
    print(f"\n--- Review {idx}: {review['scenario'].upper()} ({review['rating']}★) ---")
    print(f"Customer: \"{review['review_text']}\"")
    print(f"\nAurora: \"{review['response']}\"")
    print(f"Quality: {review['quality_score']}/100 | Sentiment: {review['sentiment']}")
    if review['validation_errors']:
        print(f"⚠️  Errors: {', '.join(review['validation_errors'])}")

print("\n" + "=" * 70)
print("✅ All tests complete!")
print("=" * 70)