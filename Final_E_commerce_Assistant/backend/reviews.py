# backend/reviews.py

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, TypedDict

from typing_extensions import Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langgraph.graph import StateGraph, END

# -----------------------------------------
# 1. Review State for LangGraph
# -----------------------------------------

class ReviewState(TypedDict, total=False):
    # Input
    review_text: str
    rating: int
    product_id: str
    timestamp: str

    # Intermediate
    sentiment: str
    themes: List[str]

    # Output
    response: str
    errors: List[str]
    metadata: Dict[str, Any]


# -----------------------------------------
# 2. LLM + prompt for review responses
# -----------------------------------------

review_llm = ChatOpenAI(
    model="gpt-4o-mini",   # swap to your fine-tuned model ID if you want
    temperature=0.7,
    max_tokens=300,
)

def build_review_prompt() -> ChatPromptTemplate:
    system_template = """
You are Aurora, a professional customer service AI assistant.

Your job:
1) Analyze the customer review.
2) Classify sentiment as "positive", "mixed" or "negative".
3) Extract 2-4 key themes (e.g., quality, comfort, delivery, price, design).
4) Write a warm, empathetic reply.

RULES:
- ALWAYS thank the customer.
- For ratings 4-5: express appreciation.
- For rating 3: acknowledge good + address concerns.
- For ratings 1-2: apologize AND include an email like support@example.com.
- Reply length MUST be 60-75 words.

RETURN JSON ONLY in this format:
{{
  "sentiment": "positive | mixed | negative",
  "themes": ["theme1", "theme2"],
  "response": "your reply text here"
}}
""".strip()

    human_template = """
Rating: {rating} stars
Review: {review_text}

Analyze the review and respond following the rules.
""".strip()

    return ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template),
        ]
    )

review_prompt = build_review_prompt()


# -----------------------------------------
# 3. LangGraph nodes
# -----------------------------------------

class ReviewNodes:
    def generate(self, state: ReviewState) -> ReviewState:
        """Generate sentiment, themes, and response with the LLM."""
        rating = state["rating"]
        review_text = state["review_text"]

        chain = review_prompt | review_llm
        raw = chain.invoke({"rating": rating, "review_text": review_text}).content

        if isinstance(raw, list):
            raw_text = "".join(str(x) for x in raw)
        else:
            raw_text = str(raw)

        # Try to parse JSON
        try:
            clean = raw_text.strip()
            # strip accidental code fences
            if clean.startswith("```"):
                clean = clean.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean)
        except json.JSONDecodeError:
            data = {
                "sentiment": "unknown",
                "themes": [],
                "response": clean,
            }

        new_state: ReviewState = dict(state)
        new_state["sentiment"] = data.get("sentiment", "unknown")
        new_state["themes"] = data.get("themes", [])
        new_state["response"] = data.get("response", "").strip()
        # ensure errors + metadata exist
        new_state.setdefault("errors", [])
        new_state.setdefault("metadata", {})
        return new_state

    def validate(self, state: ReviewState) -> ReviewState:
        """Validate the reply and add errors if needed."""
        errors: List[str] = []
        response = state.get("response", "") or ""
        rating = state.get("rating", 0)

        words = response.split()
        wc = len(words)

        if wc < 60:
            errors.append(f"Too short: {wc} words (need 60-75).")
        elif wc > 75:
            errors.append(f"Response too long: {wc} words (need 60-75).")

        lower = response.lower()
        if "thank" not in lower:
            errors.append("Missing explicit thanks to the customer.")

        if rating <= 2 and "support@" not in lower:
            errors.append("Low rating review must include support@example.com or similar support email.")

        # update state
        new_state: ReviewState = dict(state)
        new_state["errors"] = errors

        # track regeneration_count
        metadata = dict(new_state.get("metadata") or {})
        metadata["regeneration_count"] = metadata.get("regeneration_count", 0)
        new_state["metadata"] = metadata

        return new_state


# -----------------------------------------
# 4. Build LangGraph with feedback loop
# -----------------------------------------

def build_review_graph() -> Any:
    """
    Build and compile a LangGraph StateGraph for the review workflow.
    """
    workflow = StateGraph(ReviewState)
    nodes = ReviewNodes()

    workflow.add_node("generate", nodes.generate)
    workflow.add_node("validate", nodes.validate)

    workflow.set_entry_point("generate")
    workflow.add_edge("generate", "validate")

    def decide_next(state: ReviewState) -> Literal["regenerate", "end"]:
        errors = state.get("errors", [])
        metadata = state.get("metadata", {}) or {}
        regen_count = metadata.get("regeneration_count", 0)

        if errors and regen_count < 2:
            # increment regeneration count
            metadata["regeneration_count"] = regen_count + 1
            state["metadata"] = metadata
            return "regenerate"
        return "end"

    workflow.add_conditional_edges(
        "validate",
        decide_next,
        {
            "regenerate": "generate",
            "end": END,
        },
    )

    return workflow.compile()

# Compile once
review_app = build_review_graph()


# -----------------------------------------
# 5. Helper function: run workflow for a single review
# -----------------------------------------

def run_review_workflow(
    review_text: str,
    rating: int,
    product_id: str = "unknown",
) -> Dict[str, Any]:
    """
    Run the LangGraph review workflow and return the final state summary.
    """
    initial_state: ReviewState = {
        "review_text": review_text,
        "rating": rating,
        "product_id": product_id,
        "timestamp": datetime.utcnow().isoformat(),
        "sentiment": "",
        "themes": [],
        "response": "",
        "errors": [],
        "metadata": {"regeneration_count": 0},
    }

    final_state = review_app.invoke(initial_state)

    return {
        "review_text": final_state["review_text"],
        "rating": final_state["rating"],
        "sentiment": final_state.get("sentiment", "unknown"),
        "themes": final_state.get("themes", []),
        "response": final_state.get("response", ""),
        "validation_errors": final_state.get("errors", []),
        "metadata": final_state.get("metadata", {}),
    }


# -----------------------------------------
# 6. Simulate 3 reviews + responses for a product
# -----------------------------------------

simulation_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    max_tokens=450,
)

def simulate_reviews_and_responses(
    description: str,
    ai_caption: str,
    price: float,
    category: str,
    missing_features: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Generate 3 realistic reviews (5⭐, 3⭐, 1⭐) for a product and run the
    LangGraph workflow to produce Aurora-style responses.
    """
    missing_features = missing_features or []

    system = """
You are an e-commerce review simulator.

Given product information, generate 3 realistic customer reviews:
1) A very positive 5-star review
2) A mixed 3-star review
3) A negative 1-star review that mentions issues likely caused by missing features.

RETURN JSON ONLY in this format:
[
  {"scenario": "positive", "rating": 5, "review_text": "..."},
  {"scenario": "mixed", "rating": 3, "review_text": "..."},
  {"scenario": "negative", "rating": 1, "review_text": "..."}
]
""".strip()

    human = f"""
Product description: {description}
AI image caption: {ai_caption}
Category: {category}
Price: {price}
Missing features or gaps: {", ".join(missing_features) if missing_features else "None"}
""".strip()

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system),
            HumanMessagePromptTemplate.from_template("{info}"),
        ]
    )

    chain = prompt | simulation_llm
    raw = chain.invoke({"info": human}).content
    if isinstance(raw, list):
        raw_text = "".join(str(x) for x in raw)
    else:
        raw_text = str(raw)

    try:
        clean = raw_text.strip()
        if clean.startswith("```"):
            clean = clean.replace("```json", "").replace("```", "").strip()
        simulated = json.loads(clean)
    except json.JSONDecodeError:
        # fallback: single generic review
        simulated = [
            {"scenario": "positive", "rating": 5, "review_text": clean}
        ]

    results: List[Dict[str, Any]] = []
    for item in simulated:
        review_text = item.get("review_text", "")
        rating = int(item.get("rating", 5))
        scenario = item.get("scenario", "unknown")

        workflow_result = run_review_workflow(
            review_text=review_text,
            rating=rating,
            product_id=category,
        )

        results.append(
            {
                "scenario": scenario,
                "rating": rating,
                "review_text": review_text,
                **workflow_result,
            }
        )

    return {
        "product_context": {
            "description": description,
            "ai_caption": ai_caption,
            "price": price,
            "category": category,
            "missing_features": missing_features,
        },
        "predicted_reviews": results,
    }
