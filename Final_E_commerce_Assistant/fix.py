from dotenv import load_dotenv
import os

load_dotenv()  # <-- THIS MUST RUN BEFORE reading env vars

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY environment variable is not set. Please export it before running the backend."
    )
