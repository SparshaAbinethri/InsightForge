from dotenv import load_dotenv
import os

load_dotenv()  # loads .env from current working directory

# Optional debug (don’t print full key)
key = os.getenv("OPENAI_API_KEY")
if not key:
    raise RuntimeError("OPENAI_API_KEY not found. Put it in .env or set it in environment.")
