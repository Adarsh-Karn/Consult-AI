# llm_file.py

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq LLM
llm = ChatGroq(
    api_key=groq_api_key, #type: ignore
    model="llama3-8b-8192"     # or "mixtral-8x7b"
)

# Export
__all__ = ["llm"]
