import os
import groq
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = groq.Groq(api_key=GROQ_API_KEY)

def enhance_with_llm(prompt: str) -> str:
    """Generates an autocomplete suggestion using LLaMA-2 via Groq API."""
    response = client.chat.completions.create(
        model="llama2-70b-chat",  # Use "mixtral-8x7b" if you prefer Mixtral
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50
    )
    return response.choices[0].message.content
