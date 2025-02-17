import os
import groq
from dotenv import load_dotenv


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


client = groq.Groq(api_key=GROQ_API_KEY)

def enhance_with_llm(prompt: str) -> str:
    """Generates an autocomplete suggestion using LLaMA-2 via Groq API."""
    response = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",  
        messages=[{"role": "user", "content": prompt}],
        max_tokens=80
    )
    return response.choices[0].message.content
