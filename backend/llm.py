import os
import openai
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY

def enhance_with_llm(prompt: str) -> str:
    """Generates an autocomplete suggestion using GPT-4 via OpenAI API."""
    try:
        response = openai.ChatCompletion.create(  
            model="gpt-3.5-turbo",  
            messages=[
                {"role": "system", "content": "You are an AI model designed to assist with context-aware autocomplete suggestions based on provided documents."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"An error occurred: {e}"
