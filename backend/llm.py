from transformers import AutoModelForCausalLM, AutoTokenizer

# Load LLaMA model
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def enhance_with_llm(prompt: str) -> str:
    """Generates an improved autocomplete suggestion using LLaMA."""
    input_tokens = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**input_tokens, max_length=50)
    return tokenizer.decode(output[0], skip_special_tokens=True)
