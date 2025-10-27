# src/models.py - SIMPLE FUNCTION VERSION
import ollama
from config.constants import MODEL_NAME, MODEL_TEMPERATURE, MODEL_MAX_TOKENS

def check_ollama():
    """Check if Ollama works by trying a quick generation"""
    try:
        ollama.generate(
            model=MODEL_NAME,
            prompt="test",
            options={'num_predict': 1}
        )
        print(f"✅ Ollama ready with {MODEL_NAME}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def generate(prompt: str, max_tokens: int = None, temperature: float = None) -> str:
    """Generate text using Ollama"""
    if max_tokens is None:
        max_tokens = MODEL_MAX_TOKENS
    if temperature is None:
        temperature = MODEL_TEMPERATURE
    
    response = ollama.generate(
        model=MODEL_NAME,
        prompt=prompt,
        options={
            'temperature': temperature,
            'num_predict': max_tokens
        }
    )
    
    return response['response']