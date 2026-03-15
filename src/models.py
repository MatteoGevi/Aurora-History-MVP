import ollama
import openai
import anthropic
from config.constants import MODEL_NAME, MODEL_TEMPERATURE, MODEL_MAX_TOKENS, OPENAI_API_KEY, OPENAI_MODEL_NAME, CLAUDE_API_KEY, CLAUDE_MODEL_NAME


def check_ollama():
    """Check if Ollama works by trying a quick generation"""
    try:
        ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "test"}],
            options={'num_predict': 1}
        )
        print(f"✅ Ollama ready with {MODEL_NAME}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def generate(prompt: str, max_tokens: int = None, temperature: float = None) -> str:
    """Generate text using Ollama (single-turn, no system prompt)."""
    if max_tokens is None:
        max_tokens = MODEL_MAX_TOKENS
    if temperature is None:
        temperature = MODEL_TEMPERATURE

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options={'temperature': temperature, 'num_predict': max_tokens}
    )
    return response['message']['content']


def generate_chat(system: str, user: str, max_tokens: int = None, temperature: float = None) -> str:
    """Generate with explicit system/user separation using Ollama chat API.

    Uses the model's native system-role slot instead of concatenating strings,
    which significantly improves instruction-following for structured JSON tasks.
    """
    if max_tokens is None:
        max_tokens = MODEL_MAX_TOKENS
    if temperature is None:
        temperature = MODEL_TEMPERATURE

    messages = []
    if system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    response = ollama.chat(
        model=MODEL_NAME,
        messages=messages,
        options={'temperature': temperature, 'num_predict': max_tokens}
    )
    return response['message']['content']


def generate_chat_openai(system: str, user: str, max_tokens: int = None, temperature: float = None) -> str:
    """Generate with explicit system/user separation using the OpenAI chat API."""
    if max_tokens is None:
        max_tokens = MODEL_MAX_TOKENS
    if temperature is None:
        temperature = MODEL_TEMPERATURE

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    messages = []
    if system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    response = client.chat.completions.create(
        model=OPENAI_MODEL_NAME,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


def generate_chat_claude(system: str, user: str, max_tokens: int = None, temperature: float = None) -> str:
    """Generate with explicit system/user separation using the Anthropic Claude API."""
    if max_tokens is None:
        max_tokens = MODEL_MAX_TOKENS
    if temperature is None:
        temperature = MODEL_TEMPERATURE

    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    response = client.messages.create(
        model=CLAUDE_MODEL_NAME,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )

    return next(block.text for block in response.content if block.type == "text")