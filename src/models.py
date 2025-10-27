# src/models.py (NEW FILE - SIMPLE VERSION)
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from config.constants import MODEL_NAME, MODEL_QUANTIZATION, MODEL_TEMPERATURE

class AssessmentModel:
    """Single model for all assessment tasks"""
    _instance = None
    _model = None
    _tokenizer = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load(self):
        """Load model once and cache it"""
        if self._model is not None:
            print(f"✅ Model already loaded: {MODEL_NAME}")
            return
        
        print(f"🔄 Loading model: {MODEL_NAME}")
        
        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Load model
        self._model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        print(f"✅ Model loaded successfully")
    
    def generate(self, prompt: str, max_tokens: int = MODEL_MAX_TOKENS, 
                 temperature: float = MODEL_TEMPERATURE) -> str:
        """Generate response"""
        if self._model is None:
            self.load()
        
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.95,
                pad_token_id=self._tokenizer.eos_token_id
            )
        
        generated = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from output
        response = generated[len(prompt):].strip()
        return response

# Global instance
model = AssessmentModel()