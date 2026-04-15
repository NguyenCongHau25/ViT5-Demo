from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "checkpoint"
TOKENIZER_PATH = BASE_DIR / "local_tokenizer"

try:
    print(f"Loading tokenizer from {TOKENIZER_PATH}...")
    tokenizer = T5Tokenizer.from_pretrained(str(TOKENIZER_PATH), local_files_only=True)
    print(f"Loading model from {MODEL_PATH}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_PATH), local_files_only=True)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    tokenizer = None
    model = None


class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 50


@app.get("/")
def root():
    return {
        "message": "Dialect translation API is running.",
        "model_loaded": tokenizer is not None and model is not None,
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": tokenizer is not None and model is not None}


@app.post("/generate")
def generate(request: GenerationRequest):
    if tokenizer is None or model is None:
        return {"error": "Model not loaded properly on the server."}

    # Prefix required by the fine-tuned model
    input_text = "dịch: " + request.prompt

    input_ids = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True).input_ids
    outputs = model.generate(input_ids, max_length=request.max_length, num_beams=4, early_stopping=True)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"result": output_text}
