from pathlib import Path
from unicodedata import normalize

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import transformers
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer

# Giới hạn số luồng CPU để kiểm soát RAM
torch.set_num_threads(4)

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
TOKENIZER_PATH = BASE_DIR / "checkpoint"

try:
    print(f"Loading tokenizer from {TOKENIZER_PATH}...")
    tokenizer = T5Tokenizer.from_pretrained(str(TOKENIZER_PATH), local_files_only=True)
    print(f"Loading model from {MODEL_PATH}...")
    # Load model với float16/bfloat16 để giảm dung lượng RAM, hoặc tuỳ fallback. 
    # Nếu chạy CPU dùng float32 mặc định hoặc có thể giảm xuống để tiết kiệm RAM.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_PATH), local_files_only=True).to(device)
    model.eval() # Chuyển sang chế độ evaluation để tối ưu inference
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    tokenizer = None
    model = None


class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 50


def get_runtime_info():
    model_loaded = tokenizer is not None and model is not None
    model_device = str(next(model.parameters()).device) if model_loaded else None
    return {
        "model_loaded": model_loaded,
        "model_path": str(MODEL_PATH),
        "tokenizer_path": str(TOKENIZER_PATH),
        "device": model_device,
        "python_transformers": transformers.__version__,
        "python_torch": torch.__version__,
    }


@app.get("/")
def root():
    return {
        "message": "Dialect translation API is running.",
        **get_runtime_info(),
    }


@app.get("/health")
def health():
    return {"status": "ok", **get_runtime_info()}


@app.post("/generate")
def generate(request: GenerationRequest):
    if tokenizer is None or model is None:
        return {"error": "Model not loaded properly on the server."}

    # Prefix required by the fine-tuned model
    normalized_prompt = normalize("NFC", request.prompt.strip())
    input_text = "dịch: " + normalized_prompt

    device = next(model.parameters()).device
    input_ids = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True).input_ids.to(device)
    
    with torch.no_grad(): # CHÚ Ý: Bắt buộc dùng no_grad khi inference để tránh rò rỉ và tràn RAM!
        # Giảm num_beams hoặc dùng greedy search (bỏ num_beams) để tăng tốc độ dịch gấp nhiều lần
        outputs = model.generate(input_ids, max_length=request.max_length, num_beams=2, early_stopping=True)
    
    output_text = normalize(
        "NFC",
        tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False),
    )

    return {"result": output_text}
