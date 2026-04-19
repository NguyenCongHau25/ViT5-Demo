from transformers import T5Tokenizer, AutoModelForSeq2SeqLM

tokenizer = T5Tokenizer.from_pretrained("backend/checkpoint", local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained("backend/checkpoint", local_files_only=True)

test_sentences = [
    "tui",
    "xinh gái",
    "khoái",
    "bữa ni",
    "đang lèm cchi",
    "nho tôi cho co bot nho miệng"
]

for s in test_sentences:
    input_ids = tokenizer("dịch: " + s, return_tensors="pt").input_ids
    out = model.generate(input_ids, max_length=50, num_beams=4)
    print(s, "->", tokenizer.decode(out[0], skip_special_tokens=True))
