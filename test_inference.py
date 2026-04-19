from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
import torch

tokenizer = T5Tokenizer.from_pretrained("backend/checkpoint", local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained("backend/checkpoint", local_files_only=True)

input_text = "dịch: Chào ơi thằng con. Như này bo nho them, nho tôi cho co bot nho miệng"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

print("Tokenized view:", tokenizer.convert_ids_to_tokens(input_ids[0]))

outputs = model.generate(input_ids, max_length=50, num_beams=2)
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Output text:", output_text)
