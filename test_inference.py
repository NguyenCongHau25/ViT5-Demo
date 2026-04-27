from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
import torch

tokenizer = T5Tokenizer.from_pretrained("backend/checkpoint", local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained("backend/checkpoint", local_files_only=True)

input_text = "dịch: Chao ơi thằng con. Ri bơ nhớ thêm, nhớ đoạ chơ có bớt nhớ mô"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

print("Tokenized view:", tokenizer.convert_ids_to_tokens(input_ids[0]))

outputs = model.generate(input_ids, max_length=50, num_beams=2)
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Output text:", output_text)