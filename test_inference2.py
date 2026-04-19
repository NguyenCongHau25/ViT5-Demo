from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
import torch

tokenizer = T5Tokenizer.from_pretrained("backend/checkpoint", local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained("backend/checkpoint", local_files_only=True)

print("PAD ID:", tokenizer.pad_token_id)
print("EOS:", tokenizer.eos_token_id)
print("BOS:", tokenizer.bos_token_id)
print("Model PAD:", model.config.pad_token_id)
print("Model EOS:", model.config.eos_token_id)
print("Model BOS/Decoder Start:", model.config.decoder_start_token_id)

input_text = "dịch: nho tôi cho co bot nho miệng"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
print("Input IDs:", input_ids)

outputs = model.generate(input_ids, max_length=50, num_beams=4, early_stopping=True)
print("Output IDs:", outputs)

output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Final Output Text:", output_text)
