from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "context-labs/meta-llama-Llama-3.2-3B-Instruct-FP16"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto"
)

messages = [
    {
        "role": "user",
        "content": "Xin chào, bạn có biết các cấu trúc ngữ pháp nào thường hay xuất hiện trong bài thi VStep không? Hãy giúp tôi học và nắm rõ các cấu trúc đó!",
    },
]

# 1. render prompt từ chat template
prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False   # lấy ra string
)

# 2. tokenize prompt
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 3. generate text
outputs = model.generate(
    **inputs,
    max_new_tokens=500,   # tăng để tránh bị cụt
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

# 4. decode toàn bộ rồi cắt phần sinh thêm
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
reply = generated_text[len(prompt):].strip()

print(reply)
