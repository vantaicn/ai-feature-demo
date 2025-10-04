import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Tải model
model_id = "meta-llama/Llama-3.1-8B-Instruct"  # hoặc model nhỏ hơn nếu bị OOM
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    device_map="auto",        # tự động chọn GPU/CPU
    torch_dtype="auto"        # tự chọn dtype (FP16 nếu GPU hỗ trợ)
)

# Hàm trả lời
def chat(user_input, history):
    messages = [{"role": "system", "content": "Bạn là một trợ lý AI hữu ích."}]
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": user_input})

    response = pipe(messages, max_new_tokens=300, do_sample=True, temperature=0.7)
    answer = response[0]["generated_text"][-1]["content"] if isinstance(response[0]["generated_text"], list) else response[0]["generated_text"]

    history.append((user_input, answer))
    return history, history

# UI Gradio
with gr.Blocks() as demo:
    gr.Markdown("# 🚀 Chatbot chạy bằng LLaMA trên Hugging Face Space")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Nhập tin nhắn...")
    clear = gr.Button("Xoá hội thoại")

    state = gr.State([])

    msg.submit(chat, [msg, state], [chatbot, state])
    clear.click(lambda: ([], []), None, [chatbot, state], queue=False)

demo.launch()
