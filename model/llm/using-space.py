import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Model nhỏ hơn để chạy được trên Space
MODEL_ID = "context-labs/meta-llama-Llama-3.2-3B-Instruct-FP16"

# Load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",   # tự động dùng GPU nếu có
    torch_dtype="auto"
)

# Tạo pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512
)

# Hàm chat
def chat_fn(message, history):
    # format lại hội thoại cho giống chat
    prompt = ""
    for user, bot in history:
        prompt += f"User: {user}\nAssistant: {bot}\n"
    prompt += f"User: {message}\nAssistant:"

    outputs = pipe(prompt, do_sample=True, temperature=0.7, top_p=0.9)
    reply = outputs[0]["generated_text"][len(prompt):]
    return reply.strip()

# Hàm Gradio
def respond(message, history):
    reply = chat_fn(message, history)
    history.append((message, reply))
    return history, history

# UI
with gr.Blocks() as demo:
    gr.Markdown("# 💬 Chat với LLaMA 3.2 3B FP16")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Nhập tin nhắn...")
    clear = gr.Button("Xóa hội thoại")

    state = gr.State([])

    msg.submit(respond, [msg, state], [chatbot, state])
    clear.click(lambda: ([], []), None, [chatbot, state], queue=False)

demo.launch()
