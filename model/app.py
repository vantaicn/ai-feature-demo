# Tên file: app.py

# 1. Import các thư viện cần thiết
import gradio as gr
import torch
# !!! THÊM EulerDiscreteScheduler VÀO ĐÂY
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler 

# 2. Định nghĩa tên model SIÊU NHẸ
model_id = "segmind/tiny-sd"

# 3. Tải các thành phần của model một cách tường minh
# !!! BƯỚC QUAN TRỌNG: Tải scheduler tương thích với model
print(f"Bắt đầu tải model siêu nhẹ: {model_id}...")
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

# Tải pipeline và chỉ định scheduler đã tải
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    scheduler=scheduler,  # <--- DÒNG NÀY SẼ SỬA LỖI CỦA BẠN
    torch_dtype=torch.float32
)

# Chuyển model sang GPU nếu có
if torch.cuda.is_available():
    pipe = pipe.to("cuda")
    pipe.to(torch.float16)

print("Model đã được tải xong và sẵn sàng.")


# 4. Định nghĩa hàm xử lý chính
def generate_image(prompt):
    print(f"Nhận được yêu cầu với prompt: '{prompt}'")
    # Sử dụng generator để có kết quả nhất quán hơn
    generator = torch.Generator(device="cpu").manual_seed(42)
    if torch.cuda.is_available():
        generator = torch.Generator(device="cuda").manual_seed(42)

    image = pipe(prompt, num_inference_steps=25, generator=generator).images[0]
    
    print("Đã tạo ảnh thành công.")
    return image


# 5. Tạo giao diện người dùng với Gradio (giữ nguyên)
print("Đang tạo giao diện Gradio...")
demo = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(
        lines=2, 
        label="Nhập mô tả cho bức ảnh", 
        placeholder="Ví dụ: A high-quality photo of a robot playing chess"
    ),
    outputs=gr.Image(label="Kết quả"),
    title="🎨 Demo Tạo ảnh AI với Tiny Stable Diffusion (Siêu Nhẹ & Nhanh)",
    description="Đây là demo sử dụng một model AI siêu nhẹ, tối ưu để chạy nhanh trên các phần cứng miễn phí.",
    examples=[
        ["A cinematic shot of a baby raccoon wearing a tiny top hat"],
        ["A photo of a white cat sleeping on a pile of books"],
        ["An astronaut riding a horse on Mars, hd, dramatic lighting"]
    ]
)

# 6. Chạy ứng dụng và bật chế độ báo lỗi chi tiết
demo.launch(show_error=True)