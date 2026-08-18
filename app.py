import torch
import gradio as gr
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# Load model and processor
MODEL_ID = "google/medgemma-4b-it"

print("Loading medical VLM model...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_4bit=True
)

def analyze_xray(image, question):
    if image is None:
        return "Please upload a valid chest X-ray image."
    if not question.strip():
        question = "Analyze this chest X-ray and report any abnormalities."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]
        }
    ]

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=300)

    response = processor.decode(outputs[0], skip_special_tokens=True)
    final_output = response.split("model\n")[-1].strip()
    return final_output

# Launch Gradio Web Interface
demo = gr.Interface(
    fn=analyze_xray,
    inputs=[
        gr.Image(type="pil", label="Upload Chest Radiograph (X-Ray)"),
        gr.Textbox(
            lines=2,
            placeholder="E.g., Is there any sign of pneumonia or effusion?",
            label="Clinical Question"
        )
    ],
    outputs=gr.Textbox(label="AI Assistant Diagnostic Interpretation"),
    title="🩻 AI Medical X-Ray Assistant",
    description="Multimodal diagnostic assistant for chest X-ray interpretation. (Research Demo)"
)

if __name__ == "__main__":
    demo.launch(share=True)
