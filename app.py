import torch
import gradio as gr
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

MODEL_ID = "google/medgemma-4b-it"

print("Loading medical VLM model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)

def analyze_xray(image, question):
    if image is None:
        return "⚠️ **Please upload a valid chest X-ray image.**"
    if not question.strip():
        # Fallback if the user clicks submit while the box is completely empty
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
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=400)

    input_len = inputs["input_ids"].shape[-1]
    generated_tokens = outputs[0][input_len:]
    final_output = processor.decode(generated_tokens, skip_special_tokens=True).strip()
    return final_output

# Build Custom High-Resolution Layout
custom_css = """
#report_box { min-height: 480px; font-size: 15px; line-height: 1.6; }
"""

with gr.Blocks(css=custom_css, title="AI Medical X-Ray Assistant") as demo:
    gr.Markdown("# 🩻 AI Medical X-Ray Diagnostic Assistant")
    gr.Markdown("Multimodal diagnostic assistant for chest radiograph interpretation. *(Research & Academic Demo)*")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Chest Radiograph (X-Ray)", height=420)
            
            # REMOVED the 'value' argument, kept only the placeholder hint
            question_input = gr.Textbox(
                lines=2,
                placeholder="Type your clinical question here... (e.g., Is there any sign of pleural effusion?)",
                label="Clinical Question"
            )
            
            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary")
                submit_btn = gr.Button("Analyze Radiograph", variant="primary")

        with gr.Column(scale=1):
            report_output = gr.Markdown(
                label="Diagnostic Report",
                value="*Diagnostic report will appear here after clicking 'Analyze Radiograph'...*",
                elem_id="report_box"
            )

    submit_btn.click(
        fn=analyze_xray,
        inputs=[image_input, question_input],
        outputs=[report_output]
    )
    
    # UPDATED: Clicking clear now sends an empty string "" to the question box
    clear_btn.click(
        fn=lambda: (None, "", "*Diagnostic report will appear here...*"),
        outputs=[image_input, question_input, report_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)
