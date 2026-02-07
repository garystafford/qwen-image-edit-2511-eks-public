#!/usr/bin/env python3
"""
Gradio UI that calls the Qwen model service via FastAPI.

In EKS, API_BASE_URL is set to the cluster-internal service DNS.
For local testing, port-forward the model service first:

    kubectl port-forward -n qwen svc/qwen-model-service 8000:8000
    python app_ui.py
"""

import base64
import io
import os

import gradio as gr
import numpy as np
import requests
from PIL import Image

# --- Configuration ---
# Read API URL from environment variable, fallback to localhost for local development
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000") + "/api/v1"
MAX_SEED = np.iinfo(np.int32).max


# --- API Client Functions ---
def encode_image_base64(image: Image.Image) -> str:
    """Encode PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def decode_base64_image(data: str) -> Image.Image:
    """Decode base64 string to PIL Image"""
    image_bytes = base64.b64decode(data)
    return Image.open(io.BytesIO(image_bytes))


def call_remote_api(
    files,
    prompt,
    negative_prompt,
    seed,
    randomize_seed,
    true_guidance_scale,
    num_inference_steps,
    height,
    width,
    num_images_per_prompt,
    style_reference_mode,
):
    """
    Call the remote FastAPI endpoint for inference.
    """
    if not files:
        return [], seed

    # Ensure files is a list
    if not isinstance(files, list):
        files = [files]

    if len(files) == 0:
        return [], seed

    # Prepare images for API
    print(f"[Local] Processing {len(files)} file(s)")
    images_data = []
    for idx, file_path in enumerate(files):
        try:
            print(f"[Local] Loading image {idx + 1}/{len(files)}: {file_path}")
            img = Image.open(file_path).convert("RGB")
            img_b64 = encode_image_base64(img)
            images_data.append({"data": img_b64, "filename": file_path})
            print(f"[Local] Successfully encoded image {idx + 1}")
        except Exception as e:
            print(f"[Local] ERROR loading image {file_path}: {e}")
            continue

    if len(images_data) == 0:
        return [], seed

    # Prepare request payload
    payload = {
        "images": images_data,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "randomize_seed": randomize_seed,
        "guidance_scale": true_guidance_scale,
        "num_inference_steps": num_inference_steps,
        "height": height,
        "width": width,
        "num_images_per_prompt": num_images_per_prompt,
        "style_reference_mode": style_reference_mode,
    }

    print(f"[Local] Calling remote API with prompt: '{prompt}'")
    print(
        f"[Local] Images: {len(images_data)}, Steps: {num_inference_steps}, Style ref mode: {style_reference_mode}"
    )

    try:
        # Call remote API
        response = requests.post(
            f"{API_BASE_URL}/batch/infer",
            json=payload,
            timeout=300,  # 5 minute timeout for inference
        )

        if response.status_code != 200:
            error_msg = f"API error {response.status_code}: {response.text}"
            print(f"[Local] {error_msg}")
            raise Exception(error_msg)

        result = response.json()

        if not result["success"]:
            error_msg = result.get("error", "Unknown error")
            print(f"[Local] API returned error: {error_msg}")
            raise Exception(error_msg)

        # Decode result images
        output_images = []
        returned_seed = seed
        for img_output in result["images"]:
            img = decode_base64_image(img_output["data"])
            output_images.append(img)
            returned_seed = img_output["seed"]  # Use last seed

        print(
            f"[Local] Received {len(output_images)} images in {result['total_time_seconds']:.2f}s"
        )

        return output_images, returned_seed

    except requests.exceptions.ConnectionError:
        error_msg = (
            "Could not connect to remote API. "
            "Make sure port-forward is running:\n"
            "kubectl port-forward -n qwen svc/qwen-image-edit 8000:8000"
        )
        print(f"[Local] {error_msg}")
        raise Exception(error_msg)
    except requests.exceptions.Timeout:
        error_msg = "Request timed out after 5 minutes"
        print(f"[Local] {error_msg}")
        raise Exception(error_msg)
    except Exception as e:
        print(f"[Local] Error: {str(e)}")
        raise


def preview_uploaded_images(files):
    """Convert uploaded files to PIL images for preview gallery"""
    if not files:
        return []

    # Ensure files is a list
    if not isinstance(files, list):
        files = [files]

    if len(files) == 0:
        return []

    images = []
    print(f"[Local Preview] Loading {len(files)} file(s) for preview")
    for idx, file in enumerate(files):
        try:
            if isinstance(file, str):
                print(f"[Local Preview] Loading file {idx + 1}: {file}")
                images.append(Image.open(file))
            elif hasattr(file, "name"):
                print(f"[Local Preview] Loading file {idx + 1}: {file.name}")
                images.append(Image.open(file.name))
        except Exception as e:
            print(f"[Local Preview] ERROR loading image {idx + 1}: {e}")
            continue

    print(f"[Local Preview] Successfully loaded {len(images)} image(s)")
    return images


def check_api_health():
    """Check if the remote API is accessible"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            status_emoji = "✅" if health["status"] == "healthy" else "⚠️"
            gpu_emoji = "🟢" if health["gpu_available"] else "🔴"
            return f"{status_emoji} API Status: {health['status']} | {gpu_emoji} GPU: {'Available' if health['gpu_available'] else 'Unavailable'}"
        else:
            return "❌ API Status: Error"
    except Exception:
        return "❌ API Status: Not Connected (run: kubectl port-forward -n qwen svc/qwen-image-edit 8000:8000)"


# --- Gradio UI ---
css = """
#col-container {
    margin: 0 auto;
    max-width: 1024px;
}
#edit_text {
    margin-top: -62px !important;
}
#api-status {
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 10px;
    text-align: center;
    font-family: monospace;
}
.center {
    text-align: center;
}
.hint-text {
    font-size: 0.85em;
    color: #888;
    margin-top: 5px;
    margin-bottom: 5px;
}
"""

with gr.Blocks() as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML(
            '<img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_edit_logo.png" '
            'alt="Qwen-Image Logo" width="400" style="display: block; margin: 0 auto;">'
        )
        gr.Markdown(
            "[Learn more](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) about Qwen-Image-Edit-2511.",
            elem_classes="center",
        )

        with gr.Row():
            with gr.Column():
                input_images = gr.File(
                    label="Upload Images",
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath",
                )
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Images", size="sm")
                input_preview = gr.Gallery(
                    label="Preview",
                    show_label=True,
                    type="pil",
                    interactive=False,
                    height=300,
                    preview=False,
                )
            with gr.Column():
                result = gr.Gallery(
                    label="Result",
                    show_label=True,
                    type="pil",
                    height=400,
                    preview=False,
                )

        prompt = gr.Text(
            label="Prompt",
            show_label=False,
            placeholder="describe the edit instruction",
            container=False,
        )

        with gr.Row():
            run_button = gr.Button("Edit!", variant="primary")
            cancel_button = gr.Button("Cancel", variant="stop")

        with gr.Accordion("Advanced Settings", open=False):
            # API status indicator (updates every 10 seconds)
            api_status = gr.Markdown(check_api_health(), elem_id="api-status", every=10)

            style_reference_mode = gr.Checkbox(
                label="Style Reference Mode",
                value=True,
                info="When enabled: all but last image = style references, returns only edited last image. When disabled: each image edited independently.",
            )

            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                placeholder="Describe what you don't want in the image",
                value="blurry, out of focus, low resolution, low detail, low sharpness, soft edges, motion blur, depth of field blur, hazy, unclear, artifact, noisy",
                lines=2,
            )

            with gr.Row():
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=42,
                )
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

            with gr.Row():
                true_guidance_scale = gr.Slider(
                    label="True guidance scale",
                    minimum=1.0,
                    maximum=10.0,
                    step=0.1,
                    value=3.0,
                )

                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=20,
                )

                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=2048,
                    step=8,
                    value=1024,
                )

                width = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=2048,
                    step=8,
                    value=1024,
                )

            with gr.Row():
                num_images_per_prompt = gr.Slider(
                    label="Number of images per prompt",
                    minimum=1,
                    maximum=4,
                    step=1,
                    value=1,
                )

    # Update preview when files are uploaded
    input_images.change(
        fn=preview_uploaded_images, inputs=[input_images], outputs=[input_preview]
    )

    # Clear button - resets file upload and preview
    clear_btn.click(
        fn=lambda: (None, []), inputs=None, outputs=[input_images, input_preview]
    )

    # Run inference
    edit_event = gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=call_remote_api,
        inputs=[
            input_images,
            prompt,
            negative_prompt,
            seed,
            randomize_seed,
            true_guidance_scale,
            num_inference_steps,
            height,
            width,
            num_images_per_prompt,
            style_reference_mode,
        ],
        outputs=[result, seed],
        api_name="infer",
    )

    cancel_button.click(fn=None, cancels=[edit_event])

if __name__ == "__main__":
    print("=" * 60)
    print("Starting Local Gradio UI")
    print("=" * 60)
    print("\nMake sure port-forward is running in another terminal:")
    print("  kubectl port-forward -n qwen svc/qwen-image-edit 8000:8000")
    print("\nAccess the UI at: http://localhost:7860")
    print("=" * 60)

    demo.launch(
        server_name="0.0.0.0",  # Accept connections from any interface (required for containers)
        server_port=7860,
        share=False,
        css=css,  # Gradio 6.0: CSS moved to launch()
    )
