#!/usr/bin/env python3
"""
Combined server: Gradio UI (port 7860) + FastAPI batch API (port 8000).

Loads model once, runs both servers in the same process.
"""

import os

# --- Model Loading (shared by both servers) ---
import gc
import numpy as np
import torch
from diffusers import QwenImageEditPlusPipeline

print("[Server] Starting model load...")
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

model_cache_dir = os.environ.get("TRANSFORMERS_CACHE", "/models")
model_cache_path = os.path.join(
    model_cache_dir, "models--ovedrive--Qwen-Image-Edit-2511-4bit"
)
snapshot_path = os.path.join(
    model_cache_path, "snapshots", "4104233c114f9b7b2e9c235d72ae4d216720aaac"
)

if os.path.exists(snapshot_path):
    print(f"[Server] Loading model from node-local cache: {snapshot_path}")
    pretrained_id = snapshot_path
    load_kwargs = {"torch_dtype": dtype}
else:
    print("[Server] Loading model by ID: ovedrive/Qwen-Image-Edit-2511-4bit")
    pretrained_id = "ovedrive/Qwen-Image-Edit-2511-4bit"
    load_kwargs = {"torch_dtype": dtype, "cache_dir": model_cache_dir}

pipe = QwenImageEditPlusPipeline.from_pretrained(pretrained_id, **load_kwargs).to(
    device
)
print("[Server] Model loaded successfully")


def run_gradio():
    """Run Gradio UI on port 7860."""
    import gradio as gr
    from PIL import Image
    import random

    MAX_SEED = np.iinfo(np.int32).max

    def infer(
        images,
        prompt,
        negative_prompt="blurry, out of focus, low resolution, low detail, low sharpness, soft edges, motion blur, depth of field blur, hazy, unclear, artifact, noisy",
        seed=42,
        randomize_seed=False,
        true_guidance_scale=3.0,
        num_inference_steps=30,
        height=512,
        width=512,
        num_images_per_prompt=1,
        progress=gr.Progress(track_tqdm=True),
    ):
        if randomize_seed:
            seed = random.randint(0, MAX_SEED)

        generator = torch.Generator(device=device).manual_seed(seed)

        pil_images = []
        if images is not None:
            for item in images:
                try:
                    if isinstance(item[0], Image.Image):
                        pil_images.append(item[0].convert("RGB"))
                    elif isinstance(item[0], str):
                        pil_images.append(Image.open(item[0]).convert("RGB"))
                    elif hasattr(item, "name"):
                        pil_images.append(Image.open(item.name).convert("RGB"))
                except Exception:
                    continue

        print(f"[Gradio] Calling pipeline with prompt: '{prompt}'")

        pipeline_kwargs = {
            "image": pil_images if len(pil_images) > 0 else None,
            "prompt": prompt,
            "height": height,
            "width": width,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
            "true_cfg_scale": true_guidance_scale,
            "num_images_per_prompt": num_images_per_prompt,
        }

        image = pipe(**pipeline_kwargs).images

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

        return image, seed

    with gr.Blocks() as demo:
        with gr.Column(elem_id="col-container"):
            gr.HTML(
                '<img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_edit_logo.png" alt="Qwen-Image Logo" width="400" style="display: block; margin: 0 auto;">'
            )
            gr.Markdown(
                "[Learn more](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) about Qwen-Image-Edit-2511."
            )
            with gr.Row():
                with gr.Column():
                    input_images = gr.Gallery(
                        label="Input Images",
                        show_label=False,
                        type="pil",
                        interactive=True,
                        height=400,
                        preview=True,
                    )
                result = gr.Gallery(
                    label="Result",
                    show_label=False,
                    type="pil",
                    height=400,
                    preview=True,
                    format="png",
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
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value="blurry, out of focus, low resolution, low detail, low sharpness, soft edges, motion blur, depth of field blur, hazy, unclear, artifact, noisy",
                    lines=2,
                )
                with gr.Row():
                    seed = gr.Slider(
                        label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=42
                    )
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                with gr.Row():
                    true_guidance_scale = gr.Slider(
                        label="Guidance scale",
                        minimum=1.0,
                        maximum=10.0,
                        step=0.1,
                        value=3.0,
                    )
                    num_inference_steps = gr.Slider(
                        label="Steps", minimum=1, maximum=50, step=1, value=20
                    )
                    height = gr.Slider(
                        label="Height", minimum=256, maximum=2048, step=8, value=1024
                    )
                    width = gr.Slider(
                        label="Width", minimum=256, maximum=2048, step=8, value=1024
                    )
                with gr.Row():
                    num_images_per_prompt = gr.Slider(
                        label="Images per prompt", minimum=1, maximum=4, step=1, value=1
                    )

        edit_event = gr.on(
            triggers=[run_button.click, prompt.submit],
            fn=infer,
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
            ],
            outputs=[result, seed],
            api_name="infer",
        )
        cancel_button.click(fn=None, cancels=[edit_event])

    print("[Server] Starting Gradio on port 7860...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


def run_fastapi():
    """Run FastAPI on port 8000."""
    import base64
    import io
    import random
    import time
    from typing import List, Optional

    import uvicorn
    from fastapi import FastAPI, HTTPException
    from PIL import Image
    from pydantic import BaseModel, Field

    MAX_SEED = np.iinfo(np.int32).max
    DEFAULT_NEGATIVE_PROMPT = (
        "blurry, out of focus, low resolution, low detail, low sharpness, "
        "soft edges, motion blur, depth of field blur, hazy, unclear, artifact, noisy"
    )

    class ImageInput(BaseModel):
        data: str
        filename: Optional[str] = None

    class ImageOutput(BaseModel):
        data: str
        seed: int
        index: int

    class BatchInferenceRequest(BaseModel):
        images: List[ImageInput]
        prompt: str
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
        seed: int = Field(default=42, ge=0, le=MAX_SEED)
        randomize_seed: bool = False
        guidance_scale: float = Field(default=3.0, ge=1.0, le=10.0)
        num_inference_steps: int = Field(default=30, ge=1, le=50)
        height: int = Field(default=1024, ge=256, le=2048)
        width: int = Field(default=1024, ge=256, le=2048)
        num_images_per_prompt: int = Field(default=1, ge=1, le=4)
        style_reference_mode: bool = Field(default=True)

    class BatchInferenceResponse(BaseModel):
        success: bool
        images: List[ImageOutput] = []
        total_time_seconds: float
        error: Optional[str] = None

    class HealthResponse(BaseModel):
        status: str
        model_loaded: bool
        gpu_available: bool
        gpu_memory_used_gb: Optional[float] = None
        gpu_memory_total_gb: Optional[float] = None

    def decode_base64_image(data: str) -> Image.Image:
        if data.startswith("data:"):
            _, data = data.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")

    def encode_image_base64(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    app = FastAPI(
        title="Qwen Image Edit API",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    @app.get("/")
    async def root():
        """Root endpoint for ALB health checks"""
        return {"status": "ok", "service": "qwen-model-api"}

    @app.get("/api/v1/health", response_model=HealthResponse)
    async def health_check():
        gpu_available = torch.cuda.is_available()
        gpu_mem_used = (
            torch.cuda.memory_allocated() / (1024**3) if gpu_available else None
        )
        gpu_mem_total = (
            torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_available
            else None
        )
        return HealthResponse(
            status="healthy" if pipe else "unhealthy",
            model_loaded=pipe is not None,
            gpu_available=gpu_available,
            gpu_memory_used_gb=round(gpu_mem_used, 2) if gpu_mem_used else None,
            gpu_memory_total_gb=round(gpu_mem_total, 2) if gpu_mem_total else None,
        )

    @app.post("/api/v1/batch/infer", response_model=BatchInferenceResponse)
    async def batch_infer(request: BatchInferenceRequest):
        if pipe is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        start_time = time.time()
        output_images = []

        try:
            # Decode all input images first
            pil_images = []
            for idx, img_input in enumerate(request.images):
                pil_image = decode_base64_image(img_input.data)
                pil_images.append(pil_image)
                print(f"[API] Loaded image {idx + 1}/{len(request.images)}")

            seed = (
                random.randint(0, MAX_SEED) if request.randomize_seed else request.seed
            )
            generator = torch.Generator(device=device).manual_seed(seed)

            mode_str = (
                "style reference mode" if request.style_reference_mode else "batch mode"
            )
            print(
                f"[API] Processing {len(pil_images)} images ({mode_str}), seed={seed}"
            )

            # Pass all images to pipeline at once
            result = pipe(
                image=pil_images,
                prompt=request.prompt,
                height=request.height,
                width=request.width,
                negative_prompt=request.negative_prompt,
                num_inference_steps=request.num_inference_steps,
                generator=generator,
                true_cfg_scale=request.guidance_scale,
                num_images_per_prompt=request.num_images_per_prompt,
            ).images

            print(f"[API] Received {len(result)} output images from pipeline")

            if request.style_reference_mode:
                # Return only the last image (target edit), previous images were style references
                if result:
                    final_image = result[-1]
                    output_images.append(
                        ImageOutput(
                            data=encode_image_base64(final_image),
                            seed=seed,
                            index=0,
                        )
                    )
                    print(
                        f"[API] Style reference mode: Returning final image (last of {len(result)})"
                    )
            else:
                # Return all edited images (batch mode)
                for idx, img in enumerate(result):
                    output_images.append(
                        ImageOutput(
                            data=encode_image_base64(img),
                            seed=seed,
                            index=idx,
                        )
                    )
                print(f"[API] Batch mode: Returning all {len(result)} images")

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gc.collect()

            return BatchInferenceResponse(
                success=True,
                images=output_images,
                total_time_seconds=round(time.time() - start_time, 2),
            )
        except Exception as e:
            return BatchInferenceResponse(
                success=False,
                images=[],
                total_time_seconds=round(time.time() - start_time, 2),
                error=str(e),
            )

    print("[Server] Starting FastAPI on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    # Run FastAPI only (Gradio runs in separate UI container)
    run_fastapi()
