import copy
import os
import random
import time

import requests

COMFY_URL = os.environ.get("COMFY_URL", "http://192.168.6.181:8188").rstrip("/")

BASE_WORKFLOW = {
    "71": {
        "inputs": {
            "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "type": "wan",
            "device": "default",
        },
        "class_type": "CLIPLoader",
    },
    "72": {
        "inputs": {
            "text": (
                "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
                "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
                "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
                "杂乱的背景，三条腿，背景人很多，倒着走，裸露，NSFW"
            ),
            "clip": ["71", 0],
        },
        "class_type": "CLIPTextEncode",
    },
    "73": {
        "inputs": {"vae_name": "wan_2.1_vae.safetensors"},
        "class_type": "VAELoader",
    },
    "74": {
        "inputs": {"width": 848, "height": 480, "length": 81, "batch_size": 1},
        "class_type": "EmptyHunyuanLatentVideo",
    },
    "75": {
        "inputs": {
            "unet_name": "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
            "weight_dtype": "default",
        },
        "class_type": "UNETLoader",
    },
    "76": {
        "inputs": {
            "unet_name": "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
            "weight_dtype": "default",
        },
        "class_type": "UNETLoader",
    },
    "78": {
        "inputs": {
            "add_noise": "disable",
            "noise_seed": 0,
            "steps": 4,
            "cfg": 1,
            "sampler_name": "euler",
            "scheduler": "simple",
            "start_at_step": 2,
            "end_at_step": 4,
            "return_with_leftover_noise": "disable",
            "model": ["86", 0],
            "positive": ["89", 0],
            "negative": ["72", 0],
            "latent_image": ["81", 0],
        },
        "class_type": "KSamplerAdvanced",
    },
    "80": {
        "inputs": {
            "filename_prefix": "video/ComfyUI",
            "format": "auto",
            "codec": "auto",
            "video-preview": "",
            "video": ["88", 0],
        },
        "class_type": "SaveVideo",
    },
    "81": {
        "inputs": {
            "add_noise": "enable",
            "noise_seed": 687304397804133,
            "steps": 4,
            "cfg": 1,
            "sampler_name": "euler",
            "scheduler": "simple",
            "start_at_step": 0,
            "end_at_step": 2,
            "return_with_leftover_noise": "enable",
            "model": ["82", 0],
            "positive": ["89", 0],
            "negative": ["72", 0],
            "latent_image": ["74", 0],
        },
        "class_type": "KSamplerAdvanced",
    },
    "82": {
        "inputs": {"shift": 5.0, "model": ["83", 0]},
        "class_type": "ModelSamplingSD3",
    },
    "83": {
        "inputs": {
            "lora_name": "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors",
            "strength_model": 1.0,
            "model": ["75", 0],
        },
        "class_type": "LoraLoaderModelOnly",
    },
    "85": {
        "inputs": {
            "lora_name": "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors",
            "strength_model": 1.0,
            "model": ["76", 0],
        },
        "class_type": "LoraLoaderModelOnly",
    },
    "86": {
        "inputs": {"shift": 5.0, "model": ["85", 0]},
        "class_type": "ModelSamplingSD3",
    },
    "87": {
        "inputs": {"samples": ["78", 0], "vae": ["73", 0]},
        "class_type": "VAEDecode",
    },
    "88": {
        "inputs": {"fps": 16, "images": ["87", 0]},
        "class_type": "CreateVideo",
    },
    "89": {
        "inputs": {
            "text": "",
            "clip": ["71", 0],
        },
        "class_type": "CLIPTextEncode",
    },
}


def build_workflow(
    prompt: str,
    width: int = 848,
    height: int = 480,
    length: int = 81,
    seed: int | None = None,
    fps: int = 16,
) -> dict:
    wf = copy.deepcopy(BASE_WORKFLOW)
    wf["89"]["inputs"]["text"] = prompt
    wf["74"]["inputs"]["width"] = width
    wf["74"]["inputs"]["height"] = height
    wf["74"]["inputs"]["length"] = length
    wf["81"]["inputs"]["noise_seed"] = seed if seed is not None else random.randint(0, 2**32 - 1)
    wf["88"]["inputs"]["fps"] = fps
    return wf


def submit(workflow: dict) -> str:
    resp = requests.post(f"{COMFY_URL}/prompt", json={"prompt": workflow}, timeout=30)
    resp.raise_for_status()
    return resp.json()["prompt_id"]


def poll(
    prompt_id: str,
    interval: float = 5.0,
    timeout: float = 600.0,
    on_progress=None,
) -> dict:
    start = time.time()
    while True:
        elapsed = time.time() - start
        if elapsed > timeout:
            raise TimeoutError(f"Generation timed out after {timeout}s")
        if on_progress:
            on_progress(elapsed)
        time.sleep(interval)
        try:
            resp = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=10)
            resp.raise_for_status()
            history = resp.json()
        except requests.RequestException:
            continue
        if prompt_id in history:
            data = history[prompt_id]
            status = data.get("status", {}).get("status_str", "")
            if status == "error":
                raise RuntimeError(f"ComfyUI generation error: {data['status']}")
            return data


def download_video(output_info: dict) -> bytes:
    params = {
        "filename": output_info["filename"],
        "subfolder": output_info["subfolder"],
        "type": output_info["type"],
    }
    resp = requests.get(f"{COMFY_URL}/view", params=params, timeout=120)
    resp.raise_for_status()
    return resp.content


def generate(
    prompt: str,
    width: int = 848,
    height: int = 480,
    length: int = 81,
    seed: int | None = None,
    fps: int = 16,
    on_progress=None,
) -> tuple[bytes, str]:
    wf = build_workflow(prompt, width, height, length, seed, fps)
    prompt_id = submit(wf)
    data = poll(prompt_id, on_progress=on_progress)
    video_info = data["outputs"]["80"]["images"][0]
    video_bytes = download_video(video_info)
    return video_bytes, video_info["filename"]
