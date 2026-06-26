import copy
import math
import os
import random
import subprocess
import time

import requests

COMFY_URL = os.environ.get("COMFY_URL", "http://192.168.6.181:8188").rstrip("/")

# Wan 2.2 S2V aligns audio to frames on a hardcoded 16fps timeline, so the model
# always generates at this rate. Higher output fps is produced by interpolation.
S2V_GEN_FPS = 16

_NEG = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走，裸露，NSFW"
)

# ── T2V ───────────────────────────────────────────────────────────────────────

BASE_WORKFLOW = {
    "71": {"inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}, "class_type": "CLIPLoader"},
    "72": {"inputs": {"text": _NEG, "clip": ["71", 0]}, "class_type": "CLIPTextEncode"},
    "73": {"inputs": {"vae_name": "wan_2.1_vae.safetensors"}, "class_type": "VAELoader"},
    "74": {"inputs": {"width": 848, "height": 480, "length": 81, "batch_size": 1}, "class_type": "EmptyHunyuanLatentVideo"},
    "75": {"inputs": {"unet_name": "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"}, "class_type": "UNETLoader"},
    "76": {"inputs": {"unet_name": "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"}, "class_type": "UNETLoader"},
    "78": {"inputs": {"add_noise": "disable", "noise_seed": 0, "steps": 4, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "start_at_step": 2, "end_at_step": 4, "return_with_leftover_noise": "disable", "model": ["86", 0], "positive": ["89", 0], "negative": ["72", 0], "latent_image": ["81", 0]}, "class_type": "KSamplerAdvanced"},
    "80": {"inputs": {"filename_prefix": "video/ComfyUI", "format": "auto", "codec": "auto", "video-preview": "", "video": ["88", 0]}, "class_type": "SaveVideo"},
    "81": {"inputs": {"add_noise": "enable", "noise_seed": 687304397804133, "steps": 4, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "start_at_step": 0, "end_at_step": 2, "return_with_leftover_noise": "enable", "model": ["82", 0], "positive": ["89", 0], "negative": ["72", 0], "latent_image": ["74", 0]}, "class_type": "KSamplerAdvanced"},
    "82": {"inputs": {"shift": 5.0, "model": ["83", 0]}, "class_type": "ModelSamplingSD3"},
    "83": {"inputs": {"lora_name": "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors", "strength_model": 1.0, "model": ["75", 0]}, "class_type": "LoraLoaderModelOnly"},
    "85": {"inputs": {"lora_name": "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors", "strength_model": 1.0, "model": ["76", 0]}, "class_type": "LoraLoaderModelOnly"},
    "86": {"inputs": {"shift": 5.0, "model": ["85", 0]}, "class_type": "ModelSamplingSD3"},
    "87": {"inputs": {"samples": ["78", 0], "vae": ["73", 0]}, "class_type": "VAEDecode"},
    "88": {"inputs": {"fps": 16, "images": ["87", 0]}, "class_type": "CreateVideo"},
    "89": {"inputs": {"text": "", "clip": ["71", 0]}, "class_type": "CLIPTextEncode"},
}

# ── I2V ───────────────────────────────────────────────────────────────────────

_NEG_I2V = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

I2V_BASE_WORKFLOW = {
    "97":     {"inputs": {"image": ""}, "class_type": "LoadImage"},
    "108":    {"inputs": {"filename_prefix": "video/Wan2.2_image_to_video", "format": "auto", "codec": "auto", "video-preview": "", "video": ["116:94", 0]}, "class_type": "SaveVideo"},
    "116:84": {"inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}, "class_type": "CLIPLoader"},
    "116:90": {"inputs": {"vae_name": "wan_2.1_vae.safetensors"}, "class_type": "VAELoader"},
    "116:95": {"inputs": {"unet_name": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"}, "class_type": "UNETLoader"},
    "116:96": {"inputs": {"unet_name": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"}, "class_type": "UNETLoader"},
    "116:101": {"inputs": {"lora_name": "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors", "strength_model": 1.0, "model": ["116:95", 0]}, "class_type": "LoraLoaderModelOnly"},
    "116:102": {"inputs": {"lora_name": "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors", "strength_model": 1.0, "model": ["116:96", 0]}, "class_type": "LoraLoaderModelOnly"},
    "116:103": {"inputs": {"shift": 5.0, "model": ["116:102", 0]}, "class_type": "ModelSamplingSD3"},
    "116:104": {"inputs": {"shift": 5.0, "model": ["116:101", 0]}, "class_type": "ModelSamplingSD3"},
    "116:93": {"inputs": {"text": "", "clip": ["116:84", 0]}, "class_type": "CLIPTextEncode"},
    "116:89": {"inputs": {"text": _NEG_I2V, "clip": ["116:84", 0]}, "class_type": "CLIPTextEncode"},
    "116:98": {"inputs": {"width": 512, "height": 512, "length": 81, "batch_size": 1, "positive": ["116:93", 0], "negative": ["116:89", 0], "vae": ["116:90", 0], "start_image": ["97", 0]}, "class_type": "WanImageToVideo"},
    "116:86": {"inputs": {"add_noise": "enable", "noise_seed": 0, "steps": 4, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "start_at_step": 0, "end_at_step": 2, "return_with_leftover_noise": "enable", "model": ["116:104", 0], "positive": ["116:98", 0], "negative": ["116:98", 1], "latent_image": ["116:98", 2]}, "class_type": "KSamplerAdvanced"},
    "116:85": {"inputs": {"add_noise": "disable", "noise_seed": 0, "steps": 4, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "start_at_step": 2, "end_at_step": 4, "return_with_leftover_noise": "disable", "model": ["116:103", 0], "positive": ["116:98", 0], "negative": ["116:98", 1], "latent_image": ["116:86", 0]}, "class_type": "KSamplerAdvanced"},
    "116:87": {"inputs": {"samples": ["116:85", 0], "vae": ["116:90", 0]}, "class_type": "VAEDecode"},
    "116:94": {"inputs": {"fps": 16, "images": ["116:87", 0]}, "class_type": "CreateVideo"},
}

# ── S2V (Talking Head) ────────────────────────────────────────────────────────

_NEG_S2V = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

S2V_BASE_WORKFLOW = {
    "3":   {"inputs": {"seed": 764053441674844, "steps": ["103", 0], "cfg": ["105", 0], "sampler_name": "uni_pc", "scheduler": "simple", "denoise": 1, "model": ["54", 0], "positive": ["93", 0], "negative": ["93", 1], "latent_image": ["93", 2]}, "class_type": "KSampler"},
    "6":   {"inputs": {"text": "", "clip": ["38", 0]}, "class_type": "CLIPTextEncode"},
    "7":   {"inputs": {"text": _NEG_S2V, "clip": ["38", 0]}, "class_type": "CLIPTextEncode"},
    "37":  {"inputs": {"unet_name": "wan2.2_s2v_14B_fp8_scaled.safetensors", "weight_dtype": "default"}, "class_type": "UNETLoader"},
    "38":  {"inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}, "class_type": "CLIPLoader"},
    "39":  {"inputs": {"vae_name": "wan_2.1_vae.safetensors"}, "class_type": "VAELoader"},
    "52":  {"inputs": {"image": ""}, "class_type": "LoadImage"},
    "54":  {"inputs": {"shift": 8, "model": ["107", 0]}, "class_type": "ModelSamplingSD3"},
    "56":  {"inputs": {"audio_encoder": ["57", 0], "audio": ["58", 0]}, "class_type": "AudioEncoderEncode"},
    "57":  {"inputs": {"audio_encoder_name": "wav2vec2_large_english_fp16.safetensors"}, "class_type": "AudioEncoderLoader"},
    "58":  {"inputs": {"audio": ""}, "class_type": "LoadAudio"},
    "80":  {"inputs": {"samples": ["95", 0], "vae": ["39", 0]}, "class_type": "VAEDecode"},
    "82":  {"inputs": {"fps": 16, "images": ["96", 0], "audio": ["58", 0]}, "class_type": "CreateVideo"},
    "93":  {"inputs": {"width": 512, "height": 512, "length": ["104", 0], "batch_size": 1, "positive": ["6", 0], "negative": ["7", 0], "vae": ["39", 0], "audio_encoder_output": ["56", 0], "ref_image": ["52", 0]}, "class_type": "WanSoundImageToVideo"},
    "94":  {"inputs": {"dim": "t", "index": 0, "amount": 1, "samples": ["3", 0]}, "class_type": "LatentCut"},
    "95":  {"inputs": {"dim": "t", "samples1": ["94", 0], "samples2": ["3", 0]}, "class_type": "LatentConcat"},
    "96":  {"inputs": {"batch_index": ["100", 0], "length": 4096, "image": ["80", 0]}, "class_type": "ImageFromBatch"},
    "100": {"inputs": {"value": 3}, "class_type": "PrimitiveInt"},
    "103": {"inputs": {"value": 4}, "class_type": "PrimitiveInt"},
    "104": {"inputs": {"value": 77}, "class_type": "PrimitiveInt"},
    "105": {"inputs": {"value": 1}, "class_type": "PrimitiveFloat"},
    "107": {"inputs": {"lora_name": "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors", "strength_model": 1, "model": ["37", 0]}, "class_type": "LoraLoaderModelOnly"},
    "113": {"inputs": {"filename_prefix": "video/ComfyUI", "format": "auto", "codec": "auto", "video-preview": "", "video": ["82", 0]}, "class_type": "SaveVideo"},
    # Extend chunks (WanSoundImageToVideoExtend + KSampler + LatentConcat) are
    # generated dynamically in build_s2v_workflow() based on the audio duration.
}

# ── Upload helpers ────────────────────────────────────────────────────────────

def upload_image(image_bytes: bytes, filename: str) -> str:
    resp = requests.post(
        f"{COMFY_URL}/upload/image",
        data={"overwrite": "true", "type": "input"},
        files={"image": (filename, image_bytes)},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["name"]


def upload_audio(audio_bytes: bytes, filename: str) -> str:
    # ComfyUI has no /upload/audio endpoint; /upload/image accepts any file type
    resp = requests.post(
        f"{COMFY_URL}/upload/image",
        data={"overwrite": "true", "type": "input"},
        files={"image": (filename, audio_bytes)},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["name"]

# ── Workflow builders ─────────────────────────────────────────────────────────

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


def build_i2v_workflow(
    image_filename: str,
    prompt: str,
    width: int = 512,
    height: int = 512,
    length: int = 81,
    seed: int | None = None,
    fps: int = 16,
) -> dict:
    wf = copy.deepcopy(I2V_BASE_WORKFLOW)
    wf["97"]["inputs"]["image"] = image_filename
    wf["116:93"]["inputs"]["text"] = prompt
    wf["116:98"]["inputs"]["width"] = width
    wf["116:98"]["inputs"]["height"] = height
    wf["116:98"]["inputs"]["length"] = length
    wf["116:86"]["inputs"]["noise_seed"] = seed if seed is not None else random.randint(0, 2**32 - 1)
    wf["116:94"]["inputs"]["fps"] = fps
    return wf


def audio_duration_seconds(path: str) -> float:
    """Return the duration of an audio file in seconds.

    Uses soundfile's header read (fast, no full decode) and falls back to
    ffprobe for formats soundfile can't open (e.g. mp3 on some builds).
    """
    try:
        import soundfile as sf
        info = sf.info(path)
        if info.samplerate:
            return info.frames / info.samplerate
    except Exception:
        pass
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True,
    )
    return float(out.stdout.strip())


def interpolate_fps(video_bytes: bytes, src_fps: int, target_fps: int) -> bytes:
    """Motion-interpolate a video to target_fps, preserving duration and audio.

    Returns the input unchanged when target_fps == src_fps. Uses ffmpeg's
    minterpolate (motion-compensated) so lip-sync stays aligned to the audio.
    """
    if int(target_fps) == int(src_fps):
        return video_bytes
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "in.mp4")
        dst = os.path.join(d, "out.mp4")
        with open(src, "wb") as f:
            f.write(video_bytes)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
             "-vf", f"minterpolate=fps={int(target_fps)}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
             "-c:a", "copy", dst],
            check=True,
        )
        with open(dst, "rb") as f:
            return f.read()


def s2v_frames_per_chunk(chunk_length: int) -> int:
    """Real decoded frames produced per chunk.

    The node floors length to a latent count: latent_t = ((length-1)//4)+1,
    and emits latent_t*4 frames. So 41..44 all yield 44 frames, etc.
    """
    latent_t = ((chunk_length - 1) // 4) + 1
    return latent_t * 4


def plan_s2v(duration_seconds: float, fps: int, chunk_length: int) -> tuple[int, int]:
    """Plan chunk count and trim length for a given audio duration.

    Returns (num_chunks, trim_frames). num_chunks is rounded up so the
    generated video is at least as long as the audio (never truncated);
    trim_frames is the exact audio length in frames so the output can be
    trimmed to the audio, discarding any over-generated tail.
    """
    target = math.ceil(duration_seconds * fps)
    fpc = s2v_frames_per_chunk(chunk_length)
    # Generated frames = num_chunks*fpc - 2 (see build_s2v_workflow); ensure >= target.
    num_chunks = max(1, math.ceil((target + 2) / fpc))
    return num_chunks, target


def chunks_for_audio(duration_seconds: float, fps: int, chunk_length: int) -> int:
    """Backward-compatible shim: chunk count only (see plan_s2v)."""
    return plan_s2v(duration_seconds, fps, chunk_length)[0]


def build_s2v_workflow(
    image_filename: str,
    audio_filename: str,
    prompt: str = "",
    seed: int | None = None,
    fps: int = 16,
    chunk_length: int = 77,
    num_chunks: int = 3,
    trim_frames: int | None = None,
    width: int = 512,
    height: int = 512,
) -> dict:
    wf = copy.deepcopy(S2V_BASE_WORKFLOW)
    wf["52"]["inputs"]["image"] = image_filename
    wf["58"]["inputs"]["audio"] = audio_filename
    wf["6"]["inputs"]["text"] = prompt
    base_seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    wf["3"]["inputs"]["seed"] = base_seed
    wf["104"]["inputs"]["value"] = chunk_length
    wf["82"]["inputs"]["fps"] = fps
    # Resolution is set on the initial chunk (node 93); Extend nodes inherit it
    # from the previous chunk's latent shape.
    wf["93"]["inputs"]["width"] = width
    wf["93"]["inputs"]["height"] = height
    # Trim the decoded frames to the exact audio length (drop over-generated tail).
    if trim_frames is not None:
        wf["96"]["inputs"]["length"] = int(trim_frames)

    # Build the extend chain: chunk 1 is node "3"; each extra chunk adds an
    # Extend node + KSampler, accumulating latents via LatentConcat.
    num_chunks = max(1, int(num_chunks))
    prev_accum = "3"
    for i in range(2, num_chunks + 1):
        ext, ks, cat = f"ext{i}", f"ks{i}", f"cat{i}"
        wf[ext] = {"inputs": {
            "length": ["104", 0], "positive": ["6", 0], "negative": ["7", 0],
            "vae": ["39", 0], "video_latent": [prev_accum, 0],
            "audio_encoder_output": ["56", 0], "ref_image": ["52", 0],
        }, "class_type": "WanSoundImageToVideoExtend"}
        wf[ks] = {"inputs": {
            "seed": (base_seed + i) % (2**32), "steps": ["103", 0], "cfg": ["105", 0],
            "sampler_name": "uni_pc", "scheduler": "simple", "denoise": 1,
            "model": ["54", 0], "positive": [ext, 0], "negative": [ext, 1],
            "latent_image": [ext, 2],
        }, "class_type": "KSampler"}
        wf[cat] = {"inputs": {
            "dim": "t", "samples1": [prev_accum, 0], "samples2": [ks, 0],
        }, "class_type": "LatentConcat"}
        prev_accum = cat

    # Point the final trim/concat at the fully accumulated latent.
    wf["94"]["inputs"]["samples"] = [prev_accum, 0]
    wf["95"]["inputs"]["samples2"] = [prev_accum, 0]
    return wf

# ── FLF2V (6-keyframe) ────────────────────────────────────────────────────────

_NEG_FLF = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

def _flf_segment(prefix: str, start_ref: str, end_ref: str, neg_ref: str, decode_out: str) -> dict:
    """Build one FLF2V segment block (dual-pass LightX2V I2V LoRAs)."""
    return {
        f"{prefix}:clip":  {"inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}, "class_type": "CLIPLoader"},
        f"{prefix}:neg":   {"inputs": {"text": _NEG_FLF, "clip": [f"{prefix}:clip", 0]}, "class_type": "CLIPTextEncode"},
        f"{prefix}:pos":   {"inputs": {"text": "", "clip": [f"{prefix}:clip", 0]}, "class_type": "CLIPTextEncode"},
        f"{prefix}:vae":   {"inputs": {"vae_name": "wan_2.1_vae.safetensors"}, "class_type": "VAELoader"},
        f"{prefix}:uhi":   {"inputs": {"unet_name": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"}, "class_type": "UNETLoader"},
        f"{prefix}:ulo":   {"inputs": {"unet_name": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"}, "class_type": "UNETLoader"},
        f"{prefix}:lhi":   {"inputs": {"lora_name": "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors", "strength_model": 1, "model": [f"{prefix}:uhi", 0]}, "class_type": "LoraLoaderModelOnly"},
        f"{prefix}:llo":   {"inputs": {"lora_name": "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors", "strength_model": 1, "model": [f"{prefix}:ulo", 0]}, "class_type": "LoraLoaderModelOnly"},
        f"{prefix}:mhi":   {"inputs": {"shift": 5, "model": [f"{prefix}:lhi", 0]}, "class_type": "ModelSamplingSD3"},
        f"{prefix}:mlo":   {"inputs": {"shift": 5, "model": [f"{prefix}:llo", 0]}, "class_type": "ModelSamplingSD3"},
        f"{prefix}:flf":   {"inputs": {"width": 960, "height": 544, "length": 33, "batch_size": 1, "positive": [f"{prefix}:pos", 0], "negative": [f"{prefix}:neg", 0], "vae": [f"{prefix}:vae", 0], "start_image": [start_ref, 0], "end_image": [end_ref, 0]}, "class_type": "WanFirstLastFrameToVideo"},
        f"{prefix}:khi":   {"inputs": {"add_noise": "enable", "noise_seed": 0, "steps": 4, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "start_at_step": 0, "end_at_step": 2, "return_with_leftover_noise": "enable", "model": [f"{prefix}:mhi", 0], "positive": [f"{prefix}:flf", 0], "negative": [f"{prefix}:flf", 1], "latent_image": [f"{prefix}:flf", 2]}, "class_type": "KSamplerAdvanced"},
        f"{prefix}:klo":   {"inputs": {"add_noise": "disable", "noise_seed": 0, "steps": 4, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "start_at_step": 2, "end_at_step": 10000, "return_with_leftover_noise": "disable", "model": [f"{prefix}:mlo", 0], "positive": [f"{prefix}:flf", 0], "negative": [f"{prefix}:flf", 1], "latent_image": [f"{prefix}:khi", 0]}, "class_type": "KSamplerAdvanced"},
        decode_out:        {"inputs": {"samples": [f"{prefix}:klo", 0], "vae": [f"{prefix}:vae", 0]}, "class_type": "VAEDecode"},
    }

FLF2V_MIN_KEYFRAMES = 2
FLF2V_MAX_KEYFRAMES = 10


def build_flf2v_workflow(
    image_filenames: list[str],  # 2..10 keyframes
    prompt: str = "",
    width: int = 960,
    height: int = 544,
    frames_per_segment: int = 33,
    seed: int | None = None,
    fps: int = 16,
) -> dict:
    n = len(image_filenames)
    if not (FLF2V_MIN_KEYFRAMES <= n <= FLF2V_MAX_KEYFRAMES):
        raise ValueError(
            f"Expected {FLF2V_MIN_KEYFRAMES}-{FLF2V_MAX_KEYFRAMES} keyframe images, got {n}")
    base_seed = seed if seed is not None else random.randint(0, 2**32 - 1)

    wf: dict = {}
    # N keyframe LoadImage nodes.
    kf_nodes = [f"kf{i}" for i in range(1, n + 1)]
    for node_id, filename in zip(kf_nodes, image_filenames):
        wf[node_id] = {"inputs": {"image": filename}, "class_type": "LoadImage"}

    # N-1 segments between consecutive keyframe pairs.
    seg_prefixes = [f"s{i}" for i in range(1, n)]
    seg_decode = [f"{p}:dec" for p in seg_prefixes]
    pairs = list(zip(kf_nodes, kf_nodes[1:]))
    for i, (prefix, decode_out, (start_ref, end_ref)) in enumerate(zip(seg_prefixes, seg_decode, pairs)):
        wf.update(_flf_segment(prefix, start_ref, end_ref, f"{prefix}:neg", decode_out))
        wf[f"{prefix}:pos"]["inputs"]["text"] = prompt
        wf[f"{prefix}:flf"]["inputs"]["width"] = width
        wf[f"{prefix}:flf"]["inputs"]["height"] = height
        wf[f"{prefix}:flf"]["inputs"]["length"] = frames_per_segment
        wf[f"{prefix}:khi"]["inputs"]["noise_seed"] = (base_seed + i) % (2**32)

    # Concat decoded segments into one image batch (N-2 ImageBatch nodes).
    # With a single segment there's nothing to batch; feed it straight to video.
    final_images = [seg_decode[0], 0]
    prev = seg_decode[0]
    for j in range(1, len(seg_decode)):
        bn = f"batch{j}"
        wf[bn] = {"inputs": {"image1": [prev, 0], "image2": [seg_decode[j], 0]}, "class_type": "ImageBatch"}
        prev = bn
        final_images = [bn, 0]

    wf["cv"] = {"inputs": {"fps": fps, "images": final_images}, "class_type": "CreateVideo"}
    wf["save"] = {"inputs": {"filename_prefix": "video/flf2v", "format": "auto", "codec": "auto", "video-preview": "", "video": ["cv", 0]}, "class_type": "SaveVideo"}
    return wf

# ── Core API ──────────────────────────────────────────────────────────────────

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

# ── High-level generate functions ─────────────────────────────────────────────

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
    return download_video(video_info), video_info["filename"]


def generate_i2v(
    image_bytes: bytes,
    image_filename: str,
    prompt: str,
    width: int = 512,
    height: int = 512,
    length: int = 81,
    seed: int | None = None,
    fps: int = 16,
    on_progress=None,
) -> tuple[bytes, str]:
    stored_name = upload_image(image_bytes, image_filename)
    wf = build_i2v_workflow(stored_name, prompt, width, height, length, seed, fps)
    prompt_id = submit(wf)
    data = poll(prompt_id, on_progress=on_progress)
    video_info = data["outputs"]["108"]["images"][0]
    return download_video(video_info), video_info["filename"]


def generate_s2v(
    image_bytes: bytes,
    image_filename: str,
    audio_bytes: bytes,
    audio_filename: str,
    prompt: str = "",
    seed: int | None = None,
    output_fps: int = 16,
    chunk_length: int = 77,
    width: int = 512,
    height: int = 512,
    num_chunks: int | None = None,
    on_progress=None,
) -> tuple[bytes, str]:
    # S2V always generates at the native 16fps; output_fps>16 is interpolated.
    trim_frames = None
    if num_chunks is None:
        # Match the video length to the audio: enough chunks to cover it, then
        # trim the exact audio length so there's no over-generated silent tail.
        import tempfile
        suffix = os.path.splitext(audio_filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            duration = audio_duration_seconds(tmp.name)
        num_chunks, trim_frames = plan_s2v(duration, S2V_GEN_FPS, chunk_length)
    stored_image = upload_image(image_bytes, image_filename)
    stored_audio = upload_audio(audio_bytes, audio_filename)
    wf = build_s2v_workflow(stored_image, stored_audio, prompt, seed, S2V_GEN_FPS, chunk_length,
                            num_chunks, trim_frames, width=width, height=height)
    prompt_id = submit(wf)
    data = poll(prompt_id, on_progress=on_progress)
    video_info = data["outputs"]["113"]["images"][0]
    video_bytes = download_video(video_info)
    video_bytes = interpolate_fps(video_bytes, S2V_GEN_FPS, output_fps)
    return video_bytes, video_info["filename"]


def generate_flf2v(
    image_bytes_list: list[bytes],
    image_filenames: list[str],
    prompt: str = "",
    width: int = 960,
    height: int = 544,
    frames_per_segment: int = 33,
    seed: int | None = None,
    fps: int = 16,
    on_progress=None,
) -> tuple[bytes, str]:
    stored = [upload_image(b, f) for b, f in zip(image_bytes_list, image_filenames)]
    wf = build_flf2v_workflow(stored, prompt, width, height, frames_per_segment, seed, fps)
    prompt_id = submit(wf)
    data = poll(prompt_id, timeout=900.0, on_progress=on_progress)
    video_info = data["outputs"]["save"]["images"][0]
    return download_video(video_info), video_info["filename"]
