import os
from pathlib import Path

from fastmcp import FastMCP
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

import comfy_client

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

mcp = FastMCP("wan2-t2v")


@mcp.tool()
def generate_wan_video(
    prompt: str,
    width: int = 848,
    height: int = 480,
    length: int = 81,
    seed: int = -1,
    fps: int = 16,
) -> dict:
    """Generate a video from a text prompt using Wan 2.2 14B via ComfyUI.

    Args:
        prompt: Text description of the video to generate.
        width: Frame width in pixels (default 848).
        height: Frame height in pixels (default 480).
        length: Number of frames to generate (default 81, ~5s at 16fps).
        seed: Random seed; -1 for a random seed.
        fps: Output video frames per second (default 16).

    Returns:
        dict with 'path' (absolute path to saved video) and 'filename'.
    """
    resolved_seed = None if seed < 0 else seed
    video_bytes, filename = comfy_client.generate(
        prompt=prompt,
        width=width,
        height=height,
        length=length,
        seed=resolved_seed,
        fps=fps,
    )
    out_path = OUTPUT_DIR / filename
    out_path.write_bytes(video_bytes)
    return {"path": str(out_path.resolve()), "filename": filename}


@mcp.tool()
def generate_wan_flf2v(
    frame1_path: str,
    frame2_path: str,
    frame3_path: str,
    frame4_path: str,
    frame5_path: str,
    frame6_path: str,
    prompt: str = "",
    width: int = 960,
    height: int = 544,
    frames_per_segment: int = 33,
    seed: int = -1,
    fps: int = 16,
) -> dict:
    """Generate a video by interpolating between 6 keyframes using Wan 2.2 FLF2V via ComfyUI.

    Generates 5 segments between consecutive keyframe pairs and concatenates them.

    Args:
        frame1_path: Absolute path to keyframe 1 (start).
        frame2_path: Absolute path to keyframe 2.
        frame3_path: Absolute path to keyframe 3.
        frame4_path: Absolute path to keyframe 4.
        frame5_path: Absolute path to keyframe 5.
        frame6_path: Absolute path to keyframe 6 (end).
        prompt: Optional text prompt applied to all segments.
        width: Frame width in pixels (default 960).
        height: Frame height in pixels (default 544).
        frames_per_segment: Frames generated per keyframe pair (default 33).
        seed: Random seed; -1 for a random seed.
        fps: Output video frames per second (default 16).

    Returns:
        dict with 'path' (absolute path to saved video) and 'filename'.
    """
    resolved_seed = None if seed < 0 else seed
    paths = [frame1_path, frame2_path, frame3_path, frame4_path, frame5_path, frame6_path]
    image_bytes_list = [Path(p).read_bytes() for p in paths]
    image_filenames = [Path(p).name for p in paths]
    video_bytes, filename = comfy_client.generate_flf2v(
        image_bytes_list=image_bytes_list, image_filenames=image_filenames,
        prompt=prompt, width=width, height=height,
        frames_per_segment=frames_per_segment, seed=resolved_seed, fps=fps,
    )
    out_path = OUTPUT_DIR / filename
    out_path.write_bytes(video_bytes)
    return {"path": str(out_path.resolve()), "filename": filename}


CORS = [Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])]

@mcp.tool()
def generate_wan_i2v(
    image_path: str,
    prompt: str,
    width: int = 512,
    height: int = 512,
    length: int = 81,
    seed: int = -1,
    fps: int = 16,
) -> dict:
    """Generate a video from a start image and text prompt using Wan 2.2 14B I2V via ComfyUI.

    Args:
        image_path: Absolute path to the start frame image file.
        prompt: Text description of the motion to generate.
        width: Frame width in pixels (default 512).
        height: Frame height in pixels (default 512).
        length: Number of frames (default 81, ~5s at 16fps).
        seed: Random seed; -1 for a random seed.
        fps: Output video frames per second (default 16).

    Returns:
        dict with 'path' (absolute path to saved video) and 'filename'.
    """
    resolved_seed = None if seed < 0 else seed
    image_bytes = Path(image_path).read_bytes()
    image_filename = Path(image_path).name
    video_bytes, filename = comfy_client.generate_i2v(
        image_bytes=image_bytes, image_filename=image_filename,
        prompt=prompt, width=width, height=height,
        length=length, seed=resolved_seed, fps=fps,
    )
    out_path = OUTPUT_DIR / filename
    out_path.write_bytes(video_bytes)
    return {"path": str(out_path.resolve()), "filename": filename}


@mcp.tool()
def generate_wan_s2v(
    image_path: str,
    audio_path: str,
    prompt: str = "",
    seed: int = -1,
    fps: int = 16,
    chunk_length: int = 43,
) -> dict:
    """Generate a talking-head video from a reference image and audio using Wan 2.2 14B S2V via ComfyUI.

    Args:
        image_path: Absolute path to the reference face image.
        audio_path: Absolute path to the audio file (WAV recommended).
        prompt: Optional text prompt (default empty).
        seed: Random seed; -1 for a random seed.
        fps: Output video frames per second (default 16).
        chunk_length: Frames per generation chunk (default 43). Lower = less VRAM.

    Returns:
        dict with 'path' (absolute path to saved video) and 'filename'.
    """
    resolved_seed = None if seed < 0 else seed
    image_bytes = Path(image_path).read_bytes()
    audio_bytes = Path(audio_path).read_bytes()
    video_bytes, filename = comfy_client.generate_s2v(
        image_bytes=image_bytes, image_filename=Path(image_path).name,
        audio_bytes=audio_bytes, audio_filename=Path(audio_path).name,
        prompt=prompt, seed=resolved_seed, fps=fps, chunk_length=chunk_length,
    )
    out_path = OUTPUT_DIR / filename
    out_path.write_bytes(video_bytes)
    return {"path": str(out_path.resolve()), "filename": filename}


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=int(os.environ.get("MCP_PORT", 8000)), middleware=CORS, stateless_http=True)
