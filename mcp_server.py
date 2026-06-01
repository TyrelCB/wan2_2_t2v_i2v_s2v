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


CORS = [Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])]

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000, middleware=CORS, stateless_http=True)
