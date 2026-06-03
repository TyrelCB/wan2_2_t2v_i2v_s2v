# Wan 2.2 14B Video Generation

Gradio UI and MCP server for four [Wan 2.2 14B](https://github.com/Wan-Video/Wan2.1) video generation workflows via ComfyUI.

## Requirements

- ComfyUI running with the Wan 2.2 14B models loaded
- Python 3.10+

```bash
pip install -r requirements.txt
```

## Configuration

Set `COMFY_URL` to point at your ComfyUI instance (default: `http://192.168.6.181:8188`):

```bash
export COMFY_URL=http://your-comfyui-host:8188
```

## Models

| File | Directory |
|------|-----------|
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | `models/clip/` |
| `wan_2.1_vae.safetensors` | `models/vae/` |
| `wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors` | `models/unet/` |
| `wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors` | `models/unet/` |
| `wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors` | `models/unet/` |
| `wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors` | `models/unet/` |
| `wan2.2_s2v_14B_fp8_scaled.safetensors` | `models/unet/` |
| `wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors` | `models/loras/` |
| `wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors` | `models/loras/` |
| `wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors` | `models/loras/` |
| `wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors` | `models/loras/` |
| `wav2vec2_large_english_fp16.safetensors` | `models/audio_encoder/` |

## Systemd Service

`wan2-ui-mcp.service` runs both the Gradio UI and MCP server together via `start.sh`. Edit `User`, `WorkingDirectory`, `GRADIO_SERVER_PORT`, and `MCP_PORT` to match your setup, then:

```bash
sudo cp wan2-ui-mcp.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable wan2-ui-mcp
sudo systemctl start wan2-ui-mcp
```

Check status and logs:

```bash
sudo systemctl status wan2-ui-mcp
sudo journalctl -u wan2-ui-mcp -f
```

## Gradio UI

```bash
python app.py
```

Open `http://localhost:7862` (or the port set by `GRADIO_SERVER_PORT`). Four tabs:

### Text to Video
Generate video from a text prompt. Dual-pass LightX2V sampling (4 steps). Default resolution 848×480.

### Image to Video
Animate a start frame image guided by a text prompt. Same dual-pass architecture as T2V. Default resolution 512×512.

### Talking Head
Animate a reference face image driven by an audio file. Chunked `WanSoundImageToVideo` generation — audio length determines output duration. Default chunk size 43 frames.

### First/Last Frame
Upload 6 keyframes and interpolate between each consecutive pair using `WanFirstLastFrameToVideo`. Generates 5 segments of 33 frames each (≈10s at 16fps) and concatenates them. Default resolution 960×544.

## MCP Server

```bash
python mcp_server.py
```

FastMCP streamable-HTTP server on `http://0.0.0.0:8001/mcp` with CORS enabled. Override the port with `MCP_PORT` (default: 8000).

### Tools

#### `generate_wan_video`
| Parameter | Default | Description |
|-----------|---------|-------------|
| `prompt` | — | Text description |
| `width` | 848 | Frame width |
| `height` | 480 | Frame height |
| `length` | 81 | Number of frames |
| `seed` | -1 | Random seed (-1 = random) |
| `fps` | 16 | Output FPS |

#### `generate_wan_i2v`
| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_path` | — | Absolute path to start frame |
| `prompt` | — | Text description of motion |
| `width` | 512 | Frame width |
| `height` | 512 | Frame height |
| `length` | 81 | Number of frames |
| `seed` | -1 | Random seed |
| `fps` | 16 | Output FPS |

#### `generate_wan_s2v`
| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_path` | — | Absolute path to reference face image |
| `audio_path` | — | Absolute path to audio file (WAV) |
| `prompt` | `""` | Optional text prompt |
| `seed` | -1 | Random seed |
| `fps` | 16 | Output FPS |
| `chunk_length` | 43 | Frames per generation chunk |

#### `generate_wan_flf2v`
| Parameter | Default | Description |
|-----------|---------|-------------|
| `frame1_path`–`frame6_path` | — | Absolute paths to 6 keyframes in order |
| `prompt` | `""` | Optional text prompt |
| `width` | 960 | Frame width |
| `height` | 544 | Frame height |
| `frames_per_segment` | 33 | Frames generated between each keyframe pair |
| `seed` | -1 | Random seed |
| `fps` | 16 | Output FPS |

All tools return `{"path": "/absolute/path/to/video.mp4", "filename": "..."}`. Videos are saved to `./outputs/`.
