import os
import random
import time
from pathlib import Path

import gradio as gr

import comfy_client

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def _progress_fn(progress, resolved_seed):
    def on_progress(elapsed):
        progress(min(elapsed / 600.0, 0.95), desc=f"Generating… {elapsed:.0f}s  |  seed: {resolved_seed}")
    return on_progress


def _save_and_return(video_bytes, filename, t_start, resolved_seed):
    gen_time = time.time() - t_start
    out_path = OUTPUT_DIR / filename
    out_path.write_bytes(video_bytes)
    status = f"Done in {gen_time:.0f}s  |  seed: {resolved_seed}  |  {filename} ({len(video_bytes)//1024}KB)"
    return str(out_path), status


# ── T2V ───────────────────────────────────────────────────────────────────────

def generate_t2v(prompt, width, height, length, seed, fps, progress=gr.Progress()):
    if not prompt.strip():
        yield gr.update(), gr.update(value="Error: prompt is required")
        return
    resolved_seed = random.randint(0, 2**32 - 1) if seed < 0 else int(seed)
    yield gr.update(value=None), gr.update(value=f"Submitting… seed: {resolved_seed}")
    try:
        wf = comfy_client.build_workflow(prompt=prompt, width=int(width), height=int(height),
                                         length=int(length), seed=resolved_seed, fps=int(fps))
        prompt_id = comfy_client.submit(wf)
        t_start = time.time()
        yield gr.update(value=None), gr.update(value=f"Queued — seed: {resolved_seed}  |  id: {prompt_id}")
        data = comfy_client.poll(prompt_id, on_progress=_progress_fn(progress, resolved_seed))
        yield gr.update(value=None), gr.update(value="Downloading…")
        video_info = data["outputs"]["80"]["images"][0]
        video_bytes = comfy_client.download_video(video_info)
        path, status = _save_and_return(video_bytes, video_info["filename"], t_start, resolved_seed)
        yield gr.update(value=path), gr.update(value=status)
    except Exception as e:
        yield gr.update(value=None), gr.update(value=f"Error: {e}")


# ── I2V ───────────────────────────────────────────────────────────────────────

def generate_i2v(image_path, prompt, width, height, length, seed, fps, progress=gr.Progress()):
    if image_path is None:
        yield gr.update(), gr.update(value="Error: start frame image is required")
        return
    if not prompt.strip():
        yield gr.update(), gr.update(value="Error: prompt is required")
        return
    resolved_seed = random.randint(0, 2**32 - 1) if seed < 0 else int(seed)
    yield gr.update(value=None), gr.update(value=f"Uploading image… seed: {resolved_seed}")
    try:
        image_bytes = Path(image_path).read_bytes()
        image_filename = Path(image_path).name
        wf = comfy_client.build_i2v_workflow(
            image_filename=comfy_client.upload_image(image_bytes, image_filename),
            prompt=prompt, width=int(width), height=int(height),
            length=int(length), seed=resolved_seed, fps=int(fps),
        )
        prompt_id = comfy_client.submit(wf)
        t_start = time.time()
        yield gr.update(value=None), gr.update(value=f"Queued — seed: {resolved_seed}  |  id: {prompt_id}")
        data = comfy_client.poll(prompt_id, on_progress=_progress_fn(progress, resolved_seed))
        yield gr.update(value=None), gr.update(value="Downloading…")
        video_info = data["outputs"]["108"]["images"][0]
        video_bytes = comfy_client.download_video(video_info)
        path, status = _save_and_return(video_bytes, video_info["filename"], t_start, resolved_seed)
        yield gr.update(value=path), gr.update(value=status)
    except Exception as e:
        yield gr.update(value=None), gr.update(value=f"Error: {e}")


# ── S2V (Talking Head) ────────────────────────────────────────────────────────

def generate_s2v(image_path, audio_path, prompt, seed, fps, chunk_length, progress=gr.Progress()):
    if image_path is None:
        yield gr.update(), gr.update(value="Error: reference image is required")
        return
    if audio_path is None:
        yield gr.update(), gr.update(value="Error: audio file is required")
        return
    resolved_seed = random.randint(0, 2**32 - 1) if seed < 0 else int(seed)
    yield gr.update(value=None), gr.update(value=f"Uploading files… seed: {resolved_seed}")
    try:
        image_bytes = Path(image_path).read_bytes()
        audio_bytes = Path(audio_path).read_bytes()
        stored_image = comfy_client.upload_image(image_bytes, Path(image_path).name)
        stored_audio = comfy_client.upload_audio(audio_bytes, Path(audio_path).name)
        wf = comfy_client.build_s2v_workflow(
            image_filename=stored_image, audio_filename=stored_audio,
            prompt=prompt, seed=resolved_seed, fps=int(fps), chunk_length=int(chunk_length),
        )
        prompt_id = comfy_client.submit(wf)
        t_start = time.time()
        yield gr.update(value=None), gr.update(value=f"Queued — seed: {resolved_seed}  |  id: {prompt_id}")
        data = comfy_client.poll(prompt_id, timeout=900.0, on_progress=_progress_fn(progress, resolved_seed))
        yield gr.update(value=None), gr.update(value="Downloading…")
        video_info = data["outputs"]["113"]["images"][0]
        video_bytes = comfy_client.download_video(video_info)
        path, status = _save_and_return(video_bytes, video_info["filename"], t_start, resolved_seed)
        yield gr.update(value=path), gr.update(value=status)
    except Exception as e:
        yield gr.update(value=None), gr.update(value=f"Error: {e}")


# ── FLF2V (6-keyframe) ────────────────────────────────────────────────────────

def generate_flf2v(kf1, kf2, kf3, kf4, kf5, kf6, prompt, width, height, frames, seed, fps, progress=gr.Progress()):
    paths = [kf1, kf2, kf3, kf4, kf5, kf6]
    if any(p is None for p in paths):
        yield gr.update(), gr.update(value="Error: all 6 keyframe images are required")
        return
    resolved_seed = random.randint(0, 2**32 - 1) if seed < 0 else int(seed)
    yield gr.update(value=None), gr.update(value=f"Uploading 6 keyframes… seed: {resolved_seed}")
    try:
        image_bytes_list = [Path(p).read_bytes() for p in paths]
        image_filenames = [Path(p).name for p in paths]
        stored = [comfy_client.upload_image(b, f) for b, f in zip(image_bytes_list, image_filenames)]
        wf = comfy_client.build_flf2v_workflow(
            stored, prompt=prompt, width=int(width), height=int(height),
            frames_per_segment=int(frames), seed=resolved_seed, fps=int(fps),
        )
        prompt_id = comfy_client.submit(wf)
        t_start = time.time()
        yield gr.update(value=None), gr.update(value=f"Queued — seed: {resolved_seed}  |  id: {prompt_id}")
        data = comfy_client.poll(prompt_id, timeout=900.0, on_progress=_progress_fn(progress, resolved_seed))
        yield gr.update(value=None), gr.update(value="Downloading…")
        video_info = data["outputs"]["save"]["images"][0]
        video_bytes = comfy_client.download_video(video_info)
        path, status = _save_and_return(video_bytes, video_info["filename"], t_start, resolved_seed)
        yield gr.update(value=path), gr.update(value=status)
    except Exception as e:
        yield gr.update(value=None), gr.update(value=f"Error: {e}")


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Wan 2.2") as demo:
    gr.Markdown("# Wan 2.2 14B Video Generation")

    with gr.Tabs():

        with gr.Tab("Text to Video"):
            with gr.Row():
                with gr.Column(scale=2):
                    t2v_prompt = gr.Textbox(label="Prompt", placeholder="Describe the video…", lines=4)
                    with gr.Row():
                        t2v_width = gr.Number(label="Width", value=848, precision=0)
                        t2v_height = gr.Number(label="Height", value=480, precision=0)
                    with gr.Row():
                        t2v_length = gr.Slider(label="Frames", minimum=17, maximum=161, step=8, value=81)
                        t2v_fps = gr.Number(label="FPS", value=16, precision=0)
                    t2v_seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)
                    t2v_btn = gr.Button("Generate", variant="primary")
                with gr.Column(scale=3):
                    t2v_video = gr.Video(label="Output Video")
                    t2v_status = gr.Textbox(label="Status", interactive=False, lines=2)
            t2v_btn.click(fn=generate_t2v,
                          inputs=[t2v_prompt, t2v_width, t2v_height, t2v_length, t2v_seed, t2v_fps],
                          outputs=[t2v_video, t2v_status])

        with gr.Tab("Image to Video"):
            with gr.Row():
                with gr.Column(scale=2):
                    i2v_image = gr.Image(label="Start Frame", type="filepath")
                    i2v_prompt = gr.Textbox(label="Prompt", placeholder="Describe the motion…", lines=3)
                    with gr.Row():
                        i2v_width = gr.Number(label="Width", value=512, precision=0)
                        i2v_height = gr.Number(label="Height", value=512, precision=0)
                    with gr.Row():
                        i2v_length = gr.Slider(label="Frames", minimum=17, maximum=161, step=8, value=81)
                        i2v_fps = gr.Number(label="FPS", value=16, precision=0)
                    i2v_seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)
                    i2v_btn = gr.Button("Generate", variant="primary")
                with gr.Column(scale=3):
                    i2v_video = gr.Video(label="Output Video")
                    i2v_status = gr.Textbox(label="Status", interactive=False, lines=2)
            i2v_btn.click(fn=generate_i2v,
                          inputs=[i2v_image, i2v_prompt, i2v_width, i2v_height, i2v_length, i2v_seed, i2v_fps],
                          outputs=[i2v_video, i2v_status])

        with gr.Tab("Talking Head"):
            with gr.Row():
                with gr.Column(scale=2):
                    s2v_image = gr.Image(label="Reference Image", type="filepath")
                    s2v_audio = gr.Audio(label="Audio", type="filepath")
                    s2v_prompt = gr.Textbox(label="Prompt (optional)", lines=2)
                    with gr.Row():
                        s2v_seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)
                        s2v_fps = gr.Number(label="FPS", value=16, precision=0)
                    s2v_chunk = gr.Number(label="Chunk Length (frames/chunk)", value=43, precision=0)
                    s2v_btn = gr.Button("Generate", variant="primary")
                with gr.Column(scale=3):
                    s2v_video = gr.Video(label="Output Video")
                    s2v_status = gr.Textbox(label="Status", interactive=False, lines=2)
            s2v_btn.click(fn=generate_s2v,
                          inputs=[s2v_image, s2v_audio, s2v_prompt, s2v_seed, s2v_fps, s2v_chunk],
                          outputs=[s2v_video, s2v_status])

        with gr.Tab("First/Last Frame"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("Upload 6 keyframes in order. Each consecutive pair generates one segment.")
                    with gr.Row():
                        flf_kf1 = gr.Image(label="Frame 1 (start)", type="filepath")
                        flf_kf2 = gr.Image(label="Frame 2", type="filepath")
                        flf_kf3 = gr.Image(label="Frame 3", type="filepath")
                    with gr.Row():
                        flf_kf4 = gr.Image(label="Frame 4", type="filepath")
                        flf_kf5 = gr.Image(label="Frame 5", type="filepath")
                        flf_kf6 = gr.Image(label="Frame 6 (end)", type="filepath")
                    flf_prompt = gr.Textbox(label="Prompt (optional)", lines=2)
                    with gr.Row():
                        flf_width = gr.Number(label="Width", value=960, precision=0)
                        flf_height = gr.Number(label="Height", value=544, precision=0)
                    with gr.Row():
                        flf_frames = gr.Number(label="Frames/segment", value=33, precision=0)
                        flf_fps = gr.Number(label="FPS", value=16, precision=0)
                    flf_seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)
                    flf_btn = gr.Button("Generate", variant="primary")
                with gr.Column(scale=3):
                    flf_video = gr.Video(label="Output Video")
                    flf_status = gr.Textbox(label="Status", interactive=False, lines=2)
            flf_btn.click(fn=generate_flf2v,
                          inputs=[flf_kf1, flf_kf2, flf_kf3, flf_kf4, flf_kf5, flf_kf6,
                                  flf_prompt, flf_width, flf_height, flf_frames, flf_seed, flf_fps],
                          outputs=[flf_video, flf_status])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)))
