import random
import time
from pathlib import Path

import gradio as gr

import comfy_client

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def generate(prompt, width, height, length, seed, fps, progress=gr.Progress()):
    if not prompt.strip():
        yield gr.update(), gr.update(value="Error: prompt is required")
        return

    resolved_seed = random.randint(0, 2**32 - 1) if seed < 0 else int(seed)
    yield gr.update(value=None), gr.update(value=f"Submitting… seed: {resolved_seed}")

    submitted_id = None

    def on_progress(elapsed):
        nonlocal submitted_id
        msg = f"Generating… {elapsed:.0f}s elapsed  |  seed: {resolved_seed}"
        progress(min(elapsed / 600.0, 0.95), desc=msg)

    try:
        wf = comfy_client.build_workflow(
            prompt=prompt,
            width=int(width),
            height=int(height),
            length=int(length),
            seed=resolved_seed,
            fps=int(fps),
        )
        submitted_id = comfy_client.submit(wf)
        t_start = time.time()
        yield gr.update(value=None), gr.update(value=f"Queued — seed: {resolved_seed}  |  id: {submitted_id}")

        data = comfy_client.poll(submitted_id, on_progress=on_progress)
        gen_time = time.time() - t_start
        yield gr.update(value=None), gr.update(value="Downloading video…")

        video_info = data["outputs"]["80"]["images"][0]
        video_bytes = comfy_client.download_video(video_info)

        out_path = OUTPUT_DIR / video_info["filename"]
        out_path.write_bytes(video_bytes)

        yield gr.update(value=str(out_path)), gr.update(
            value=f"Done in {gen_time:.0f}s  |  seed: {resolved_seed}  |  {out_path.name} ({len(video_bytes)//1024}KB)"
        )

    except TimeoutError as e:
        yield gr.update(value=None), gr.update(value=f"Timeout: {e}")
    except Exception as e:
        yield gr.update(value=None), gr.update(value=f"Error: {e}")


with gr.Blocks(title="Wan 2.2 T2V") as demo:
    gr.Markdown("# Wan 2.2 14B Text-to-Video")

    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Describe the video you want to generate...",
                lines=4,
            )
            with gr.Row():
                width = gr.Number(label="Width", value=848, precision=0)
                height = gr.Number(label="Height", value=480, precision=0)
            with gr.Row():
                length = gr.Slider(
                    label="Frames",
                    minimum=17,
                    maximum=161,
                    step=8,
                    value=81,
                )
                fps = gr.Number(label="FPS", value=16, precision=0)
            seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)
            submit_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=3):
            video_out = gr.Video(label="Output Video")
            status = gr.Textbox(label="Status", interactive=False, lines=2)

    submit_btn.click(
        fn=generate,
        inputs=[prompt, width, height, length, seed, fps],
        outputs=[video_out, status],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
