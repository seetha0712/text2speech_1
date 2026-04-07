"""
Gradio Web UI for Indian TTS.

Usage:
    python app.py --checkpoint outputs/checkpoints/checkpoint_final.pt
"""

import argparse

import gradio as gr
import numpy as np

from indian_tts.inference import IndianTTS


def create_app(tts: IndianTTS) -> gr.Blocks:
    """Create Gradio interface."""

    def synthesize(text, voice, speed, expressiveness):
        if not text.strip():
            return (tts.sampling_rate, np.zeros(1000, dtype=np.float32))

        audio = tts.synthesize(
            text=text,
            voice=voice.lower(),
            speed=speed,
            expressiveness=expressiveness,
        )
        return (tts.sampling_rate, audio)

    with gr.Blocks(title="Indian TTS", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Indian Text-to-Speech")
        gr.Markdown("Generate natural Indian-sounding speech in male and female voices.")

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Text",
                    placeholder="Enter text to synthesize...",
                    lines=3,
                    value="Hello, welcome to our Indian text to speech system. How are you doing today?",
                )
                voice_select = gr.Radio(
                    choices=["Male", "Female"],
                    value="Female",
                    label="Voice",
                )

            with gr.Column(scale=1):
                speed_slider = gr.Slider(
                    minimum=0.5, maximum=2.0, value=1.0, step=0.1,
                    label="Speed",
                )
                expr_slider = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.667, step=0.05,
                    label="Expressiveness",
                )

        generate_btn = gr.Button("Generate Speech", variant="primary")
        audio_output = gr.Audio(label="Generated Speech", type="numpy")

        generate_btn.click(
            fn=synthesize,
            inputs=[text_input, voice_select, speed_slider, expr_slider],
            outputs=audio_output,
        )

        gr.Markdown("---")
        gr.Markdown("### Example Texts")
        gr.Examples(
            examples=[
                ["Good morning! The weather in Bangalore is very pleasant today.", "Female", 1.0, 0.667],
                ["Namaste, welcome to our conference. Please take your seats.", "Male", 1.0, 0.667],
                ["The train from Delhi to Mumbai will depart at platform number three.", "Female", 0.9, 0.5],
                ["India is celebrating its seventy-fifth year of independence.", "Male", 1.0, 0.8],
            ],
            inputs=[text_input, voice_select, speed_slider, expr_slider],
        )

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    tts = IndianTTS(args.checkpoint, device=args.device)
    app = create_app(tts)
    app.launch(server_port=args.port, share=args.share)
