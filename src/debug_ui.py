import gradio as gr
from model_loader import TTSModelManager
import numpy as np

model = TTSModelManager()

with gr.Blocks(title="MemoryServerTTS Debug UI") as demo:
    gr.Markdown("# MemoryServerTTS 调试界面")
    gr.Markdown("输入文本，点击生成即可测试语音合成效果。")

    with gr.Row():
        text_input = gr.Textbox(label="输入文本", lines=3, value="Hello, welcome to Memory English Learning App!")
        voice_input = gr.Dropdown(
            label="音色",
            choices=["aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian"],
            value="ono_anna"
        )
        language_input = gr.Dropdown(
            label="语言",
            choices=["auto", "chinese", "english", "french", "german", "italian", "japanese", "korean", "portuguese", "russian", "spanish"],
            value="english"
        )
        instructions_input = gr.Textbox(label="语音指令", value="Speak with a happy and encouraging tone.")

    audio_output = gr.Audio(label="合成音频")
    error_output = gr.Textbox(label="错误信息", interactive=False)
    generate_btn = gr.Button("生成语音")

    def tts_generate(text, voice, language, instructions):
        try:
            wavs, sr = model.generate(text, voice, language, instructions)
            return (sr, wavs[0]), ""
        except Exception as e:
            return None, str(e)

    generate_btn.click(
        fn=tts_generate,
        inputs=[text_input, voice_input, language_input, instructions_input],
        outputs=[audio_output, error_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)