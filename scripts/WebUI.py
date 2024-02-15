from stable_cascade import web_demo
import gradio as gr


gradio_app = gr.Blocks()
with gradio_app:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    Stable Cascade🌟
    </h1>
    """)
    with gr.Row():
        with gr.Column():
            web_demo()

gradio_app.queue()
gradio_app.launch(share=False, inbrowser=True)