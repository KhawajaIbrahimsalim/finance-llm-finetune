import gradio as gr

def chat_fn(prompt):
    return f"MODEL RESPONSE: {prompt}"

demo = gr.Interface(
    fn=chat_fn,
    inputs="text",
    outputs="text",
    title="Local Enterprise LLM System"
)

demo.launch()