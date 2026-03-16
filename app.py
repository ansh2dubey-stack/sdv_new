import gradio as gr

def predict(input_text):
    # Replace with your model/function
    return "Output: " + input_text

iface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Input"),
    outputs=gr.Textbox(label="Output"),
    title="SDV App",
    description="My deployed project"
)

iface.launch()