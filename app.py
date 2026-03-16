import gradio as gr
import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

data = None
model = None
metadata = None


def upload_data(file):
    global data, metadata

    data = pd.read_csv(file.name)

    # create ID column if it does not exist
    if "ID" not in data.columns:
        data.insert(0, "ID", range(1, len(data) + 1))

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)

    return data

def train_model():
    global model, data, metadata

    if data is None:
        return "Please upload a dataset first."

    model = CTGANSynthesizer(metadata)
    model.fit(data)

    return "Model trained successfully."


def generate_data(rows):
    global model

    if model is None:
        return "Train the model first."

    synthetic = model.sample(int(rows))
    return synthetic


def delete_record(record_id):
    global data, model, metadata

    record_id = int(record_id)

    # remove record
    data = data[data["ID"] != record_id]

    # retrain model (machine unlearning)
    model = CTGANSynthesizer(metadata)
    model.fit(data)

    return data


with gr.Blocks() as demo:

    gr.Markdown("# Synthetic Data Vault with Machine Unlearning")

    with gr.Tab("Upload Dataset"):
        file = gr.File(label="Upload CSV Dataset")
        preview = gr.Dataframe(label="Dataset Preview")
        file.upload(upload_data, file, preview)

    with gr.Tab("Train Model"):
        train = gr.Button("Train Model")
        status = gr.Textbox(label="Training Status")
        train.click(train_model, outputs=status)

    with gr.Tab("Generate Synthetic Data"):
        rows = gr.Slider(10, 5000, label="Number of Synthetic Rows")
        generate = gr.Button("Generate Synthetic Data")
        output = gr.Dataframe(label="Synthetic Dataset")
        generate.click(generate_data, rows, output)

    with gr.Tab("Machine Unlearning"):
        delete_id = gr.Textbox(label="Record ID to Delete")
        delete_btn = gr.Button("Delete Record")

        updated_table = gr.Dataframe(label="Dataset After Deletion")

        delete_btn.click(delete_record, delete_id, updated_table)

demo.launch()
