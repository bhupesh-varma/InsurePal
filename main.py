import gradio as gr
import requests

BACKEND_URL = "http://localhost:8000"

def upload_and_query(file, question):
    if file is None:
        return "Please upload a document."
    filename = file.split("\\")[-1]  # Adjust for Windows paths; use split("/") on Unix if needed
    with open(file, "rb") as f:
        files = {"file": (filename, f)}
        upload_response = requests.post(f"{BACKEND_URL}/upload/", files=files)
        if upload_response.status_code != 200:
            return f"Upload failed: {upload_response.text}"

    data = {"question": question}
    query_response = requests.post(f"{BACKEND_URL}/query/", data=data)
    if query_response.status_code == 200:
        return query_response.json().get("answer", "No answer found.")
    else:
        return f"Query failed: {query_response.text}"

with gr.Blocks() as demo:
    gr.Markdown("# InsurePal Document Query System")

    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload Document", type="filepath")
            question_input = gr.Textbox(label="Ask a Question", placeholder="Type your question here...")
            submit_button = gr.Button("Submit")

        with gr.Column():
            answer_output = gr.Textbox(label="Answer", interactive=False)

    submit_button.click(upload_and_query, inputs=[file_input, question_input], outputs=answer_output)

demo.launch()
