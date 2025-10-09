import gradio as gr
from transformers import pipeline
import fitz  # PyMuPDF to extract text from the PDF

# Load the question-answering pipeline from Hugging Face
qa_pipeline = pipeline("question-answering", model="t5-base")

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to answer the query based on the extracted text
def answer_query(pdf_file, user_query):
    text = extract_text_from_pdf(pdf_file.name)  # Use .name if type is filepath
    answer = qa_pipeline(question=user_query, context=text)
    return answer['answer']

# Create the Gradio interface
iface = gr.Interface(fn=answer_query,
                     inputs=[gr.File(type="filepath", label="Upload PDF"),
                             gr.Textbox(label="Enter your question", placeholder="What this paper is talking about?")],
                     outputs="text",
                     title="AI RAG QA Bot",
                     description="Upload a PDF.")

# Launch the app
iface.launch('share=True')
