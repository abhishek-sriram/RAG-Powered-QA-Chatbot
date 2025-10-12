# RAG Powered QA Chatbot
![Project Banner](https://github.com/abhishek-sriram/RAG-based-QA-Chatbot/blob/main/RAG%20based%20QA%20Chatbot/bot_img.png)

## Project Overview
The project involved building a bot that leveraged LangChain and a large language model (LLM) to answer questions based on content from loaded PDF documents.

### Problem Statement:
The goal of this project was to develop a question-answering bot that could answer user queries based on information extracted from PDF documents. The bot used a combination of retrieval-augmented generation (RAG) and a large language model to retrieve relevant information from the documents and generate accurate answers. The following tasks were completed in order to build the QA bot web app:

1. Load documents from PDFs.
2. Split large documents into smaller, manageable chunks.
3. Generate text embeddings for the document chunks.
4. Store the embeddings in a vector database for efficient retrieval.
5. Implement a retriever function for querying the vector database.
6. Set up a Gradio interface for users to interact with the QA bot.

---

### Key Steps and Completed Tasks:

#### 1. Loaded Document Using LangChain for Different Sources
- **Task:** The `document_loader` function was implemented using PyPDFLoader from the langchain_community library to load PDF files.

```python
from langchain_community.document_loaders import PyPDFLoader

def document_loader(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents
```
#### 2. Split Long Documents Using Text Splitters
- **Task:** The 'text_splitter' function was completed using RecursiveCharacterTextSplitter to split the loaded PDF content into manageable text chunks.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def text_splitter(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(documents)
    return split_docs
```

#### 3. Generated Embeddings Using Embedding Models
- **Task:** The 'watsonx_embedding()' function was completed using the WatsonxEmbeddings class from the langchain_ibm library to generate text embeddings.

```python
from langchain_ibm import WatsonxEmbeddings

def watsonx_embedding(text_chunks):
    embeddings = WatsonxEmbeddings()
    embedding_vectors = embeddings.embed_documents(text_chunks)
    return embedding_vectors
```

#### 4. Stored Embeddings Using Vector Databases
- **Task:** The 'vector_database()' function was completed to embed the text chunks using the watsonx_embedding() model and store them in a Chroma vector store using Chroma.from_documents().

```python
from langchain.vectorstores import Chroma

def vector_database(text_chunks, embeddings):
    vector_store = Chroma.from_documents(text_chunks, embeddings)
    return vector_store
```

#### 5. Defined Retrievers
- **Task:** The 'retriever(file)' function was completed to load, split, embed, and convert documents into a retriever using similarity search from a Chroma vector store.

```python
def retriever(file):
    documents = document_loader(file)
    split_docs = text_splitter(documents)
    embeddings = watsonx_embedding(split_docs)
    vector_store = vector_database(split_docs, embeddings)
    retriever = vector_store.as_retriever()
    return retriever
  ```

#### 6. Set Up Gradio as the Front-End Interface
- **Task:** The retriever_qa(file, query) function was defined using the RetrievalQA chain from LangChain to perform question-answering over documents using RAG (Retrieval-Augmented Generation). This logic was connected to a Gradio interface for interactive use.

```python
import gradio as gr
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

def get_llm():
    return OpenAI()

def retriever_qa(file, query):
    retriever_instance = retriever(file)
    qa_chain = RetrievalQA.from_chain_type(llm=get_llm(), chain_type="map_reduce", retriever=retriever_instance)
    result = qa_chain.run(query)
    return result

interface = gr.Interface(fn=retriever_qa, inputs=[gr.File(), gr.Textbox()], outputs="text")
interface.launch()
  ```

### Gradio Interface 

- The Gradio interface was created using gr.Interface.
- A PDF was uploaded to the interface.
- User query entered: query = "What this paper is talking about?"
![Gradio Interface](https://github.com/abhishek-sriram/RAG-based-QA-Chatbot/blob/main/RAG%20based%20QA%20Chatbot/QA_bot.png)









