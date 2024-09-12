import sys
import os
import tempfile
from pathlib import Path
import re
import gradio as gr
from huggingface_hub import login
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
import chromadb
from unidecode import unidecode
import pikepdf

# Your Hugging Face token
hf_token = "Your Token"
login(token=hf_token)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

def decrypt_pdf(file_path, password):
    try:
        with pikepdf.open(file_path, password=password) as pdf:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            pdf.save(temp_file.name)
        return temp_file.name
    except Exception as e:
        raise ValueError(f"Failed to decrypt PDF: {str(e)}")

def load_doc(file_path, password=None, chunk_size=600, chunk_overlap=40):
    if password:
        file_path = decrypt_pdf(file_path, password)

    loader = PyPDFLoader(file_path)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap)
    doc_splits = text_splitter.split_documents(pages)

    if password:
        os.unlink(file_path)

    return doc_splits

def create_db(splits, collection_name):
    embedding = HuggingFaceEmbeddings()
    new_client = chromadb.EphemeralClient()
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        client=new_client,
        collection_name=collection_name,
    )
    return vectordb

def initialize_llm(temperature=0.7, max_tokens=1024, top_k=3):
    return HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        temperature=temperature,
        max_new_tokens=max_tokens,
        top_k=top_k,
    )

def initialize_qa_chain(vector_db, llm):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key='answer',
        return_messages=True
    )
    retriever = vector_db.as_retriever()
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        chain_type="stuff",
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )
    return qa_chain

def create_collection_name(filepath):
    collection_name = Path(filepath).stem
    collection_name = collection_name.replace(" ","-")
    collection_name = unidecode(collection_name)
    collection_name = re.sub('[^A-Za-z0-9]+', '-', collection_name)
    collection_name = collection_name[:50]
    if len(collection_name) < 3:
        collection_name = collection_name + 'xyz'
    if not collection_name[0].isalnum():
        collection_name = 'A' + collection_name[1:]
    if not collection_name[-1].isalnum():
        collection_name = collection_name[:-1] + 'Z'
    return collection_name

def initialize_database(file_path, password=None):
    collection_name = create_collection_name(file_path)
    doc_splits = load_doc(file_path, password)
    vector_db = create_db(doc_splits, collection_name)
    return vector_db, collection_name

llm = initialize_llm()
pdf_qa_chain = None

def analyze_and_chat(file, password):
    global pdf_qa_chain
    if file is None:
        return "Please upload a bank statement PDF first."

    try:
        file_path = file.name
        vector_db, collection_name = initialize_database(file_path, password)
        pdf_qa_chain = initialize_qa_chain(vector_db, llm)

        analysis = []

        # Analyze current expenses
        current_analysis = pdf_qa_chain({"question": "Summarize the current expenses based on the bank statement."})
        analysis.append(f"Current Expense Analysis:\n{current_analysis['answer']}\n")

        # Predict next month's expenses
        next_month = pdf_qa_chain({"question": "Based on the current expenses, predict the expenses for next month."})
        analysis.append(f"Next Month Expense Prediction:\n{next_month['answer']}\n")

        # Predict next year's expenses
        next_year = pdf_qa_chain({"question": "Based on the current expenses, predict the expenses for next year."})
        analysis.append(f"Next Year Expense Prediction:\n{next_year['answer']}\n")

        # Provide money management strategy
        strategy = pdf_qa_chain({"question": "Provide a strategy to manage money based on the current expenses and predictions."})
        analysis.append(f"Money Management Strategy:\n{strategy['answer']}\n")

        # Provide investment strategy
        investment = pdf_qa_chain({"question": "Suggest an investment strategy based on the expense analysis and predictions."})
        analysis.append(f"Investment Strategy:\n{investment['answer']}\n")

        return "\n".join(analysis), gr.update(visible=True), gr.update(visible=True)
    except ValueError as e:
        return str(e), gr.update(visible=False), gr.update(visible=False)
    except Exception as e:
        return f"An error occurred: {str(e)}", gr.update(visible=False), gr.update(visible=False)

def expense_chat(message, history):
    global pdf_qa_chain
    if pdf_qa_chain:
        response = pdf_qa_chain({"question": message})
        return [(message, response["answer"])]
    else:
        return [(message, "Please upload and analyze a PDF first.")]

with gr.Blocks() as demo:
    gr.Markdown("# AI Assistant with Expense Analysis")

    with gr.Tab("Expense Analysis and Chat"):
        expense_file_input = gr.File(label="Upload Bank Statement PDF")
        expense_password_input = gr.Textbox(label="PDF Password (if applicable)", type="password")
        analyze_button = gr.Button("Analyze Expenses and Start Chat")
        analysis_output = gr.Textbox(label="Analysis Results", lines=10)

        pdf_chatbot = gr.Chatbot(visible=False)
        pdf_msg = gr.Textbox(visible=False)
        pdf_clear = gr.Button("Clear", visible=False)

        analyze_button.click(analyze_and_chat, inputs=[expense_file_input, expense_password_input], outputs=[analysis_output, pdf_chatbot, pdf_msg])
        pdf_msg.submit(expense_chat, [pdf_msg, pdf_chatbot], pdf_chatbot)
        pdf_clear.click(lambda: None, None, pdf_chatbot, queue=False)

    with gr.Tab("Finance Chat"):
        general_chatbot = gr.Chatbot()
        general_msg = gr.Textbox()
        general_clear = gr.Button("Clear")

        general_msg.submit(lambda message, history: [(message, llm(message))], [general_msg, general_chatbot], general_chatbot)
        general_clear.click(lambda: None, None, general_chatbot, queue=False)

demo.launch()
