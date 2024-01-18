import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
# Importing necessary libraries and modules

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
# Loading environment variables from a .env file

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# Configuring Google Generative AI with the API key from the environment variables


def get_pdf_text(pdf_docs):
    # Function to extract text from PDF documents
    # Uses PyPDF2 to read the text from each page in the PDFs
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(text):
    # Function to split text into smaller chunks for processing
    # Uses RecursiveCharacterTextSplitter from langchain library
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    # Function to generate vector representations for text chunks
    # Uses GoogleGenerativeAIEmbeddings for embeddings and FAISS for vector storage
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store=FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    # Function to set up a conversational chain for question-answering
    # Uses ChatGoogleGenerativeAI and load_qa_chain from langchain library
    prompt_template= """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)

    prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])
    chain=load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain

def user_input(user_question):
    # Function to handle user input (question) and retrieve relevant information
    # Utilizes GoogleGenerativeAIEmbeddings and FAISS for similarity search
    # Uses the conversational chain to provide a response to the user's question
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    # Main function to create the Streamlit web app interface
    st.set_page_config("Chat with Multiple PDF")
    st.header("Chat with PDF using Gemini")

    user_question = st.text_input("Ask a Question from the PDF Files")
    # Text input for user's question

    if user_question:
        user_input(user_question)
        # Calls user_input function if a question is provided by the user

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
    # Sidebar menu for uploading PDF files
    # Processes PDF files to extract text chunks and generate vector representations
    # Displays a success message when processing is done

if __name__=="__main__":
    main()