import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
apikey = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=apikey)

# ✅ Use Hugging Face local embeddings
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = get_embeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in provided context just say,
    "answer is not available in the context".
    Don't provide the wrong answer.
    Context:
    {context}
    Question:
    {input}
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.3
    )
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "input"]
    )
    document_chain = create_stuff_documents_chain(model, prompt)
    embeddings = get_embeddings()
    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = db.as_retriever(search_kwargs={"k": 3})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

def user_input(user_question):
    if not os.path.exists("faiss_index"):
        st.warning("⚠️ Please upload and process a PDF first.")
        return
    try:
        chain = get_conversational_chain()
        response = chain.invoke({"input": user_question})
        st.write("Reply:", response["answer"])
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini + HuggingFace💁")
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings  # ✅ new import

# Load environment variables
load_dotenv()
apikey = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=apikey)


# ✅ Use Hugging Face local embeddings
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = get_embeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in provided context just say,
    "answer is not available in the context".
    Don't provide the wrong answer.

    Context:
    {context}

    Question:
    {input}
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",   # ✅ unchanged
        temperature=0.3
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "input"]
    )

    document_chain = create_stuff_documents_chain(model, prompt)

    embeddings = get_embeddings()
    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = db.as_retriever(search_kwargs={"k": 3})

    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain


def user_input(user_question):

    if not os.path.exists("faiss_index"):
        st.warning("⚠️ Please upload and process a PDF first.")
        return

    try:
        chain = get_conversational_chain()
        response = chain.invoke({"input": user_question})

        st.write("Reply:", response["answer"])

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")



def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini + HuggingFace💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


if __name__ == "__main__":
    main()
