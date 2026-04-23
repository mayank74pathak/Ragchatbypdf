import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# ✅ Use Hugging Face local embeddings
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = get_embeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the provided context, just say:
    "Answer is not available in the context".
    Do NOT make up answers.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGroq(
        groq_api_key=st.secrets["GROQ_API_KEY"],
        model_name="llama-3.1-8b-instant",  # 🔥 fast model
        temperature=0.3
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = load_qa_chain(
        model,
        chain_type="stuff",
        prompt=prompt
    )

    return chain


def user_input(user_question):
    embeddings = get_embeddings()

    new_db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    st.write("### Reply:")
    st.write(response["output_text"])


def main():
    st.set_page_config(page_title="Chat PDF", layout="wide")

    st.header("📄 Chat with PDF using Groq + HuggingFace")

    user_question = st.text_input("Ask a question from your PDF files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("📂 Menu")

        pdf_docs = st.file_uploader(
            "Upload your PDF files",
            accept_multiple_files=True
        )

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)

                if not raw_text.strip():
                    st.error("No readable text found in PDFs")
                    return

                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)

                st.success("Processing complete!")


if __name__ == "__main__":
    main()
