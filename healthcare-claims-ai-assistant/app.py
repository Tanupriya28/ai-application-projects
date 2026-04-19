import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

st.set_page_config(page_title="Healthcare Claims Assistant", page_icon="🏥")
st.sidebar.title("About Project")
st.sidebar.write("GenAI Healthcare Claims Assistant using RAG")
st.sidebar.write("Documents Loaded: 3 HDFC Policy PDFs")
st.sidebar.write("LLM: Groq Llama 3.1")

st.title("🏥 Healthcare Claims AI Assistant")
st.write("Ask questions from HDFC insurance policy documents")

docs = []

uploaded_files = st.file_uploader(
    "Upload Insurance Policy PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files and len(uploaded_files) > 5:
    st.warning("Please upload maximum 5 PDFs.")
    st.stop()

if uploaded_files:
    with st.spinner("Processing uploaded PDFs..."):
        for file in uploaded_files:
            with open(file.name, "wb") as f:
                f.write(file.getbuffer())

            loader = PyPDFLoader(file.name)
            docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100
)

if docs:
    chunks = splitter.split_documents(docs)
else:
    st.warning("Please upload at least one PDF.")
    st.stop()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_documents(chunks, embeddings)

llm = ChatGroq(
    model_name="llama-3.1-8b-instant"
)

prompt = PromptTemplate(
    template="""
You are a healthcare insurance claims assistant.

Use only the context provided below.

If answer is not available, say:
"Information not found in provided policy documents."

Give concise professional bullet-point answers.

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k":4}),
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

query = st.text_input(
    "Enter your question",
    placeholder="Example: What are claim submission timelines?"
)
if query:
    with st.spinner("Searching documents and generating answer..."):
        result = qa(query)

        st.divider()
        st.subheader("Answer")
        st.write(result["result"])

        st.divider()
        st.subheader("Sources")
        shown = set()

        for doc in result["source_documents"]:
            name = os.path.splitext(os.path.basename(doc.metadata["source"]))[0]
            page = doc.metadata["page"] + 1

            key = f"{name} - Page {page}"

            if key not in shown:
                st.write(key)
                shown.add(key)

    st.success("Answer ready!")