import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

@st.cache_resource
def load_and_process_pdf():
    file_path = "avr_overview.pdf" # File ka naam jise tum upload karoge
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    chunks = [doc.page_content for doc in docs]

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    return model, index, chunks

@st.cache_resource
def load_qa_pipeline():
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

st.set_page_config(
    page_title="RAG-Based Q&A",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("Musk Abhishek ka RAG Project 🤖")
st.markdown("---")
st.subheader("Sawal-Jawab: Ek Continuous Process")

st.write("Apne PDF se sambandhit sawal yahan type karein:")

model, index, chunks = load_and_process_pdf()
qa_pipeline = load_qa_pipeline()

user_question = st.text_area("Sawal", height=100, placeholder="AVR Architecture ke baare mein bataiye?")

if st.button("Jawab Dhoondhein"):
    if user_question:
        with st.spinner("Jawab Dhoonda jaa raha hai..."):
            question_embedding = model.encode([user_question])
            k = 3
            distances, indices = index.search(question_embedding, k)
            
            context = ""
            for i in indices[0]:
                context += chunks[i] + " "
            
            result = qa_pipeline(question=user_question, context=context)
            
            st.markdown("### Answer")
            st.write(result['answer'])
            
            st.markdown("---")
            st.caption("Pustak se prapt jankari:")
            st.write(context)
    else:
        st.warning("Kripya apna sawal likhein.")