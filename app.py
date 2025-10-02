import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
from rank_bm25 import BM25Okapi

# Tumhari file ka sahi naam
PDF_FILE_NAME = "avr.pdf"

@st.cache_resource
def load_and_process_pdf():
    file_path = os.path.join(os.path.dirname(__file__), PDF_FILE_NAME)
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    chunks = [doc.page_content for doc in docs]
    
    return chunks

@st.cache_resource
def load_embedding_model_and_qa_pipeline():
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    return sbert_model, qa_pipeline

def perform_hybrid_search(query, chunks, sbert_model, faiss_index):
    # Semantic Search
    query_embedding = sbert_model.encode([query])
    distances, indices = faiss_index.search(query_embedding, 5) # Top 5 semantic results
    semantic_results = [chunks[i] for i in indices[0]]
    
    # Keyword Search (BM25)
    tokenized_chunks = [doc.split(" ") for doc in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    tokenized_query = query.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Keyword results ko score ke hisab se sort karein
    keyword_indices = sorted(range(len(bm25_scores)), key=lambda k: bm25_scores[k], reverse=True)[:5]
    keyword_results = [chunks[i] for i in keyword_indices]
    
    # Dono results ko jod dein (unique results ko hi lein)
    combined_results = list(dict.fromkeys(semantic_results + keyword_results))
    
    return " ".join(combined_results[:5]) # Top 5 ko hi wapas karein

st.set_page_config(
    page_title="Hybrid RAG",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("Hybrid RAG: Keyword aur Semantic Search ka Sangam")
st.markdown("---")
st.subheader("Sawal-Jawab: Ek Naya Tarika")

st.write("Apna sawal likhein:")

pdf_chunks = load_and_process_pdf()
sbert_model, qa_pipeline = load_embedding_model_and_qa_pipeline()

# FAISS index PDF chunks ke liye
embeddings = sbert_model.encode(pdf_chunks)
dimension = embeddings.shape[1]
pdf_index = faiss.IndexFlatL2(dimension)
pdf_index.add(embeddings)

user_question = st.text_area("Sawal", height=100, placeholder="AVR ki internal memory kya hai?")

if st.button("Jawab Dhoondhein"):
    if user_question:
        with st.spinner("Jawab Dhoonda jaa raha hai..."):
            # Hybrid Search ka upyog karein
            combined_context = perform_hybrid_search(user_question, pdf_chunks, sbert_model, pdf_index)
            
            # Generation Step
            result = qa_pipeline(question=user_question, context=combined_context)
            
            st.markdown("### Jawab")
            st.write(result['answer'])
            
            st.markdown("---")
            st.caption("Context:")
            st.write(combined_context)
    else:
        st.warning("Kripya apna sawal likhein.")
