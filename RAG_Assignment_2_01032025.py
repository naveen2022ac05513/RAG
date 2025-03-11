import pandas as pd
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import normalize
import os
import traceback
from huggingface_hub import snapshot_download

# Install missing dependencies
os.system("pip install --upgrade sentence-transformers pandas numpy streamlit faiss-cpu rank-bm25 scikit-learn huggingface_hub")

# 1. Data Collection & Preprocessing
url = 'https://raw.githubusercontent.com/naveen2022ac05513/RAG/main/Financial%20Statements.csv'
financial_data = pd.read_csv(url, on_bad_lines='skip')

financial_data.fillna("None", inplace=True)

# Ensure 'Year' column is numeric
try:
    financial_data['Year'] = pd.to_numeric(financial_data['Year'], errors='coerce')
except KeyError:
    st.error("Error: 'Year' column not found in dataset.")
    st.stop()

recent_data = financial_data[financial_data['Year'] >= (pd.to_datetime('today').year - 2)]
financial_texts = recent_data.apply(lambda row: ' '.join(row.values.astype(str)), axis=1).tolist()

# 2. Basic RAG Implementation
## Chunking Financial Documents
def chunk_text(text, chunk_size=300):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

chunks = []
for text in financial_texts:
    chunks.extend(chunk_text(text))

## Embedding Model & Indexing
try:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    embeddings = normalize(embeddings, axis=1, norm='l2')

    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings)
except Exception as e:
    st.error(f"Error loading embedding model: {e}")
    st.text(traceback.format_exc())

# 3. Advanced RAG Implementation
## BM25 Index
tokenized_corpus = [text.split() for text in chunks]
bm25 = BM25Okapi(tokenized_corpus)

## Cross-Encoder for Re-Ranking
try:
    reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Corrected model name
    reranker = CrossEncoder(reranker_model)
except Exception as e:
    st.error(f"Error loading cross-encoder model: {e}")
    st.text(traceback.format_exc())
    
    # Attempt manual download
    try:
        snapshot_download(repo_id=reranker_model, cache_dir="./models")
        reranker = CrossEncoder("./models/cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception as e:
        st.error(f"Failed to manually load cross-encoder model: {e}")
        st.text(traceback.format_exc())

# 4. Guard Rail Implementation
def validate_query(query):
    # Guardrail: Ensure query is finance-related and prevent harmful inputs
    blacklist = ['hack', 'attack', 'delete', 'fraud']
    keywords = ['revenue', 'profit', 'loss', 'earnings', 'income', 'expenses']
    if any(b in query.lower() for b in blacklist):
        return False
    return any(kw in query.lower() for kw in keywords)

# 5. Confidence Scoring
def calculate_confidence(score):
    return min(1.0, max(0.1, score / 10))  # Normalize confidence score

## Hybrid Retrieval (BM25 + FAISS) with Re-Ranking
def retrieve_documents(query):
    """ Hybrid retrieval using BM25 and FAISS """
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    query_embedding = normalize(query_embedding, axis=1, norm='l2')
    
    # BM25 Retrieval
    bm25_scores = bm25.get_scores(query.split())
    bm25_top_k = np.argsort(bm25_scores)[-5:][::-1]
    
    # FAISS Retrieval
    _, faiss_top_k = faiss_index.search(query_embedding, 5)
    faiss_top_k = faiss_top_k.flatten()
    
    # Combine & Re-rank
    combined_indices = list(set(bm25_top_k) | set(faiss_top_k))
    retrieved_texts = [chunks[i] for i in combined_indices]
    rerank_scores = reranker.predict([(query, text) for text in retrieved_texts])
    ranked_results = sorted(zip(combined_indices, rerank_scores), key=lambda x: x[1], reverse=True)
    
    return [(chunks[i], score) for i, score in ranked_results]

# 6. UI Development (e.g., Streamlit)
st.title("Financial Q&A using Advanced RAG")
query = st.text_input("Enter your financial question:")
if query:
    if validate_query(query):
        results = retrieve_documents(query)
        if results:
            st.write("### Top Answers")
            for text, score in results:
                st.write(f"{text}\n**Confidence Score:** {calculate_confidence(score):.2f}")
        else:
            st.write("No relevant information found.")
    else:
        st.write("Invalid query. Please ask a financial-related question.")
