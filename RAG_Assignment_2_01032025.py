import pandas as pd
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import normalize
import os
import traceback
import re
from huggingface_hub import snapshot_download

# Install missing dependencies
os.system("pip install --upgrade sentence-transformers pandas numpy streamlit faiss-cpu rank-bm25 scikit-learn huggingface_hub")

# 1. Load & Preprocess Financial Data
url = 'https://raw.githubusercontent.com/naveen2022ac05513/RAG/main/Financial%20Statements.csv'
financial_data = pd.read_csv(url, on_bad_lines='skip')
financial_data.fillna("None", inplace=True)

# Ensure 'Year' column is numeric
try:
    financial_data['Year'] = pd.to_numeric(financial_data['Year'], errors='coerce')
except KeyError:
    st.error("Error: 'Year' column not found in dataset.")
    st.stop()

# 2. Extract Year from Query
def extract_year(query):
    match = re.search(r'\b(20[0-2][0-9])\b', query)
    return int(match.group(1)) if match else 2023  # Default year

# 3. Retrieve Lowest Revenue Company
def find_lowest_revenue_company(year):
    relevant_data = financial_data[financial_data['Year'] == year]
    if relevant_data.empty:
        return None, None
    try:
        relevant_data['Revenue'] = pd.to_numeric(relevant_data['Revenue'], errors='coerce')
        lowest_revenue_row = relevant_data.loc[relevant_data['Revenue'].idxmin()]
        return lowest_revenue_row.to_string(), 1.0
    except KeyError:
        return None, None

# 4. Retrieve General Information using RAG
def retrieve_documents(query):
    year = extract_year(query)

    # Special case: Lowest Revenue Company
    if "lowest revenue" in query.lower() or "lowest earning" in query.lower():
        return find_lowest_revenue_company(year)

    # Filter Data for the Year
    relevant_data = financial_data[financial_data['Year'] == year]
    if relevant_data.empty:
        return None, None
    financial_texts = relevant_data.apply(lambda row: ' '.join(row.values.astype(str)), axis=1).tolist()

    # Chunk Texts
    def chunk_text(text, chunk_size=300):
        words = text.split()
        return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    chunks = [chunk for text in financial_texts for chunk in chunk_text(text)]

    # Load Embedding Model & FAISS Index
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    chunk_embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    chunk_embeddings = normalize(chunk_embeddings, axis=1, norm='l2')
    
    faiss_index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
    faiss_index.add(chunk_embeddings)

    # BM25 Indexing
    tokenized_corpus = [text.split() for text in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    # Encode Query
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    query_embedding = normalize(query_embedding, axis=1, norm='l2')

    # BM25 & FAISS Retrieval
    bm25_scores = bm25.get_scores(query.split())
    bm25_top_k = np.argsort(bm25_scores)[-5:][::-1]
    _, faiss_top_k = faiss_index.search(query_embedding, 5)
    faiss_top_k = faiss_top_k.flatten()

    # Combine & Re-rank
    combined_indices = list(set(bm25_top_k) | set(faiss_top_k))
    retrieved_texts = [chunks[i] for i in combined_indices]

    if not retrieved_texts:
        return None, None  # No relevant results

    # Load Cross-Encoder for Re-Ranking
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    rerank_scores = reranker.predict([(query, text) for text in retrieved_texts])
    ranked_results = sorted(zip(retrieved_texts, rerank_scores), key=lambda x: x[1], reverse=True)

    return ranked_results[0] if ranked_results else (None, None)

# 5. Normalize Confidence Score
def calculate_confidence(score):
    return min(1.0, max(0.1, score / 10))

# 6. Streamlit UI
st.title("Financial Q&A using Advanced RAG")
query = st.text_input("Enter your financial question:")

if query:
    best_text, best_score = retrieve_documents(query)
    if best_text:
        st.write(f"### Answer from {extract_year(query)} Financial Data")
        st.write(f"{best_text}\n\n**Confidence Score:** {calculate_confidence(best_score):.2f}")
    else:
        st.write(f"No relevant information found for the year {extract_year(query)}.")
