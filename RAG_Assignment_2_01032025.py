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

# 2. Extract Year from Query
def extract_year(query):
    """Extracts a four-digit year from the query, defaulting to 2023 if none is found."""
    match = re.search(r'\b(20[0-2][0-9])\b', query)  # Matches years 2000-2029
    if match:
        return int(match.group(1))
    return 2023  # Default to the latest available data

# 3. Get Data for the Specified Year
def get_relevant_data(year):
    """Filters financial data for the specified year."""
    relevant_data = financial_data[financial_data['Year'] == year]
    if relevant_data.empty:
        return None  # No data for the requested year
    return relevant_data.apply(lambda row: ' '.join(row.values.astype(str)), axis=1).tolist()

# 4. Chunking Financial Documents
def chunk_text(text, chunk_size=300):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# 5. Load Embedding Model & FAISS Index
try:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    st.error(f"Error loading embedding model: {e}")
    st.text(traceback.format_exc())

# 6. Load Cross-Encoder for Re-Ranking
try:
    reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker = CrossEncoder(reranker_model)
except Exception as e:
    st.error(f"Error loading cross-encoder model: {e}")
    st.text(traceback.format_exc())

# 7. Hybrid Retrieval (BM25 + FAISS) with Re-Ranking
def retrieve_documents(query):
    """Retrieve the best-matching document for the specified year."""
    year = extract_year(query)  # Extract year from query
    filtered_texts = get_relevant_data(year)  # Get data only for that year

    if not filtered_texts:
        return None, None  # No relevant data found for the year

    # Chunk the filtered texts
    filtered_chunks = []
    for text in filtered_texts:
        filtered_chunks.extend(chunk_text(text))

    # Encode the chunks
    chunk_embeddings = embedding_model.encode(filtered_chunks, convert_to_numpy=True)
    chunk_embeddings = normalize(chunk_embeddings, axis=1, norm='l2')

    # Create FAISS index for the filtered data
    faiss_index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
    faiss_index.add(chunk_embeddings)

    # BM25 Indexing for filtered chunks
    tokenized_filtered_corpus = [text.split() for text in filtered_chunks]
    bm25_filtered = BM25Okapi(tokenized_filtered_corpus)

    # Encode query
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    query_embedding = normalize(query_embedding, axis=1, norm='l2')

    # BM25 Retrieval
    bm25_scores = bm25_filtered.get_scores(query.split())
    bm25_top_k = np.argsort(bm25_scores)[-5:][::-1]

    # FAISS Retrieval
    _, faiss_top_k = faiss_index.search(query_embedding, 5)
    faiss_top_k = faiss_top_k.flatten()

    # Combine results
    combined_indices = list(set(bm25_top_k) | set(faiss_top_k))
    retrieved_texts = [filtered_chunks[i] for i in combined_indices]

    if not retrieved_texts:
        return None, None  # No results found for the year

    # Re-rank with Cross-Encoder
    rerank_scores = reranker.predict([(query, text) for text in retrieved_texts])
    ranked_results = sorted(zip(retrieved_texts, rerank_scores), key=lambda x: x[1], reverse=True)

    return ranked_results[0]  # Return the best-ranked result

# 8. Confidence Scoring
def calculate_confidence(score):
    """Normalize confidence score between 0.1 and 1.0"""
    return min(1.0, max(0.1, score / 10))

# 9. Streamlit UI Development
st.title("Financial Q&A using Advanced RAG")
query = st.text_input("Enter your financial question:")

if query:
    best_text, best_score = retrieve_documents(query)
    if best_text:
        st.write(f"### Answer from {extract_year(query)} Financial Data")
        st.write(f"{best_text}\n\n**Confidence Score:** {calculate_confidence(best_score):.2f}")
    else:
        st.write(f"No relevant information found for the year {extract_year(query)}.")
