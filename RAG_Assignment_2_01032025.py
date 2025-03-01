import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import streamlit as st

# Disable symlink warning
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Data Collection & Preprocessing
def preprocess(data):
    data.fillna("None", inplace=True)
    data.dropna(inplace=True)
    return data

# Provide the correct file path to your CSV file
financial_data = preprocess(pd.read_csv('C:/Users/USER/Documents/Assignment 2 RAG/Financial Statements.csv'))

# Combine relevant columns into text chunks for embedding
financial_data['text_chunk'] = financial_data.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
chunks = financial_data['text_chunk'].tolist()

# Basic RAG Implementation
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks, convert_to_tensor=True)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings.cpu()))

def retrieve(query, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    distances, indices = index.search(np.array(query_embedding.cpu()).reshape(1, -1), top_k)
    return [(chunks[idx], distances[0][i]) for i, idx in enumerate(indices[0])]

# Advanced RAG Implementation
bm25 = BM25Okapi([chunk.split() for chunk in chunks])
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def bm25_retrieve(query, top_k=5):
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(chunks[idx], scores[idx]) for idx in top_indices]

def re_rank(query, candidates):
    inputs = [[query, candidate] for candidate, _ in candidates]
    scores = cross_encoder.predict(inputs)
    ranked_candidates = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return ranked_candidates

# Guard Rail Implementation
def validate_query(query):
    allowed_keywords = ['revenue', 'profit', 'expenses', 'assets', 'liabilities']
    if any(keyword in query.lower() for keyword in allowed_keywords):
        return True
    else:
        return False

# UI Development (Streamlit)
st.title("Financial Q&A using RAG Model")

query = st.text_input("Enter your financial question:")
if query:
    if validate_query(query):
        bm25_results = bm25_retrieve(query)
        re_ranked_results = re_rank(query, bm25_results)
        answer, confidence = re_ranked_results[0]
        
        st.write("### Answer")
        st.write(answer)
        st.write("### Confidence Score")
        st.write(confidence)
    else:
        st.write("Invalid query. Please ask a relevant financial question.")
