import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import streamlit as st

# Data Collection & Preprocessing
def preprocess(data):
    data.replace('null', None, inplace=True)
    data.fillna("None", inplace=True)
    data.dropna(inplace=True)
    return data

# Provide the correct file path to your CSV file
financial_data = preprocess(pd.read_csv('C:/Users/USER/Documents/Assignment 2 RAG/Financial Statements.csv'))

# Combine relevant columns into text chunks for embedding
financial_data['text_chunk'] = financial_data.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
chunks = financial_data['text_chunk'].tolist()

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(chunks)

def retrieve(query, top_k=5):
    query_vec = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[-top_k:][::-1]
    return [(chunks[idx], similarity_scores[idx]) for idx in top_indices]

# Advanced RAG Implementation
bm25 = BM25Okapi([chunk.split() for chunk in chunks])

def bm25_retrieve(query, top_k=5):
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(chunks[idx], scores[idx]) for idx in top_indices]

def re_rank(query, candidates):
    query_vec = vectorizer.transform([query])
    inputs = [vectorizer.transform([candidate]) for candidate, _ in candidates]
    scores = [cosine_similarity(query_vec, candidate_vec).flatten()[0] for candidate_vec in inputs]
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
    else:
        st.write("Invalid query. Please ask a relevant financial question.")
