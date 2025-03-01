import pandas as pd
import numpy as np
import streamlit as st
from collections import Counter
from math import log, sqrt

# Data Collection & Preprocessing
def preprocess(data):
    data.replace('null', None, inplace=True)
    data.fillna("None", inplace=True)
    data.dropna(inplace=True)
    return data

# Provide the URL to your CSV file in the GitHub repository
url = 'https://raw.githubusercontent.com/naveen2022ac05513/RAG/main/Financial%20Statements.csv'

# Read the CSV file with error handling
try:
    financial_data = pd.read_csv(url, on_bad_lines='skip')
except pd.errors.ParserError as e:
    print("Error parsing file: ", e)
    financial_data = pd.read_csv(url, on_bad_lines='skip')

# Preprocess the data
financial_data = preprocess(financial_data)

# Combine relevant columns into text chunks for embedding
financial_data['text_chunk'] = financial_data.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
chunks = financial_data['text_chunk'].tolist()

# Basic Word Frequency Vectorizer
def word_frequency_vectorizer(docs):
    doc_count = len(docs)
    word_freq = [Counter(doc.split()) for doc in docs]
    doc_freq = Counter()
    for word in set().union(*[doc.keys() for doc in word_freq]):
        doc_freq[word] = sum(1 for doc in word_freq if word in doc)
    return word_freq, doc_freq, doc_count

def tf_idf_vectorizer(docs):
    word_freq, doc_freq, doc_count = word_frequency_vectorizer(docs)
    tf_idf = []
    for doc in word_freq:
        tf_idf.append({word: (freq / sum(doc.values())) * log(doc_count / doc_freq[word]) for word, freq in doc.items()})
    return tf_idf

def cosine_similarity(vec1, vec2):
    common_words = set(vec1.keys()) & set(vec2.keys())
    dot_product = sum(vec1[word] * vec2[word] for word in common_words)
    magnitude1 = sqrt(sum(val**2 for val in vec1.values()))
    magnitude2 = sqrt(sum(val**2 for val in vec2.values()))
    if not magnitude1 or not magnitude2:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)

tf_idf_docs = tf_idf_vectorizer(chunks)

def retrieve(query, top_k=5):
    query_vec = tf_idf_vectorizer([query])[0]
    similarity_scores = [cosine_similarity(query_vec, doc_vec) for doc_vec in tf_idf_docs]
    top_indices = np.argsort(similarity_scores)[-top_k:][::-1]
    return [(chunks[idx], similarity_scores[idx]) for idx in top_indices]

# Advanced RAG Implementation using BM25
def bm25_score(query, doc, k=1.5, b=0.75):
    words = query.split()
    query_freq = Counter(words)
    avg_doc_len = sum(len(doc.split()) for doc in chunks) / len(chunks)
    score = 0
    for word in words:
        if word in doc:
            term_freq = doc.split().count(word)
            doc_len = len(doc.split())
            score += (query_freq[word] * (k + 1) * term_freq) / (term_freq + k * (1 - b + b * (doc_len / avg_doc_len)))
    return score

def bm25_retrieve(query, top_k=5):
    scores = [bm25_score(query, doc) for doc in chunks]
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [(chunks[idx], scores[idx]) for idx in top_indices]

def re_rank(query, candidates):
    query_vec = tf_idf_vectorizer([query])[0]
    scores = [(doc, cosine_similarity(query_vec, tf_idf_vectorizer([doc])[0])) for doc, _ in candidates]
    ranked_candidates = sorted(scores, key=lambda x: x[1], reverse=True)
    return ranked_candidates

# Guard Rail Implementation
def validate_query(query):
    # Extract column names from the dataset
    allowed_keywords = list(financial_data.columns)
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
        
        # Display the column headers along with the answer
        columns = list(financial_data.columns)
        data = answer.split(' ')
        answer_with_headers = '\n'.join([f"{columns[i]}: {data[i]}" for i in range(len(columns))])
        st.write(answer_with_headers)
        
        st.write("### Confidence Score")
        st.write(confidence)

        # Option for new query
        st.text_input("Enter another question:")
        
    else:
        st.write("Invalid query. Please ask a relevant financial question.")
