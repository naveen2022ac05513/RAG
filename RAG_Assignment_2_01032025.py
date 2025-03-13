import pandas as pd
import numpy as np
import faiss
import requests
from io import StringIO
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import streamlit as st

# Component 1: Data Collection & Preprocessing
def download_data():
    url = "https://raw.githubusercontent.com/naveen2022ac05513/RAG/main/Financial%20Statements.csv"
    response = requests.get(url)
    if response.status_code == 200:
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)
        df.columns = df.columns.str.strip()  # Fix column names
        df[['Revenue', 'Net Income', 'EBITDA']] /= 1e3  # Convert to billions
        return df
    else:
        st.error("Failed to download dataset from GitHub.")
        return None

# Component 2: Basic RAG Implementation
def generate_text_chunks(df):
    chunks = [
        f"In {row['Year']}, {row['Company']} ({row['Category']}) had a revenue of ${row['Revenue']}B, "
        f"a net income of ${row['Net Income']}B, and an EBITDA of ${row['EBITDA']}B. "
        f"The debt-to-equity ratio was {row['Debt/Equity Ratio']}, with an ROE of {row['ROE']}%. "
        f"The company had {row['Number of Employees']} employees."
        for _, row in df.iterrows()
    ]
    return chunks

# Creating Vector Store for Embeddings
def create_vector_store(chunks):
    embed_model = SentenceTransformer('BAAI/bge-small-en')
    embeddings = embed_model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, chunks, embed_model

# Creating BM25 Index for Keyword Search
def create_bm25(chunks):
    tokenized_corpus = [chunk.split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, tokenized_corpus

# Component 3: Advanced RAG Implementation
def retrieve(query, index, chunks, bm25, bm25_corpus, embed_model):
    # Retrieve using FAISS
    query_embedding = embed_model.encode([query])
    faiss_distances, faiss_indices = index.search(query_embedding, 5)
    faiss_results = [(chunks[i], 1 / (1 + faiss_distances[0][j])) for j, i in enumerate(faiss_indices[0])]
    
    # Retrieve using BM25
    bm25_scores = bm25.get_scores(query.split())
    bm25_indices = np.argsort(bm25_scores)[-5:][::-1]
    bm25_results = [(chunks[i], bm25_scores[i]) for i in bm25_indices]
    
    # Combine and return results with confidence scores
    combined_results = list(set(faiss_results + bm25_results))
    return sorted(combined_results, key=lambda x: x[1], reverse=True)[:5]

# Implementing Re-Ranking with Cross-Encoders
def rerank(query, results):
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    pairs = [(query, doc[0]) for doc in results]
    scores = reranker.predict(pairs)
    ranked_results = sorted(zip(scores, results), reverse=True)
    return ranked_results[0][1][0], ranked_results[0][0]  # Returning best answer and confidence score

# Component 4: UI Development (Streamlit App)
def main():
    st.title("Financial RAG Chatbot")
    df = download_data()
    if df is None:
        return
    chunks = generate_text_chunks(df)
    index, chunks, embed_model = create_vector_store(chunks)
    bm25, bm25_corpus = create_bm25(chunks)
    
    query = st.text_input("Enter your financial question:")
    if st.button("Ask"):
        results = retrieve(query, index, chunks, bm25, bm25_corpus, embed_model)
        best_answer, confidence_score = rerank(query, results)
        
        # Component 5: Guard Rail Implementation (Output Filtering)
        if "revenue" in best_answer.lower() or "net income" in best_answer.lower():
            st.write(f"**Answer:** {best_answer}")
            st.write(f"**Confidence Score:** {confidence_score:.2f}")
        else:
            st.write("**Response:** This question might be out of scope for financial data.")
            st.write(f"**Confidence Score:** {confidence_score:.2f}")
        
if __name__ == "__main__":
    main()
