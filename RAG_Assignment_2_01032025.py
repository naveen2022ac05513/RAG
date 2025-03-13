import pandas as pd
import numpy as np
import faiss
import requests
from io import StringIO
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import streamlit as st

"""
### Component 1: Data Collection & Preprocessing
- Load financial dataset (last two years of company financials).
- Clean and structure data for retrieval.
"""

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

"""
### Component 2: Basic RAG Implementation
- Convert financial documents into text chunks.
- Embed using a pre-trained model.
- Store and retrieve using a basic vector database (FAISS).
"""

def generate_text_chunks(df):
    chunks = [
        f"In {row['Year']}, {row['Company']} ({row['Category']}) had a revenue of ${row['Revenue']}B, "
        f"a net income of ${row['Net Income']}B, and an EBITDA of ${row['EBITDA']}B. "
        f"The debt-to-equity ratio was {row['Debt/Equity Ratio']}, with an ROE of {row['ROE']}%. "
        f"The company had {row['Number of Employees']} employees."
        for _, row in df.iterrows()
    ]
    return chunks

# Embed text chunks and store in FAISS
def create_vector_store(chunks):
    embed_model = SentenceTransformer('BAAI/bge-small-en')
    embeddings = embed_model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, chunks, embed_model

"""
### Component 3: Advanced RAG Implementation
- Improve retrieval by combining BM25 keyword-based search with vector embeddings.
- Experiment with chunk sizes and retrieval methods.
- Implement re-ranking using a cross-encoder.
"""

def create_bm25(chunks):
    tokenized_corpus = [chunk.split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, tokenized_corpus

# Query Processing & Retrieval
def retrieve(query, index, chunks, bm25, bm25_corpus, embed_model):
    query_embedding = embed_model.encode([query])
    _, faiss_indices = index.search(query_embedding, 5)
    faiss_results = [chunks[i] for i in faiss_indices[0]]
    
    bm25_scores = bm25.get_scores(query.split())
    bm25_indices = np.argsort(bm25_scores)[-5:][::-1]
    bm25_results = [chunks[i] for i in bm25_indices]
    
    combined_results = list(set(faiss_results + bm25_results))
    return combined_results[:5]

# Re-ranking with Cross-Encoder
def rerank(query, results):
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    pairs = [(query, doc) for doc in results]
    scores = reranker.predict(pairs)
    ranked_results = [doc for _, doc in sorted(zip(scores, results), reverse=True)]
    return ranked_results[0]

"""
### Component 4: UI Development
- Build an interactive Streamlit UI.
- Accept user queries and display the best-ranked financial response.
"""

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
        best_answer = rerank(query, results)
        
        """
        ### Component 5: Guardrail Implementation
        - Output-side filtering: Ensure responses are financial-related and non-misleading.
        """
        if "revenue" in best_answer.lower() or "net income" in best_answer.lower():
            st.write(f"**Answer:** {best_answer}")
        else:
            st.write("**Response:** This question might be out of scope for financial data.")

        """
        ### Component 6: Testing & Validation
        - Test cases: High-confidence, low-confidence, and irrelevant questions.
        """
        
if __name__ == "__main__":
    main()
