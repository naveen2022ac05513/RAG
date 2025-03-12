import os
import faiss
import torch
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================
# 1. Data Collection & Preprocessing
# ==========================

# Load Dataset
url = 'https://github.com/naveen2022ac05513/RAG/raw/main/Financial%20Statements.csv'
df = pd.read_csv(url)

# Preprocess Data
def chunk_text(text, max_tokens=100):
    words = str(text).split()
    return [' '.join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]

df['text_chunks'] = df.iloc[:, 1].apply(lambda x: chunk_text(x))  # Assuming relevant text is in the second column

# Flatten chunks
all_chunks = [chunk for sublist in df['text_chunks'] for chunk in sublist]

# ==========================
# 2. Basic RAG Implementation
# ==========================

# Load Embedding Model
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = embedding_model.encode(all_chunks, convert_to_numpy=True)

# Vector Database Setup (FAISS)
vector_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(vector_dim)
index.add(embeddings)

# ==========================
# 3. Advanced RAG Implementation
# ==========================

# BM25 Setup
tokenized_chunks = [chunk.split() for chunk in all_chunks]
bm25 = BM25Okapi(tokenized_chunks)

# Load Cross-Encoder for Re-Ranking
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# ==========================
# 4. UI Development (Streamlit)
# ==========================

def hybrid_search(query, top_k=5):
    """Perform BM25 + Vector Retrieval + Re-Ranking"""
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)[0]
    
    # Vector Search
    D, I = index.search(query_embedding.reshape(1, -1), top_k)
    vector_results = [all_chunks[idx] for idx in I[0] if idx < len(all_chunks)]
    
    # BM25 Search
    bm25_results = bm25.get_top_n(query.lower().split(), all_chunks, n=top_k)
    
    # Merge Results
    combined_results = list(set(vector_results + bm25_results))
    
    # Re-Rank using Cross-Encoder
    scores = cross_encoder.predict([(query, doc) for doc in combined_results])
    ranked_results = [doc for _, doc in sorted(zip(scores, combined_results), reverse=True)]
    
    return ranked_results[:top_k]

# ==========================
# 5. Guardrail Implementation
# ==========================

# Load Small Language Model (SLM) for Response Generation
slm_model_name = "mistralai/Mistral-7B-Instruct-v0.1"

try:
    tokenizer = AutoTokenizer.from_pretrained(slm_model_name)
    slm_model = AutoModelForCausalLM.from_pretrained(
        slm_model_name, torch_dtype=torch.float16, device_map="auto"
    )
except Exception as e:
    st.error("Error loading the language model. Ensure model is accessible.")
    slm_model, tokenizer = None, None

def generate_response(query, context):
    """Generate a response using the small language model"""
    if not slm_model:
        return "Error: Model not loaded properly."

    input_text = f"Context: {context}\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt")
    output = slm_model.generate(**inputs, max_length=150)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def apply_guardrail(response, retrieved_docs):
    """Ensure response aligns with retrieved context"""
    return response if any(doc in response for doc in retrieved_docs) else "I'm not confident in my answer."

# ==========================
# 6. Testing & Validation (Streamlit UI)
# ==========================

st.title("Financial Q&A with RAG & Re-Ranking")
user_query = st.text_input("Enter your financial question:")
if user_query:
    retrieved_docs = hybrid_search(user_query)
    raw_response = generate_response(user_query, ' '.join(retrieved_docs))
    final_response = apply_guardrail(raw_response, retrieved_docs)
    st.write("### Answer:", final_response)
    st.write("### Retrieved Documents:", retrieved_docs)
