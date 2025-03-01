{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "e89c26e6-6751-4ba8-9ae5-1ec2f6bed655",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Index(['Year', 'Company ', 'Category', 'Market Cap(in B USD)', 'Revenue',\n",
      "       'Gross Profit', 'Net Income', 'Earning Per Share', 'EBITDA',\n",
      "       'Share Holder Equity', 'Cash Flow from Operating',\n",
      "       'Cash Flow from Investing', 'Cash Flow from Financial Activities',\n",
      "       'Current Ratio', 'Debt/Equity Ratio', 'ROE', 'ROA', 'ROI',\n",
      "       'Net Profit Margin', 'Free Cash Flow per Share',\n",
      "       'Return on Tangible Equity', 'Number of Employees',\n",
      "       'Inflation Rate(in US)'],\n",
      "      dtype='object')\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "\n",
    "# Load the CSV file\n",
    "financial_data = pd.read_csv('C:/Users/USER/Documents/Assignment 2 RAG/Financial Statements.csv')\n",
    "\n",
    "# Print the column names\n",
    "print(financial_data.columns)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "558741fc-2ca7-4247-944e-41f9f56d9b26",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "abca4c2e-18b4-4fa0-b582-b490c507c994",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import pandas as pd\n",
    "from sentence_transformers import SentenceTransformer, util\n",
    "import numpy as np\n",
    "import faiss\n",
    "from rank_bm25 import BM25Okapi\n",
    "from sentence_transformers import CrossEncoder\n",
    "import streamlit as st\n",
    "\n",
    "# Disable symlink warning\n",
    "os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'\n",
    "\n",
    "# Data Collection & Preprocessing\n",
    "def preprocess(data):\n",
    "    data.dropna(inplace=True)\n",
    "    return data\n",
    "\n",
    "# Provide the correct file path to your CSV file\n",
    "financial_data = preprocess(pd.read_csv('C:/Users/USER/Documents/Assignment 2 RAG/Financial Statements.csv'))\n",
    "\n",
    "# Combine relevant columns into text chunks for embedding\n",
    "financial_data['text_chunk'] = financial_data.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)\n",
    "chunks = financial_data['text_chunk'].tolist()\n",
    "\n",
    "# Basic RAG Implementation\n",
    "model = SentenceTransformer('all-MiniLM-L6-v2')\n",
    "embeddings = model.encode(chunks, convert_to_tensor=True)\n",
    "\n",
    "index = faiss.IndexFlatL2(embeddings.shape[1])\n",
    "index.add(np.array(embeddings.cpu()))\n",
    "\n",
    "def retrieve(query, top_k=5):\n",
    "    query_embedding = model.encode(query, convert_to_tensor=True)\n",
    "    distances, indices = index.search(np.array(query_embedding.cpu()).reshape(1, -1), top_k)\n",
    "    return [(chunks[idx], distances[0][i]) for i, idx in enumerate(indices[0])]\n",
    "\n",
    "# Advanced RAG Implementation\n",
    "bm25 = BM25Okapi([chunk.split() for chunk in chunks])\n",
    "cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')\n",
    "\n",
    "def bm25_retrieve(query, top_k=5):\n",
    "    tokenized_query = query.split()\n",
    "    scores = bm25.get_scores(tokenized_query)\n",
    "    top_indices = np.argsort(scores)[::-1][:top_k]\n",
    "    return [(chunks[idx], scores[idx]) for idx in top_indices]\n",
    "\n",
    "def re_rank(query, candidates):\n",
    "    inputs = [[query, candidate] for candidate, _ in candidates]\n",
    "    scores = cross_encoder.predict(inputs)\n",
    "    ranked_candidates = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)\n",
    "    return ranked_candidates\n",
    "\n",
    "# Guard Rail Implementation\n",
    "def validate_query(query):\n",
    "    allowed_keywords = ['revenue', 'profit', 'expenses', 'assets', 'liabilities']\n",
    "    if any(keyword in query.lower() for keyword in allowed_keywords):\n",
    "        return True\n",
    "    else:\n",
    "        return False\n",
    "\n",
    "# UI Development (Streamlit)\n",
    "st.title(\"Financial Q&A using RAG Model\")\n",
    "\n",
    "query = st.text_input(\"Enter your financial question:\")\n",
    "if query:\n",
    "    if validate_query(query):\n",
    "        bm25_results = bm25_retrieve(query)\n",
    "        re_ranked_results = re_rank(query, bm25_results)\n",
    "        answer, confidence = re_ranked_results[0]\n",
    "        \n",
    "        st.write(\"### Answer\")\n",
    "        st.write(answer)\n",
    "        st.write(\"### Confidence Score\")\n",
    "        st.write(confidence)\n",
    "    else:\n",
    "        st.write(\"Invalid query. Please ask a relevant financial question.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ba722f0c-a58f-4341-9744-e15a44dd9a27",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
