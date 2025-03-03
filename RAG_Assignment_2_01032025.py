import pandas as pd
import numpy as np
import streamlit as st

# Data Collection & Preprocessing
def preprocess(data):
    data.replace('null', None, inplace=True)
    data.fillna("None", inplace=True)
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

# Guard Rail Implementation
def validate_query(query):
    # Extract column names from the dataset
    allowed_keywords = list(financial_data.columns)
    # Check if any of the keywords is in the query
    if any(keyword.lower() in query.lower() for keyword in allowed_keywords):
        return True
    else:
        return False

# Calculate Confidence Score
def calculate_confidence(query, result):
    if "highest revenue" in query.lower() or "lowest revenue" in query.lower():
        return 1.0  # High confidence for specific queries
    elif result.empty:
        return 0.0  # No matching data
    else:
        return min(1.0, max(0.1, len(result) / len(financial_data)))

# Query Processing
def process_query(query):
    query_lower = query.lower()
    if "highest revenue" in query_lower:
        result = financial_data.loc[financial_data['Revenue'].idxmax()]
    elif "lowest revenue" in query_lower:
        result = financial_data.loc[financial_data['Revenue'].idxmin()]
    elif "no revenue" in query_lower:
        result = financial_data[financial_data['Revenue'] == 0]
    else:
        result = None
    return result

# UI Development (Streamlit)
st.title("Financial Q&A using RAG Model")

query = st.text_input("Enter your financial question:")
if query:
    if validate_query(query):
        result = process_query(query)
        confidence = calculate_confidence(query, result)
        if result is not None and not result.empty:
            st.write("### Answer")
            for col, val in result.items():
                st.write(f"{col}: {val}")
            st.write(f"**Confidence Score:** {confidence:.2f}")
        else:
            st.write("No matching data found for your query.")
            st.write(f"**Confidence Score:** {confidence:.2f}")

        # Option for new query
        next_query = st.text_input("Enter your next financial question:")
        if next_query:
            if validate_query(next_query):
                result = process_query(next_query)
                confidence = calculate_confidence(next_query, result)
                if result is not None and not result.empty:
                    st.write("### Answer")
                    for col, val in result.items():
                        st.write(f"{col}: {val}")
                    st.write(f"**Confidence Score:** {confidence:.2f}")
                else:
                    st.write("No matching data found for your query.")
                    st.write(f"**Confidence Score:** {confidence:.2f}")
            else:
                st.write("Invalid query. Please ask a relevant financial question.")
    else:
        st.write("Invalid query. Please ask a relevant financial question.")
