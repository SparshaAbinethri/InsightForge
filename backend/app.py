import streamlit as st
import requests

st.title("InsightForge AI")

query = st.text_input("Ask something:")

if st.button("Submit"):
    response = requests.post(
        "http://backend:8000/ask",
        params={"query": query}
    )
    st.write(response.json()["response"])
