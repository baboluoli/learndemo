import streamlit as st
import faiss
import requests
import pandas as pd
import gdown
from sentence_transformers import SentenceTransformer

@st.experimental_singleton
def init_faiss():
    url = 'https://drive.google.com/file/d/1ANmjtV9rk9Tkv1f-7CSI9SC7lFLOeuVb/view?usp=sharing'

    gdown.download(url, "investo.index", quiet=False, fuzzy=True)

    return faiss.read_index("investo.index")
    
@st.experimental_singleton
def init_retriever():
    return SentenceTransformer('microsoft/mpnet-base')

@st.experimental_singleton
def init_passages():
    url = "https://drive.google.com/file/d/1kMnSiDJ5r6J2P5kOK_L0ww9utF8WfnP2/view?usp=sharing"

    gdown.download(url, "data.csv", quiet=False)

    df = pd.read_csv("data.csv")

    passages = list(df["full_text"])

    return passages

index = init_faiss()
retriever = init_retriever()
passages = init_passages()


def card(passages):
    items = [f"""
        <p>{passage}</p>
    """ for passage in passages]
    return st.markdown(f"""
        <div style="display: flex; flex-flow: row wrap; text-align: center; justify-content: center;">
        {''.join(items)}
        </div>
    """, unsafe_allow_html=True)

 
st.write("""
## ⚡️ Semantic Search Demo ⚡️
""")

query = st.text_input("What are you looking for?", "")

if query != "":
    with st.spinner(text="Similarity Searching..."):
        xq = retriever.encode([query]).tolist()
        xc = index.query(xq, top_k=30, include_metadata=True)
        
        urls = []
        for context in xc['results'][0]['matches']:
            urls.append(context['metadata']['url'])

    with st.spinner(text="Fetching GIFs 🚀🚀🚀"):
        card(urls)