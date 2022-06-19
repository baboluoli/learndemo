import streamlit as st
import faiss
import requests
import pandas as pd
import gdown
from sentence_transformers import SentenceTransformer
from os.path import exists
import zipfile
import numpy as np

@st.experimental_singleton
def init_faiss():
    url = 'https://drive.google.com/file/d/1--rpyd_vkPQKbSv0KbXcJGHKHoTDj5iC/view?usp=sharing'

    if not exists("investo.index"):
        gdown.download(url, "investo.index", quiet=False, fuzzy=True)

    return faiss.read_index("investo.index")
    
@st.experimental_singleton
def init_retriever():
    url = 'https://drive.google.com/file/d/1-12-MNuSUGMNAulFFoWANI4wvgta1zdH/view?usp=sharing'

    if not exists("model.zip"):
        gdown.download(url, "model.zip", quiet=False, fuzzy=True)

    with zipfile.ZipFile("model.zip","r") as zip_ref:
        zip_ref.extractall("model")

    return SentenceTransformer("./model")

@st.experimental_singleton
def init_passages():
    url = "https://drive.google.com/file/d/1XU35ze1d-DrzFPcb4y0VHLUHdwAcWFDA/view?usp=sharing"

    if not exists("data.csv"):
        gdown.download(url, "data.csv", quiet=False, fuzzy=True)

    df = pd.read_csv("data.csv")

    return df

index = init_faiss()
retriever = init_retriever()
passages_df = init_passages()


def card(passage_tuples):
    items = [f"""
        <p>{passage[0]}: {passage[1]}</p>
    """ for passage in passage_tuples]
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
        xq = retriever.encode([query], convert_to_tensor=False)
        D, I = index.search(np.array(xq), k=5)
        print(I)
        
        found_passages = []
        for score, id in zip(D[0], I[0]):
            passage_row = passages_df[passages_df["id"] == id]
            id = passage_row["id"].item()
            passage = passage_row["passages"].item()
            found_passages.append((id, passage))

    with st.spinner(text="Fetching passages 🚀🚀🚀"):
        card(found_passages)