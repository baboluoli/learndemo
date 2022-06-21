import streamlit as st
import faiss
import pandas as pd
import gdown
from sentence_transformers import SentenceTransformer
from os.path import exists
import zipfile
import numpy as np
from haystack.document_stores import FAISSDocumentStore
from haystack.retriever.dense import EmbeddingRetriever
from haystack.generator.transformers import Seq2SeqGenerator
from haystack.pipelines import GenerativeQAPipeline


@st.experimental_singleton
def init_faiss():
    url1 = 'https://drive.google.com/file/d/1nGBi0rsPVUf-TJfe89h15ZXUeXwXfEUE/view?usp=sharing'
    url2 = 'https://drive.google.com/file/d/1VilhsB1qIHrenVnjRmXJREBtPWLkfYqA/view?usp=sharing'
    url3 = 'https://drive.google.com/file/d/1HX2Ikq_t6UX7TTjyArW9PLFdJ6U6WK7A/view?usp=sharing'

    if not exists("investoqa.index"):
        gdown.download(url1, "investoqa.index", quiet=False, fuzzy=True)
    if not exists("investoqa.json"):
        gdown.download(url2, "investoqa.json", quiet=False, fuzzy=True)
    if not exists("faiss_document_store.db"):
        gdown.download(url3, "faiss_document_store.db", quiet=False, fuzzy=True)

    return FAISSDocumentStore.load("investoqa.index")


@st.experimental_singleton
def init_retriever():
    return EmbeddingRetriever(
        document_store=index,
        embedding_model="flax-sentence-embeddings/all_datasets_v3_mpnet-base",
        model_format="sentence_transformers"
    )


@st.experimental_singleton
def init_generator():
    return Seq2SeqGenerator(model_name_or_path="vblagoje/bart_lfqa")


@st.experimental_singleton
def init_pipe():
    return GenerativeQAPipeline(generator, retriever)



index = init_faiss()
retriever = init_retriever()
generator = init_generator()
pipe = init_pipe()


def card(answers):
    items = [f"""
        <p>{answer_container.answer}</p>
    """ for answer_container in answers]
    return st.markdown(f"""
        <div style="display: flex; flex-flow: row wrap; text-align: center; justify-content: center;">
        {''.join(items)}
        </div>
    """, unsafe_allow_html=True)


st.write("""
## ⚡️ LongformQA Demo ⚡️
""")

query = st.text_input("What are you looking for?", "")

if query != "":
    with st.spinner(text="Fetching answers..."):
        result = pipe.run(
            query=query,
            params={
                "Retriever": {"top_k": 5},
                "Generator": {"top_k": 1}
            })

        answers = result["answers"]

    with st.spinner(text="Here is your answer 💦💦💦 🚀🚀🚀"):
        card(answers)
