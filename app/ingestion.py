import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_documents():

    docs = []

    for file in os.listdir("data/documents"):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(f"data/documents/{file}")
            docs.extend(loader.load())

    print("Total Pages Loaded:", len(docs))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    print("Total Chunks:", len(chunks))

    return chunks