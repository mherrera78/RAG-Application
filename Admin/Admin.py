import boto3
import streamlit as st
import os
import uuid
from langchain_community.vectorstores import FAISS


## s3 client
region = 'us-west-2'
s3_client = boto3.client('s3', region_name=region)
BUCKET_NAME = os.getenv('BUCKET_NAME')

## Bedrock client
from langchain_community.embeddings import BedrockEmbeddings    
## Text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
#pdf loader
from langchain_community.document_loaders import PyPDFLoader

bedrock_client = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)


## Split the text into chunks
def split_text(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs

def create_vector_store(request_id, docs):
    vectorStore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    file_name = f"{request_id}.bin"
    folder_path = "/tmp"
    vectorStore_faiss.save_local(index_name=file_name, folder_path=folder_path)

    #st.write(f"Uploading vector store to s3: {file_name}")  
    #st.write(f"Folder path: {folder_path}")
    #st.write(f"{folder_path}/{file_name}.faiss")
    #st.write(f"{folder_path}/{file_name}.pkl")
    
    ## upload the vector store to s3
    s3_client.upload_file(Filename=f"{folder_path}/{file_name}.faiss", Bucket=BUCKET_NAME, Key="my_faiss.faiss")
    s3_client.upload_file(Filename=f"{folder_path}/{file_name}.pkl", Bucket=BUCKET_NAME, Key="my_faiss.pkl")

    return True


#main method
def main():
    st.write("This is an Admin site for Compensation Chat application!")
    uploaded_file = st.file_uploader("Choose a file", type="pdf")

    if uploaded_file is not None:
        request_id = str(uuid.uuid4())
        st.write(f"Request ID: {request_id} generated successfully...")
        st.write(f"================================================")
        saved_file_name = f"{request_id}.pdf"

        with open(saved_file_name, mode="wb") as w:
            w.write(uploaded_file.getvalue())

        loader = PyPDFLoader(saved_file_name)
        pages = loader.load_and_split()
        st.write(f"Total pages in PDF: {len(pages)} pages...")
        st.write(f"================================================")

        #split the text into chunks
        splitted_docs = split_text(pages, 1000, 200)
        st.write(f"Document successfully splitted in {len(splitted_docs)} chunks...")
        st.write(f"================================================")
        
        #st.write(f"================================================")
        #st.write(splitted_docs[0])
        #st.write(f"================================================")
        #st.write(splitted_docs[1])

        st.write("Creating vector store...")
        st.write(f"================================================")
        result = create_vector_store(request_id, splitted_docs)
    
        if result:
            st.write("PDF processed successfully!")
        else:
            st.write("Failed to create vector store! Check the logs.")


if __name__ == "__main__":
    main()
