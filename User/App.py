import boto3
import streamlit as st
import os
import uuid
from langchain_community.vectorstores import FAISS
from langchain.llms.bedrock import Bedrock 

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

## prompts and chains
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

bedrock_client = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

#define the folder path
folder_path = "/tmp/"

def load_index():
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.faiss", Filename=f"{folder_path}my_faiss.faiss")
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.pkl", Filename=f"{folder_path}my_faiss.pkl")


def get_llm():
    llm = Bedrock(model_id = "anthropic.claude-v2:1", client=bedrock_client, 
                  model_kwargs={
                      "max_tokens_to_sample":512
                      }
                )
    return llm



def get_response(llm, vectorstore, question):
    ## create prompt / template
    prompt_template = """

    Human: Please use the given context to provide concise answer to the question
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    <context>
    {context}
    </context>

    Question: {question}

    Assistant:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # search_kwargs={"k": 5} means that this similarity search will return 5 most similar chunks (documents) to the question
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    answer=qa({"query":question})
    return answer['result']



#main method
def main():
    st.header("This is the client site for Compensation Chat application using RAG technique!")
    
    #load indexes from s3

    load_index()
    dir_list = os.listdir(folder_path)
    #st.write(f"Files and Directories in {folder_path}")
    #st.write(dir_list)
    st.write("Preparing agent...")
    #create index
    faiss_index = FAISS.load_local(
        index_name="my_faiss", 
        folder_path=folder_path, 
        embeddings=bedrock_embeddings,
        allow_dangerous_deserialization=True
        )
    st.write("Agent ready!")
    st.write("")
    #create retriever
    question = st.text_input("What would you like to ask our agent? ")
    if st.button("Ask Question... "):
        with st.spinner("Generating response..."):
            llm = get_llm()

            #fetching the response
            response = get_response(llm, faiss_index, question)
            #st.write("========================================================")
            st.write(response)
            st.success("Done! Would you like to ask another question?")



if __name__ == "__main__":
    main()


