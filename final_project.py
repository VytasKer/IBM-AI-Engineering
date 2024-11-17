from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from ibm_watsonx_ai import Credentials
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.chains import RetrievalQA

import gradio as gr

# You can use this section to suppress warnings generated by your code:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

## LLM
def get_llm():
    model_id = 'mistralai/mixtral-8x7b-instruct-v01'
    parameters = {
        GenParams.MAX_NEW_TOKENS: 256,
        GenParams.TEMPERATURE: 0.5,
    }
    project_id = "skills-network"
    watsonx_llm = WatsonxLLM(
        model_id=model_id,
        url="https://us-south.ml.cloud.ibm.com",
        project_id=project_id,
        params=parameters,
    )
    return watsonx_llm

## Document loader
def document_loader(file):
    if file.name.endswith('.pdf'):
        # Load PDF files
        loader = PyPDFLoader(file.name)
        loaded_document = loader.load()
    elif file.name.endswith('.txt'):
        # Load TXT files
        with open(file.name, 'r', encoding='utf-8') as f:
            content = f.read()
        # Wrap content in LangChain's Document schema
        loaded_document = [Document(page_content=content)]
    else:
        raise ValueError("Unsupported file format. Please upload a .pdf or .txt file.")
    
    return loaded_document

## Text splitter
def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_documents(data)
    return chunks

## Vector db
def vector_database(chunks):
    embedding_model = watsonx_embedding()
    vectordb = Chroma.from_documents(chunks, embedding_model)
    
    # Perform similarity search
    query = "Smoking policy"
    top_results = vectordb.similarity_search(query, k=5)
    for i, result in enumerate(top_results):
        print(f"Result {i+1}:\n{result.page_content}\n{'-'*50}")
    return vectordb

## Embedding model
def watsonx_embedding():
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
    }
    watsonx_embedding = WatsonxEmbeddings(
        model_id="ibm/slate-125m-english-rtrvr",
        url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",
        params=embed_params,
    )
    # Embed the query
    query = "How are you?"
    embedding_result = watsonx_embedding.embed_query(text=query)
    print("First 5 embedding numbers:", embedding_result[:5])  # Display first 5 numbers
    return watsonx_embedding

## Retriever
def retriever(file):
    # Load and split the document
    splits = document_loader(file)
    chunks = text_splitter(splits)
    
    # Create vector database
    embedding_model = watsonx_embedding()
    vectordb = Chroma.from_documents(chunks, embedding_model)
    
    # Perform similarity search
    query = "Email policy"
    top_results = vectordb.similarity_search(query, k=2)
    
    # Print results for verification
    for i, result in enumerate(top_results):
        print(f"Result {i+1}:\n{result.page_content}\n{'-'*50}")
    
    return vectordb

## QA Chain
def retriever_qa(file, query):
    llm = get_llm()
    retriever_obj = retriever(file)
    qa = RetrievalQA.from_chain_type(llm=llm, 
                                    chain_type="stuff", 
                                    retriever=retriever_obj, 
                                    return_source_documents=False)
    response = qa.invoke(query)
    return response['result']

# Create Gradio interface
rag_application = gr.Interface(
    fn=retriever_qa,
    allow_flagging="never",
    inputs=[
        gr.File(
            label="Upload Document (PDF or TXT)",
            file_count="single",
            file_types=['.pdf', '.txt'],  # Allow both .pdf and .txt files
            type="filepath"
        ),
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    ],
    outputs=gr.Textbox(label="Output"),
    title="RAG Chatbot",
    description="Upload a PDF or TXT document and ask any question. The chatbot will try to answer using the provided document."
)

# Launch the app
rag_application.launch(server_name="0.0.0.0", server_port= 7861)