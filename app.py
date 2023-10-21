from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["api_key"] # ENTER HUGGING FACE API CODE

# function to parse PDFs
@st.cache_resource
def parse_pdf(file):
    '''
        Loading the pdf file.
    '''
    loader = UnstructuredPDFLoader(file)

    documents = loader.load()
    
    return documents

# We can't fit the whole document inside the prompt so split the document into smaller chunks
@st.cache_resource
def embed_text(_documents):
    '''
        Storing the document chunks into vector store.
    '''
    split = CharacterTextSplitter(separator= '\n\nQ:',chunk_size=100, chunk_overlap=10)

    split_text = split.split_documents(_documents)
    
    embeddings = HuggingFaceEmbeddings()
    
    faiss = FAISS.from_documents(split_text, embeddings)
    return faiss

@st.cache_data
def create_buffer_memory():
    '''
        Create conversation buffer window memory which save the last *k=2* chats.
    '''
    
    memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)
    
    return memory

def create_llm():
    ''' 
        Get LLM from HuggingFaceHub, Use model: "google/flan-t5-xxl"
    '''
    # Getting LLM from HuggingFaceHub
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":1, "max_length":1000})
    
    return llm

def get_answer(llm, retriever, memory, query):
    '''
        Create a conversational chain and get answer
    '''
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                           retriever=faiss.as_retriever(),
                                                           memory=memory,
                                                           verbose=True,)  
    
    answer = conversation_chain.run(query)
    return answer

st.header("FAQ QnA")
uploaded_file = "us_en_faqs.pdf"

if uploaded_file is not None:
    document = parse_pdf(uploaded_file)
    faiss = embed_text(document)
    retriever = faiss.as_retriever()
    memory = create_buffer_memory()
    llm = create_llm()
    query = st.text_area("Ask Aramex FAQ Bot")
    button = st.button("Submit")
    if button:
        st.write(get_answer(llm, retriever, memory, query))
