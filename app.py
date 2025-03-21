import streamlit as st
import os
from langchain.chains.llm import LLMChain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
import numpy as np

llm = OllamaLLM(base_url="http://localhost:11434", model="llama3.2:3b")


class CustomEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model
        
    def embed_documents(self, texts):
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, text):
        embedding = self.model.encode([text])[0]
        return embedding.tolist()
    
    
    def __call__(self, text):
        if isinstance(text, str):
            return self.embed_query(text)
        elif isinstance(text, list):
            return self.embed_documents(text)
        else:
            raise ValueError('Input must be a string or list of strings')
        
        
        
if 'embeddings_model' not in st.session_state:
    model = SentenceTransformer('all-MiniLM-L12-v2')
    st.session_state.embeddings_model = CustomEmbeddings(model)
    
if 'vectors' not in st.session_state:
    st.session_state.vectors = None
    
if 'loader' not in st.session_state:
    st.session_state.loader = None
    
if 'docs' not in st.session_state:
    st.session_state.docs = None
    
if 'text_splitter' not in st.session_state:
    st.session_state.text_splitter = None
    
if 'final_documents' not in st.session_state:
    st.session_state.final_documents = None
    
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    
    
    
def vector_embedding(uploaded_file=None):
    if uploaded_file:
        with open('pdfs', 'wb') as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.loader = PyPDFDirectoryLoader('pdfs')
    else:
        st.session_state.loader = PyPDFDirectoryLoader('pdfs')
        
    
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 70
    )
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
    
    st.session_state.vectors = FAISS.from_documents(
        st.session_state.final_documents,
        st.session_state.embeddings_model
    )
    
    
def summarize_document():
    if 'final_documents' not in st.session_state or not st.session_state.final_documents:
        return 'Please upload a document first.'
    
    try:
        full_text = ' '.join([doc.page_content for doc in st.session_state.final_documents])
        
        summary_prompt = ChatPromptTemplate.format_prompt(
            '''
            Please provide a concise summary of the following document and understand the document carefully:
            {text}
            
            Summary:
            '''
        )
        summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
        summary = summary_chain.run(text=full_text[:4000])
        return summary
    
    except Exception as e:
        return f'Error: {str(e)}'
    
    
    
st.title('chat with pdf')

st.sidebar.header('Upload PDF')
uploaded_file = st.sidebar.file_uploader('Choose a PDF', type=['pdf'])

if uploaded_file and st.sidebar.button('Upload & Process'):
    vector_embedding(uploaded_file)
    st.sidebar.success('Document uploaded and processed')
    
if st.sidebar.button('Summarize the Document'):
    with st.spinner('Generating summary....'):
        summary = summarize_document()
        st.write('### Document Summary')
        st.write(summary)
        

prompt1 = st.text_input('Ask a question from the document')

if st.button('ASK'):
    if prompt1:
        if 'vectors' not in st.session_state or not st.session_state.vectors:
            vector_embedding()
        
        prompt = ChatPromptTemplate.from_template(
            '''
            You are a teacher and good assistant who can help human to fully
            understand about provided document and give a perfect feed back about the context. \
                
            Answer the questions based on the provided context only.
            Please provide the most accurate response based on the question:
            <context>
            {context}
            </context>
            Question: {input}
            '''
        )
        
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriver = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriver, document_chain)
        
        response = retrieval_chain.invoke({'input': prompt1})
        answer = response['answer']
        st.session_state.chat_history.append({'question': prompt1,
                                              'answer': answer})
        
        st.write('### Answer')
        st.write(answer)
        
        with st.expander('Chat History'):
            for i, chat in enumerate(st.session_state.chat_history):
                st.write(f'**Q{i+1}:** {chat['question']}')
                st.write(f'**A{i+1}:** {chat['answer']}')
                st.write('---------------------------------------------')