from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.chains.retrieval import create_retrieval_chain
from langchain_ollama import ChatOllama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.document_loader import load_document
import streamlit as st




# Create a Streamlit app
st.title("AI-Powered Document Q&A")

# Load document to streamlit
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# If a file is uploaded, create the TextSplitter and vector database
if uploaded_file :

    # Code to work around document loader from Streamlit and make it readable by langchain
    temp_file = "./temp.pdf"
    with open(temp_file, "wb") as file:
        file.write(uploaded_file.getvalue())
        file_name = uploaded_file.name

    # Load document and split it into chunks for efficient retrieval.
    chunks = load_document(temp_file)

    # Message user that document is being processed with time emoji
    st.write("Processing document... :watch:")
    
# Generate embeddings
    # Embeddings are numerical vector representations of data, typically used to capture relationships, similarities,
    # and meanings in a way that machines can understand. They are widely used in Natural Language Processing (NLP),
    # recommender systems, and search engines.
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

    # Can also use HuggingFaceEmbeddings
    # from langchain_huggingface.embeddings import HuggingFaceEmbeddings
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create vector database containing chunks and embeddings
    vector_db = FAISS.from_documents(chunks, embeddings)
    
    
# Create a document retriever
    retriever = vector_db.as_retriever()
    llm = ChatOllama(model="llama3.2:3b")
    
    
# Create a system prompt
    # It sets the overall context for the model.
    # It influences tone, style, and focus before user interaction starts.
    # Unlike user inputs, a system prompt is not visible to the end user.

    system_prompt = (
        "You are a helpful teacher and assistant. Use the given context to answer the question."
        "If you don't know the answer, say you don't know. "
        "{context}"
    )

    # Create a prompt Template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Create a chain
    # It creates a StuffDocumentsChain, which takes multiple documents (text data) and "stuffs" them together before passing them to the LLM for processing.

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    
# Creates the RAG
    chain = create_retrieval_chain(retriever, question_answer_chain)
    
# Streamlit input for question
    question = st.text_input("Ask a question about the document:")
    if question:
        # Answer
        response = chain.invoke({"input": question})['answer']
        st.write(response)