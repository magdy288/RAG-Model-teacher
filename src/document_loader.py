# Imports
from  langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_document(pdf):
    # Load a PDF
    """
    Load a PDF and split it into chunks for efficient retrieval.

    :param pdf: PDF file to load
    :return: List of chunks of text
    """

    loader = PyPDFLoader(pdf)
    docs = loader.load()

    # Instantiate Text Splitter with Chunk Size of 500 words and Overlap of 100 words so that context is not lost
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    
    # Split into chunks for efficient retrieval
    chunks = text_splitter.split_documents(docs)

    # Return
    return chunks