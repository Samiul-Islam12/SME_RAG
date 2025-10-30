"""
PDF-based RAG System with Multiple LLM Support
Supports: Mistral AI, Qwen3, and Llama models
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# PDF and document processing
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

# For OpenAI-compatible APIs (Mistral)
from langchain.chat_models import ChatOpenAI


class PDFRAGSystem:
    """RAG system that works with PDF documents and multiple LLM models"""
    
    def __init__(self, pdf_directory: str, model_name: str = "llama3"):
        """
        Initialize the RAG system
        
        Args:
            pdf_directory: Path to directory containing PDF files
            model_name: Name of the model to use ('mistral', 'qwen3', 'llama3')
        """
        self.pdf_directory = pdf_directory
        self.model_name = model_name
        self.vectorstore = None
        self.qa_chain = None
        self.embeddings = None
        self.documents = []
        
    def load_pdfs(self) -> List[Any]:
        """Load all PDF documents from the specified directory"""
        print(f"Loading PDFs from {self.pdf_directory}...")
        
        # Load PDFs
        loader = DirectoryLoader(
            self.pdf_directory,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        
        documents = loader.load()
        print(f"Loaded {len(documents)} document pages from PDFs")
        
        return documents
    
    def split_documents(self, documents: List[Any]) -> List[Any]:
        """Split documents into chunks for better retrieval"""
        print("Splitting documents into chunks...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} document chunks")
        
        return chunks
    
    def create_vectorstore(self, chunks: List[Any]):
        """Create FAISS vectorstore from document chunks"""
        print("Creating embeddings and vectorstore...")
        
        # Use HuggingFace embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Create FAISS vectorstore
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        print("Vectorstore created successfully")
    
    def initialize_llm(self):
        """Initialize the LLM based on model_name"""
        print(f"Initializing {self.model_name} model...")
        
        if self.model_name.lower() in ['mistral', 'mistralai']:
            # Using Ollama for Mistral
            llm = Ollama(model="mistral", temperature=0.1)
        elif self.model_name.lower() in ['qwen3', 'qwen2.5']:
            # Using Ollama for Qwen
            llm = Ollama(model="qwen2.5", temperature=0.1)
        elif self.model_name.lower() in ['llama3', 'llama']:
            # Using Ollama for Llama
            llm = Ollama(model="llama3", temperature=0.1)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        return llm
    
    def setup_qa_chain(self):
        """Setup the QA chain with retrieval"""
        if self.vectorstore is None:
            raise ValueError("Vectorstore not created. Call create_vectorstore first.")
        
        llm = self.initialize_llm()
        
        # Create retrieval QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
            ),
            return_source_documents=True
        )
        
        print(f"QA chain setup complete with {self.model_name}")
    
    def initialize(self):
        """Initialize the complete RAG system"""
        # Load and process PDFs
        documents = self.load_pdfs()
        self.documents = documents
        
        # Split into chunks
        chunks = self.split_documents(documents)
        
        # Create vectorstore
        self.create_vectorstore(chunks)
        
        # Setup QA chain
        self.setup_qa_chain()
        
        print("RAG system initialization complete!")
    
    def query(self, question: str, record_time: bool = True) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            question: The question to ask
            record_time: Whether to record response time
            
        Returns:
            Dictionary containing answer, sources, and metadata
        """
        if self.qa_chain is None:
            raise ValueError("QA chain not setup. Call initialize first.")
        
        start_time = time.time()
        
        # Get response
        response = self.qa_chain({"query": question})
        
        end_time = time.time()
        response_time = end_time - start_time
        
        result = {
            "question": question,
            "answer": response['result'],
            "source_documents": response['source_documents'],
            "response_time": response_time,
            "model": self.model_name
        }
        
        return result
    
    def save_vectorstore(self, path: str = "vectorstore"):
        """Save the vectorstore to disk"""
        if self.vectorstore:
            self.vectorstore.save_local(path)
            print(f"Vectorstore saved to {path}")
    
    def load_vectorstore(self, path: str = "vectorstore"):
        """Load vectorstore from disk"""
        if self.embeddings is None:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
        
        self.vectorstore = FAISS.load_local(path, self.embeddings)
        print(f"Vectorstore loaded from {path}")


def main():
    """Example usage"""
    # Create sample PDF directory if it doesn't exist
    pdf_dir = "pdfs"
    os.makedirs(pdf_dir, exist_ok=True)
    
    print("=" * 60)
    print("PDF-based RAG System with Multiple LLM Support")
    print("=" * 60)
    print("\nSupported models: mistral, qwen3, llama3")
    print(f"\nPlace your PDF files in the '{pdf_dir}' directory")
    print("\nNote: Make sure Ollama is running with the required models")
    print("      Install models: ollama pull mistral, ollama pull qwen2.5, ollama pull llama3")
    print("=" * 60)
    
    # Example: Initialize system with Llama3
    # rag = PDFRAGSystem(pdf_directory=pdf_dir, model_name="llama3")
    # rag.initialize()
    
    # Example query
    # result = rag.query("What is this document about?")
    # print(f"\nQuestion: {result['question']}")
    # print(f"Answer: {result['answer']}")
    # print(f"Response Time: {result['response_time']:.2f}s")


if __name__ == "__main__":
    main()
