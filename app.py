import streamlit as st
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import PyPDF2
import os
from dotenv import load_dotenv

"""
PDF Chatbot - Ask questions about your PDF documents
=======================================================
This application uses LangChain and Groq API to enable semantic search
and question-answering over PDF documents using RAG (Retrieval-Augmented Generation).

Features:
- Upload and analyze PDF files
- Ask natural language questions
- Get intelligent responses based on document content
- Maintain conversation history
"""

# Load environment variables from .env file
load_dotenv()

# Configure Streamlit page
st.set_page_config(page_title="PDF & CV Chatbot", page_icon="🤖")
st.title("🤖 AI Chatbot — Ask your PDF & CV")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Load Groq API key from environment (.env file) or user input
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# File uploader widget for PDF documents
uploaded_file = st.sidebar.file_uploader("Upload your PDF or CV", type="pdf")

# ============================================================================
# FUNCTIONS
# ============================================================================

def extract_text(pdf_file):
    """
    Extract text from a PDF file.
    
    Args:
        pdf_file: Streamlit uploaded file object (PDF format)
        
    Returns:
        str: Concatenated text content from all pages in the PDF
        
    Note:
        Handles empty pages gracefully by checking if extraction returns None
    """
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted
    return text

def build_chain(text, api_key):
    """
    Build a RAG (Retrieval-Augmented Generation) chain for document Q&A.
    
    This function:
    1. Splits the document into manageable chunks
    2. Creates embeddings for semantic search
    3. Builds a vector store (FAISS) for efficient retrieval
    4. Creates a chain combining retrieval + LLM for generating responses
    
    Args:
        text (str): Raw document text extracted from PDF
        api_key (str): Groq API key for LLM access
        
    Returns:
        tuple: (chain, retriever)
            - chain: LangChain Runnable that processes questions
            - retriever: FAISS retriever for document chunks
    
    Architecture:
        Text → Split into chunks → Embed → FAISS index
                                            ↓
        Question → Retrieve context → Format → Prompt → LLM → Answer
    """
    # Split document into chunks with overlap for better context
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    
    # Convert chunks to Document objects (required by FAISS)
    documents = [Document(page_content=chunk) for chunk in chunks]
    
    # Initialize embeddings model (all-MiniLM-L6-v2: lightweight, efficient)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Build FAISS vector store for semantic search
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # Initialize Groq LLM client
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")
    
    # Define system prompt that guides LLM behavior
    system_prompt = """You are a precise and professional AI assistant specialized in analyzing documents, CVs, and PDFs.

Rules:
- Answer directly, naturally and concisely — no small talk
- Extract only what is relevant to the question
- If information is not in the document, say clearly: "This information is not in the document."
- Structure answers with bullet points when listing multiple items
- Never guess or hallucinate — stick strictly to the document content

Context from document:
{context}

Question: {question}"""
    
    # Create prompt template with placeholders for context and question
    prompt = ChatPromptTemplate.from_template(system_prompt)
    
    # Create retriever with k=4 (retrieve top 4 most relevant chunks)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # Helper function to format retrieved documents into a string
    def format_docs(docs):
        """Join document chunks with separators for readability."""
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Build LangChain Expression Language (LCEL) chain
    # This defines the flow: question → retrieve context → format → prompt → LLM
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    
    return chain, retriever

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
# Streamlit session_state persists variables across page reruns
# This allows us to maintain conversation history and avoid rebuilding the chain

if "messages" not in st.session_state:
    st.session_state.messages = []  # Store chat history

if "chain" not in st.session_state:
    st.session_state.chain = None   # RAG chain instance
    
if "retriever" not in st.session_state:
    st.session_state.retriever = None  # FAISS retriever instance

if "current_file" not in st.session_state:
    st.session_state.current_file = None  # Track which file is loaded

# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================
# Build the RAG chain when a file is uploaded
# Only rebuild if a different file is uploaded (not on every rerun)

if uploaded_file and GROQ_API_KEY:
    # Check if this is a new file or same file already processed
    if st.session_state.chain is None or st.session_state.current_file != uploaded_file.name:
        with st.spinner("Reading your document..."):
            text = extract_text(uploaded_file)
            chain, retriever = build_chain(text, GROQ_API_KEY)
            st.session_state.chain = chain
            st.session_state.retriever = retriever
            st.session_state.current_file = uploaded_file.name
            st.session_state.messages = []  # Reset chat when a new file is uploaded
        st.success("Document ready! Ask me anything.")

# ============================================================================
# UI: CHAT HISTORY & INPUT
# ============================================================================

# Display previous messages from session state
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input widget - processes when user submits
if prompt := st.chat_input("Ask a question about your document..."):
    # Validation checks
    if not GROQ_API_KEY:
        st.warning("Please enter your Groq API key in the sidebar.")
    elif not uploaded_file:
        st.warning("Please upload a PDF first.")
    else:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Invoke RAG chain with user question
                    response = st.session_state.chain.invoke(prompt)
                    # Extract answer from LLM response object
                    answer = response.content if hasattr(response, 'content') else str(response)
                    st.write(answer)
                    # Store assistant response in history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error: {str(e)}")