import streamlit as st
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import PyPDF2

st.set_page_config(page_title="PDF & CV Chatbot", page_icon="🤖")

# Description avant le titre
st.markdown("""
This app uses **LangChain** and **Groq API** to enable semantic search and question-answering over PDF documents using **RAG** (Retrieval-Augmented Generation).

**Features:**
- 📤 Upload and analyze any PDF or CV
- ❓ Ask natural language questions
- 💡 Get precise answers based on document content
- 🎯 Professional analysis and evaluation on demand
- 💬 Conversation history maintained
""")

st.title("🤖 AI Chatbot — Ask your PDF & CV")

# Sidebar
st.sidebar.markdown("### 📄 Upload your PDF or CV")
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
uploaded_file = st.sidebar.file_uploader("Upload your PDF or CV", type="pdf")

st.sidebar.markdown("---")
st.sidebar.markdown("### 👨‍💻 Built by")
st.sidebar.markdown("**Powell FAGNON**")
st.sidebar.markdown("AI Integration Developer")
st.sidebar.markdown("[LinkedIn](https://linkedin.com/in/powell-fagnon-5a826839a) | [Email](mailto:powellfagnon06@gmail.com)")

# ============================================================================
# FUNCTIONS
# ============================================================================

def extract_text(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted
    return text

def build_chain(text, api_key):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")
    
    system_prompt = """You are a professional AI assistant analyzing documents and CVs.

- Answer directly and concisely
- Analyze, evaluate, and give opinions based on document content
- If asked "what do you think?" give professional assessment
- If info not in document: say "This information is not in the document"
- Never invent facts but CAN interpret what's written

Context:
{context}

Question: {question}"""
    
    prompt = ChatPromptTemplate.from_template(system_prompt)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    
    return chain, retriever

# ============================================================================
# SESSION STATE
# ============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None

# ============================================================================
# MAIN LOGIC
# ============================================================================

if uploaded_file and GROQ_API_KEY:
    if st.session_state.chain is None or st.session_state.current_file != uploaded_file.name:
        with st.spinner("Reading your document..."):
            text = extract_text(uploaded_file)
            chain, retriever = build_chain(text, GROQ_API_KEY)
            st.session_state.chain = chain
            st.session_state.retriever = retriever
            st.session_state.current_file = uploaded_file.name
            st.session_state.messages = []
        st.success("Document ready! Ask me anything.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask a question about your document..."):
    if not uploaded_file:
        st.warning("Please upload a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chain.invoke(prompt)
                    answer = response.content if hasattr(response, 'content') else str(response)
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error: {str(e)}")