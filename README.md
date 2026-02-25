# PDF Chatbot 🤖

An intelligent PDF document Q&A application using **RAG (Retrieval-Augmented Generation)** with LangChain and Groq API.

## Features

✨ **Smart Document Analysis**
- Upload and analyze PDF files (CVs, reports, documents, etc.)
- Ask natural language questions about your documents
- Get accurate, context-aware responses

🔍 **Advanced Retrieval**
- Semantic search using embeddings
- FAISS vector store for efficient similarity search
- Multi-document chunk retrieval for comprehensive context

🚀 **Fast Performance**
- Groq API for lightning-fast LLM inference
- HuggingFace embeddings for semantic understanding
- Real-time streaming responses

💬 **Conversation History**
- Maintains chat history within a session
- Automatic reset when uploading a new document
- File-aware conversation management

## Architecture

```
PDF Upload
    ↓
[Text Extraction] → PyPDF2
    ↓
[Chunking] → RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
    ↓
[Embeddings] → HuggingFace "all-MiniLM-L6-v2"
    ↓
[Vector Store] → FAISS (4 top results per query)
    ↓
[RAG Chain] → Retriever | Prompt Template | Groq LLM
    ↓
Answer
```

## Installation

### Prerequisites
- Python 3.10+
- Groq API key (get free at https://groq.com)

### Setup

```bash
# 1. Create virtual environment
python -m venv ml_env
source ml_env/bin/activate  # On Windows: ml_env\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env file
echo "api_key=your_groq_api_key_here" > .env
```

## Usage

```bash
# Activate environment
source ml_env/bin/activate

# Run the application
streamlit run app.py
```

The app will open at `http://localhost:8501`

### How to Use
1. **Enter API Key** (optional if in .env)
2. **Upload PDF** via the sidebar
3. **Ask Questions** about the document content
4. **Get Instant Answers** based on document context

## Configuration

### environment Variables (.env)
```
api_key=your_groq_api_key_here
```

### Model Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Chunk Size | 1000 | Characters per text chunk |
| Chunk Overlap | 200 | Overlap between chunks for context continuity |
| Retrieval K | 4 | Number of chunks retrieved per query |
| Embedding Model | all-MiniLM-L6-v2 | 384-dim, lightweight embeddings |
| LLM | llama-3.3-70b | Groq's fastest available model |

## Project Structure

```
pdf_chatbot/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env               # API keys (git-ignored)
├── ml_env/           # Virtual environment
└── README.md         # This file
```

## Dependencies

```
streamlit              # Web UI framework
langchain             # LLM orchestration
langchain-core        # Core abstractions
langchain-groq        # Groq API integration
langchain-text-splitters  # Document splitting
langchain-community   # Community integrations
sentence-transformers # Semantic embeddings
faiss-cpu            # Vector similarity search
PyPDF2               # PDF text extraction
python-dotenv        # Environment variable loading
```

## Key Components

### `extract_text(pdf_file)`
Extracts text from PDF using PyPDF2.

### `build_chain(text, api_key)`
Creates RAG chain:
1. Splits document into chunks
2. Generates embeddings
3. Builds FAISS vector store
4. Creates LCEL chain combining retrieval + LLM

### Session State Management
- `messages`: Chat history
- `chain`: RAG pipeline instance
- `retriever`: FAISS retriever
- `current_file`: Tracks loaded file to avoid reprocessing

## Performance Tips

⚡ **Optimization**
- Uses CPU FAISS (install `faiss-gpu` for GPU acceleration)
- HuggingFace embeddings are cached locally
- Groq API provides sub-second inference

🎯 **Best Practices**
- Keep chunk size between 500-2000 for balance
- Use k=3-5 for retrieval to balance relevance/scope
- Small PDFs (<500KB) work best for instant processing

## Troubleshooting

### "No module named 'langchain.chains'"
Your installed LangChain version is incompatible. Update:
```bash
pip install --upgrade langchain langchain-core langchain-groq
```

### "Input to ChatPromptTemplate is missing variables"
The prompt template expects specific variables. Ensure `build_chain()` passes correct keys.

### Slow responses
- Reduce chunk_size to speed up embedding
- Reduce k (retrieval count) from 4 to 2
- Use `faiss-gpu` for GPU acceleration

### Empty PDF extraction
Some PDFs are image-only. They need OCR (not supported in this version).

## Advanced Customization

### Change LLM Model
In `build_chain()`:
```python
llm = ChatGroq(groq_api_key=api_key, model_name="mixtral-8x7b-32768")
```

### Adjust Chunk Strategy
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,  # Larger chunks
    chunk_overlap=500  # More overlap
)
```

### Custom System Prompt
Modify the `system_prompt` variable in `build_chain()` to change LLM behavior.

## Limitations

⚠️ **Known Limitations**
- Cannot process image-based PDFs (OCR not implemented)
- Limited to 500MB+ file sizes depending on memory
- No support for multi-language documents (English optimized)
- Chat history resets on application restart

## Future Enhancements

📚 **Planned Features**
- [ ] Multi-document Q&A
- [ ] Persistent chat history (database)
- [ ] OCR for image-based PDFs
- [ ] Citation tracking (show source pages)
- [ ] Custom prompt templates UI
- [ ] Batch document processing
- [ ] Export chat to PDF

## License

MIT License - Feel free to modify and share!

## Support

- **Groq API**: https://groq.com/documentation
- **LangChain**: https://python.langchain.com/
- **Streamlit**: https://docs.streamlit.io/

---

**Created with ❤️ using LangChain, Streamlit, and Groq API**
