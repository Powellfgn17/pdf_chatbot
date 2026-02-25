# Architecture Documentation

## System Overview

This PDF Chatbot implements a **Retrieval-Augmented Generation (RAG)** system that combines document retrieval with language model generation.

```
┌─────────────────────────────────────────────────────────────────┐
│                      PDF CHATBOT SYSTEM                          │
└─────────────────────────────────────────────────────────────────┘

                            USER INTERFACE
                         (Streamlit Web App)
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
              Upload PDF    Enter API Key   Ask Question
                    │             │             │
                    └─────────────┴─────────────┘
                            │
                    ┌───────▼────────┐
                    │ PDF Processing │
                    └────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
   [Extract Text]   [Split Text]        [Create Docs]
   (PyPDF2)         (250-2000 chars)     (LangChain)
        │                   │                   │
        └───────────────────┴───────────────────┘
                            │
                    ┌───────▼────────┐
                    │ Embedding      │
                    │ Generation     │
                    │(HuggingFace)   │
                    └────────────────┘
                            │
                    ┌───────▼────────┐
                    │ FAISS Vector   │
                    │ Store Index    │
                    └────────────────┘
        
┌──────────────────────────────────────────────────────────────────┐
│                        RETRIEVAL PHASE                           │
└──────────────────────────────────────────────────────────────────┘

        User Question
            │
            ▼
    [Query Embedding]
    (Same HF Model)
            │
            ▼
    [FAISS Similarity]
    (Find k=4 closest)
            │
            ▼
    [Retrieved Chunks]
    (Context docs)

┌──────────────────────────────────────────────────────────────────┐
│                      GENERATION PHASE                            │
└──────────────────────────────────────────────────────────────────┘

    Retrieved Context  +  Question
            │                 │
            └────────┬────────┘
                     ▼
         [Format into Prompt]
                     │
                     ▼
       [System Prompt Template]
                     │
                     ▼
        [Groq LLM - Llama 3.3]
                     │
                     ▼
            [Generated Answer]
                     │
                     ▼
       [Display to User + Save]
```

## Data Flow

### 1. Document Preparation (One-time)

```python
PDF File
  └─→ PyPDF2.PdfReader()
       └─→ Extract text from all pages
           └─→ RecursiveCharacterTextSplitter()
               └─→ Create overlapping chunks (1000 chars, 200 overlap)
                   └─→ Document() objects (LangChain)
                       └─→ HuggingFaceEmbeddings.embed_documents()
                           └─→ 384-dimensional vectors
                               └─→ FAISS.from_documents()
                                   └─→ Indexed Vector Store
```

**Why overlapping chunks?**
- Prevents questions from spanning chunk boundaries
- Ensures context continuity
- Example: If a sentence extends from chunk 2→3, both get retrieved

### 2. Inference (Per question)

```
User Question ("What is the candidate's experience?")
  └─→ LCEL Chain processes:
      1. Question → HF Embeddings → 384-dim vector
      2. FAISS retrieves k=4 most similar chunks
      3. Format chunks: "\n\n".join() → single context string
      4. Create prompt:
         ├─ System instructions
         ├─ Retrieved context
         └─ Question
      5. Groq LLM generates response
      6. Extract .content from response object
      7. Display to user + save to history
```

## Component Details

### Text Processing Pipeline

**RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)**

```
Original Text: "Lorem ipsum dolor sit amet... [very long] ...consectetur adipiscing elit"

Split Logic:
1. Try to split by "\n\n" (paragraphs)
2. If too long, try splitting by "\n" (lines)
3. If still too long, try splitting by " " (words)
4. Finally split by character if necessary

Result:
Chunk 1: [0-1000 chars]
Chunk 2: [800-1800] ← overlaps 200 chars with Chunk 1
Chunk 3: [1600-2600] ← overlaps 200 chars with Chunk 2
...
```

### Embedding Model

**HuggingFace "all-MiniLM-L6-v2"**
- Dimension: 384
- Max token length: 256 words
- Speed: ~50 documents/sec (CPU)
- Size: 22MB
- Similarity metric: Cosine distance

Example:
```
"Senior Developer with 5 years Python experience"
  └─→ Tokenizer
      └─→ [49, 202, 15, 890, ...] (tokens)
          └─→ Embedding Layer
              └─→ [-0.23, 0.45, -0.12, ..., 0.98] (384 values)
```

### Vector Store (FAISS)

**Facebook AI Similarity Search**

```
Indexed Documents:
├─ Doc1: "5 years Python..." → [-0.23, 0.45, ...]
├─ Doc2: "Java specialist..." → [0.12, -0.34, ...]
├─ Doc3: "Python Django..." → [-0.25, 0.43, ...]
└─ Doc4: "C++ embedded..." → [0.98, -0.12, ...]

Query: "Python experience" → [-0.24, 0.44, ...]

Similarity Scores:
├─ Doc1: 0.98 ✓ (most similar)
├─ Doc3: 0.96 ✓
├─ Doc2: 0.62
└─ Doc4: 0.15

Retrieval (k=4): [Doc1, Doc3, Doc2, Doc4]
```

### LLM Integration (Groq)

**Model: Llama 3.3 70B Versatile**
- Inference speed: ~100 tokens/sec
- Context window: 8192 tokens
- Training data: Cut-off unknown (trained on public data)

Prompt format:
```
[System] You are a precise assistant...
[Context] Doc1: "..." \n\n Doc2: "..."
[Question] "What is the experience?"
[Assistant] "The candidate has..."
```

### Session State Management

Streamlit's `session_state` ensures data persists across widget interactions:

```python
# First visit: All initialized
st.session_state.messages = []
st.session_state.chain = None

# User uploads PDF → build_chain() → chain created
st.session_state.chain = RAGChain(...)

# User asks question → chain reused
for _ in range(10_questions):
    response = st.session_state.chain.invoke(question)
    # No rebuild needed!

# Different file uploaded → rebuild triggered
if st.session_state.current_file != new_file.name:
    st.session_state.chain = build_chain(new_text, api_key)
    st.session_state.messages = []  # Clear history
```

## Error Handling

### Common Failure Points

1. **PDF Extraction**
   - Empty pages → check `if extracted:`
   - Image-only PDFs → would fail (OCR not implemented)
   - Corrupted files → PyPDF2 raises exception

2. **Embedding Generation**
   - Network error → offline check needed
   - Token limit exceeded → split long texts
   - OOM on large documents → reduce chunk_size

3. **FAISS Indexing**
   - Empty documents → validates at runtime
   - Memory constraints → should warn user

4. **LLM Generation**
   - API key invalid → 401 error
   - Rate limiting → exponential backoff (user handles)
   - Context too long → truncates gracefully

## Performance Characteristics

### Time Breakdown (typical 50-page PDF)

| Phase | Time | Notes |
|-------|------|-------|
| Extract Text | 0.5s | I/O bound |
| Split into chunks | 0.1s | CPU bound |
| Generate embeddings | 3-5s | Depends on CPU |
| Build FAISS index | 0.5s | Memory bound |
| **Per Question** | | |
| Produce query embedding | 0.05s | Cached model |
| FAISS retrieval | 0.01s | Vec DB lookup |
| LLM inference | 2-4s | Network + compute |
| **Total** | 5-10s | Per question |

### Memory Usage

| Component | Size |
|-----------|------|
| Embedding model | 22 MB |
| FAISS index (50 pages) | ~5-10 MB |
| Chat history (100 msgs) | ~1 MB |
| **Total** | ~30 MB user |

### Scalability Limits

| Metric | Limit | Reason |
|--------|-------|--------|
| PDF Size | ~200 MB | RAM embedding |
| Chunks | ~10K | FAISS perf |
| Users | Single | No backend |
| Questions/session | Unlimited | History grows |

## Optimization Opportunities

### Quick Wins
1. ✅ Use `faiss-gpu` for 10x speedup
2. ✅ Cache embeddings to disk
3. ✅ Lazy-load model on first use
4. ⚠️ Batch questions in notebook

### Future Improvements
- [ ] Multi-user backend (Flask/FastAPI)
- [ ] Database persistence
- [ ] Citation/source tracking
- [ ] Semantic caching
- [ ] Reranking step (ColBERT)

## Security Considerations

⚠️ **Current**: No user authentication
- API keys stored in .env (local only)
- No document encryption
- No access controls

🔒 **For Production**:
- Use secrets manager (AWS Secrets Manager)
- Encrypt PDFs at rest
- Rate limiting per user
- Audit logging
- HTTPS only
