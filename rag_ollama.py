import streamlit as st
import PyPDF2
import requests
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List

# Set page config
st.set_page_config(
    page_title="RAG with Ollama (Local AI)",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "vector_index" not in st.session_state:
    st.session_state.vector_index = None
if "text_chunks" not in st.session_state:
    st.session_state.text_chunks = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "ollama_status" not in st.session_state:
    st.session_state.ollama_status = None

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    .answer-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f0f4ff;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🤖 RAG PDF Assistant with Local AI (Ollama)</h1>', unsafe_allow_html=True)
st.markdown("Upload PDFs and ask questions using **FREE local AI models** - No API keys needed!")

# Function to check Ollama status
def check_ollama() -> dict:
    """Check if Ollama is running and get available models"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            return {"running": True, "models": models}
        return {"running": False, "models": []}
    except:
        return {"running": False, "models": []}

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Check Ollama status
    if st.button("🔄 Check Ollama Status"):
        st.session_state.ollama_status = check_ollama()
    
    if st.session_state.ollama_status is None:
        st.session_state.ollama_status = check_ollama()
    
    status = st.session_state.ollama_status
    
    if status["running"]:
        st.success("✅ Ollama is running!")
        
        if status["models"]:
            st.info(f"📦 Installed models: {len(status['models'])}")
            
            # Model selection
            model_option = st.selectbox(
                "Choose Local AI Model",
                status["models"],
                index=0 if status["models"] else None
            )
        else:
            st.warning("⚠️ No models installed yet!")
            model_option = None
    else:
        st.error("❌ Ollama is not running!")
        model_option = None
        st.markdown("""
        ### 🚀 Quick Setup:
        1. Download Ollama from [ollama.ai](https://ollama.ai)
        2. Install it (takes 2 minutes)
        3. Open Command Prompt and run:
           ```
           ollama pull llama2
           ```
        4. Wait for download (4GB)
        5. Click "Check Ollama Status" above
        """)
    
    st.markdown("---")
    
    # Chunk size
    chunk_size = st.slider("Chunk Size (characters)", 500, 2000, 1000, 100)
    
    # Number of chunks to retrieve
    top_k = st.slider("Chunks to Retrieve", 1, 10, 3)
    
    st.markdown("---")
    st.markdown("### 💡 Recommended Models:")
    st.markdown("""
    - **llama2** (4GB) - Good balance
    - **mistral** (4GB) - Fast & smart
    - **llama3** (4.7GB) - Best quality
    - **phi** (2GB) - Smallest, fastest
    
    Download any model:
    ```
    ollama pull llama2
    ```
    """)
    
    st.markdown("---")
    st.markdown("Made with ❤️ using Ollama (100% Free)")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from uploaded PDF"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

# Function to split text into chunks
def split_text_into_chunks(text: str, chunk_size: int = 1000) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to end at sentence boundary
        sentence_end = max(chunk.rfind('.'), chunk.rfind('!'), chunk.rfind('?'))
        if sentence_end > chunk_size * 0.5:
            chunk = chunk[:sentence_end + 1]
            end = start + len(chunk)
        
        chunks.append(chunk.strip())
        start = end - 200
    
    return [chunk for chunk in chunks if chunk.strip()]

# Function to create embeddings
@st.cache_resource
def load_embedding_model():
    """Load and cache the embedding model"""
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(texts: List[str]) -> np.ndarray:
    """Get embeddings using local model"""
    model = load_embedding_model()
    with st.spinner("Creating embeddings..."):
        embeddings = model.encode(texts, show_progress_bar=True)
    return np.array(embeddings)

# Function to create vector index
def create_vector_index(chunks: List[str]):
    """Create FAISS vector index from text chunks"""
    if not chunks:
        return None, None
    
    # Get embeddings
    embeddings = get_embeddings(chunks)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    return index, embeddings

# Function to query Ollama
def query_ollama(prompt: str, model: str) -> str:
    """Query local Ollama model"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 500
                }
            },
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "No response generated.")
        else:
            return f"❌ Error: {response.status_code} - {response.text}"
            
    except requests.Timeout:
        return "⏳ Request timed out. The model might be slow on your computer."
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Function to search for relevant chunks
def search_relevant_chunks(question: str, index, chunks: List[str], embeddings: np.ndarray, top_k: int = 3) -> List[str]:
    """Search for relevant text chunks using embeddings"""
    if index is None or not chunks:
        return []
    
    # Create embedding for question
    model = load_embedding_model()
    question_embedding = model.encode([question])
    
    # Search in index
    distances, indices = index.search(question_embedding, top_k)
    
    # Get relevant chunks
    relevant_chunks = []
    for idx in indices[0]:
        if idx < len(chunks):
            relevant_chunks.append(chunks[idx])
    
    return relevant_chunks

# Main app layout
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload PDF")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a PDF document to analyze"
    )
    
    if uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            # Extract text
            text = extract_text_from_pdf(uploaded_file)
            
            if text:
                st.markdown(f'<div class="success-box">✅ Extracted {len(text):,} characters</div>', unsafe_allow_html=True)
                
                # Create chunks
                chunks = split_text_into_chunks(text, chunk_size)
                
                # Create vector index
                index, embeddings = create_vector_index(chunks)
                
                if index is not None:
                    # Store in session state
                    st.session_state.vector_index = index
                    st.session_state.text_chunks = chunks
                    st.session_state.embeddings = embeddings
                    
                    st.markdown(f'<div class="success-box">✅ Created {len(chunks)} searchable chunks</div>', unsafe_allow_html=True)
                    
                    # Show sample chunks
                    with st.expander("🔍 View Sample Chunks", expanded=False):
                        for i, chunk in enumerate(chunks[:3]):
                            st.text_area(f"Chunk {i+1}", chunk[:300] + "..." if len(chunk) > 300 else chunk, height=100, key=f"chunk_{i}")

with col2:
    st.header("💬 Ask Questions")
    
    # Check if Ollama is ready
    if not status["running"]:
        st.markdown('<div class="warning-box">⚠️ Please install and start Ollama first (see left sidebar)</div>', unsafe_allow_html=True)
    elif not status["models"]:
        st.markdown('<div class="warning-box">⚠️ Please download a model first (see instructions in sidebar)</div>', unsafe_allow_html=True)
    elif st.session_state.vector_index is None:
        st.markdown('<div class="info-box">👈 First upload a PDF to get started!</div>', unsafe_allow_html=True)
    else:
        # Question input
        question = st.text_area(
            "Ask a question about the PDF:",
            placeholder="e.g., What is the main topic? What are the key findings?",
            height=100
        )
        
        if st.button("Get Answer", type="primary") and question and model_option:
            with st.spinner("Searching and generating answer (may take 10-30 seconds)..."):
                # Search for relevant chunks
                relevant_chunks = search_relevant_chunks(
                    question,
                    st.session_state.vector_index,
                    st.session_state.text_chunks,
                    st.session_state.embeddings,
                    top_k
                )
                
                if relevant_chunks:
                    # Combine chunks into context
                    context = "\n\n".join(relevant_chunks)
                    
                    # Create prompt
                    prompt = f"""You are a helpful assistant. Answer the question based ONLY on the context provided below.

Context:
{context[:2000]}

Question: {question}

Instructions:
- Answer using ONLY information from the context above
- Be concise and accurate
- If the context doesn't contain the answer, say "The provided text doesn't mention this"

Answer:"""
                    
                    # Get answer from Ollama
                    answer = query_ollama(prompt, model_option)
                    
                    # Display answer
                    st.markdown("### 💡 Answer")
                    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
                    
                    # Show sources
                    with st.expander("📚 View Sources Used", expanded=False):
                        for i, chunk in enumerate(relevant_chunks):
                            st.markdown(f"**Source {i+1}:**")
                            st.text(chunk[:500] + "..." if len(chunk) > 500 else chunk)
                else:
                    st.error("No relevant text found in the PDF")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Powered by 🤖 Ollama • 100% FREE • Runs Locally • No Internet Required</p>
    <p>Models run on your computer - first response may take 10-30 seconds</p>
</div>
""", unsafe_allow_html=True)