import streamlit as st
import tempfile
import PyPDF2
import chromadb
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="DocuQuery — RAG System", page_icon="📚")
st.title("📚 DocuQuery — Ask Questions About Your Documents")
st.markdown("Upload a PDF and ask questions. Powered by free open-source RAG.")

# Sidebar info
with st.sidebar:
    st.header("📖 About RAG")
    st.markdown("""
    **R**etrieval - Finds relevant text
    
    **A**ugmented - Adds context
    
    **G**eneration - Answers using your data
    
    ---
    🆓 100% Free — No API keys needed
    """)

# Load free embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Upload file
uploaded_file = st.file_uploader("Upload a PDF document:", type="pdf")

if uploaded_file:
    # Extract text from PDF
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    
    pdf_reader = PyPDF2.PdfReader(tmp_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    st.success(f"✅ Extracted {len(text)} characters from PDF")
    
    # Chunk text
    chunks = [chunk.strip() for chunk in text.split('\n\n') if len(chunk.strip()) > 100]
    st.info(f"📦 Split into {len(chunks)} chunks")
    
    # Create embeddings and store
    chroma_client = chromadb.Client()

    # Delete old collection if it exists
    try:
        chroma_client.delete_collection(name="pdf_docs")
    except:
        pass

    # Create new collection
    collection = chroma_client.create_collection(name="pdf_docs")   
    
    with st.spinner("🔢 Creating embeddings (free, no API needed)..."):
        for i, chunk in enumerate(chunks):
            embedding = model.encode(chunk).tolist()
            collection.add(
                embeddings=[embedding],
                documents=[chunk],
                ids=[str(i)]
            )
    
    st.success("✅ Document indexed! Ask your questions below.")
    
    # Query
    question = st.text_input("Ask a question about your document:")
    
    if question:
        with st.spinner("🔍 Searching..."):
            # Get question embedding
            q_embedding = model.encode(question).tolist()
            
            # Retrieve top 3 chunks
            results = collection.query(
                query_embeddings=[q_embedding],
                n_results=3
            )
            
            st.subheader("📄 Retrieved Context (RAG Results)")
            
            for i, doc in enumerate(results['documents'][0]):
                with st.expander(f"Chunk {i+1} (Relevance Score: {results['distances'][0][i]:.2f})"):
                    st.write(doc[:800])
            
            st.markdown("---")
            st.success("✅ RAG pipeline complete! The retrieved chunks above are what would be sent to an LLM for answer generation.")
            st.info("💡 To add AI answer generation, connect any LLM API (OpenAI, Claude, or free Llama).")