# Must be at very beginning of file before any other imports
import sys

# Create a custom module to replace torch._classes
class FakeTorchClasses:
    def __getattr__(self, name):
        # Return empty list for __path__ to prevent Streamlit file watcher errors
        if name == '__path__':
            return []
        # Prevent errors for other attribute access
        return None
    
    # These properties are needed to make the module seem legitimate
    @property
    def __file__(self):
        return None
    
    @property
    def __name__(self):
        return 'torch._classes'

# Schedule our replacement to happen after torch is imported
_original_import = __import__

def _patched_import(name, *args, **kwargs):
    module = _original_import(name, *args, **kwargs)
    
    # If torch is being imported, replace torch._classes immediately
    if name == 'torch' and hasattr(module, '_classes'):
        sys.modules['torch._classes'] = FakeTorchClasses()
    
    return module

# Install our import hook
sys.meta_path = [type('TorchImportFixer', (), {'find_module': lambda *_: None, 'find_spec': lambda *_: None})()] + sys.meta_path
sys.__import__ = sys.modules['builtins'].__import__ = _patched_import

# Disable Streamlit's file watcher to prevent torch._classes errors - must be done BEFORE importing streamlit
import os
os.environ["STREAMLIT_SERVER_WATCH_FILE_SYSTEM"] = "false"

# More aggressive patching to prevent torch._classes errors
import types

# Monkey patch the module system to prevent inspection of torch._classes
# This needs to happen before streamlit is imported
class MockPath:
    def __init__(self):
        self._path = []

# If torch is already imported, patch it
if 'torch' in sys.modules and hasattr(sys.modules['torch'], '_classes'):
    torch_classes = sys.modules['torch']._classes
    if not hasattr(torch_classes, '__path__'):
        torch_classes.__path__ = MockPath()

# Now import streamlit
import streamlit as st

# Even more direct patching of Streamlit's watcher functionality
if hasattr(st, '_is_running_with_streamlit'):
    # Patch the path watcher
    import streamlit.watcher.path_watcher
    streamlit.watcher.path_watcher.disable_watcher()
    
    # If available, modify the local_sources_watcher to avoid torch._classes
    if hasattr(streamlit, 'watcher') and hasattr(streamlit.watcher, 'local_sources_watcher'):
        # Import the correct module first
        import streamlit.watcher.local_sources_watcher as local_sources_watcher
        
        # Create a patched version that skips torch._classes
        original_extract_paths = local_sources_watcher.extract_paths
        
        def patched_extract_paths(module):
            if module.__name__ == 'torch._classes':
                return []
            return original_extract_paths(module)
        
        # Apply the patch
        streamlit.watcher.local_sources_watcher.extract_paths = patched_extract_paths

# Continue with the rest of your imports
import subprocess
import json
import tempfile
import requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from ollama_ocr import OCRProcessor
from PIL import Image
import fitz  # PyMuPDF for PDF handling

# Set page configuration
st.set_page_config(
    page_title="Legal Document Q&A Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Near the top of your file, let's add Ollama configuration
if 'OLLAMA_HOST' not in st.session_state:
    st.session_state.OLLAMA_HOST = "http://localhost:11434"  # Default value

# Near the top of your file, add these to check available models
import requests

def get_available_ollama_models():
    """Get list of available models from Ollama"""
    try:
        response = requests.get(f"{st.session_state.OLLAMA_HOST}/api/tags")
        if response.status_code == 200:
            models = [model["name"] for model in response.json().get("models", [])]
            return models
        return []
    except:
        return []

# Replace your call_ollama function with this one
def call_ollama(prompt, model="llava:34b", max_tokens=512, temperature=0.0):
    """
    Call Ollama API with the provided prompt.
    Tries multiple endpoint URLs until one succeeds.
    Temperature is set to 0.0 to reduce hallucinations.
    """
    # List of candidate endpoints (adjust as needed based on your server configuration)
    endpoints = [
        "http://127.0.0.1:42243/api/generate",  # first try: runner endpoint with /api/generate
        "http://127.0.0.1:42243/generate",        # fallback: runner endpoint without /api
        "http://127.0.0.1:11434/api/generate"       # fallback to main port (if applicable)
    ]
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    for url in endpoints:
        try:
            response = requests.post(url, json=payload, timeout=60)
            if response.status_code == 200:
                response_json = response.json()
                return response_json.get("response", "")
            else:
                st.warning(f"Endpoint {url} returned status: {response.status_code}")
        except Exception as e:
            st.warning(f"Endpoint {url} failed with error: {str(e)}")
    
    error_msg = "Error: All endpoints failed for Ollama API call."
    st.error(error_msg)
    return error_msg


# -----------------------------
# OCR Extraction
# -----------------------------
def extract_text_with_ollama(file_path):
    """
    Uses the ollama-ocr package to extract text from a file.
    For PDFs, attempts direct text extraction first using PyMuPDF (fitz); if that fails or yields no text,
    converts the first page to an image and then performs OCR via Ollama.
    """
    import fitz  # PyMuPDF for PDFs
    from PIL import Image
    import io
    import base64

    # For PDFs, try to extract text directly
    if file_path.lower().endswith('.pdf'):
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            if text.strip():
                return text
            # Otherwise, continue to OCR extraction
        except Exception as e:
            st.warning(f"PDF text extraction failed: {e}")

    # Process as image: if PDF, convert first page to image; otherwise, open directly
    try:
        if file_path.lower().endswith('.pdf'):
            doc = fitz.open(file_path)
            page = doc[0]
            pix = page.get_pixmap()
            img_path = f"{file_path}_page0.png"
            pix.save(img_path)
            img = Image.open(img_path)
        else:
            img = Image.open(file_path)
        
        # Convert image to base64 string
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        # Get available models and filter for vision models, excluding any that end with "latest"
        available_models = get_available_ollama_models()
        vision_models = [m for m in available_models if "vision" in m.lower()]
        vision_models = [m for m in vision_models if not m.endswith("latest")]
        
        # If filtering removed all, fallback to all vision models available
        if not vision_models:
            vision_models = [m for m in available_models if "vision" in m.lower()]
        
        if not vision_models:
            st.warning("No vision models found. Trying with standard models.")
            if available_models:
                vision_models = [available_models[0]]
            else:
                return "No models available in Ollama."
        
        # Try each available vision model
        for model in vision_models:
            st.info(f"Trying to extract text with model: {model}")
            try:
                url = f"{st.session_state.OLLAMA_HOST}/api/generate"
                payload = {
                    "model": model,
                    "prompt": "Extract all text visible in this image. Return ONLY the extracted text, no explanation.",
                    "images": [img_str],
                    "stream": False
                }
                
                response = requests.post(url, json=payload, timeout=90)
                if response.status_code == 200:
                    extracted_text = response.json().get("response", "")
                    if extracted_text.strip():
                        # Clean up temporary image file if created
                        if file_path.lower().endswith('.pdf'):
                            try:
                                os.remove(img_path)
                            except Exception as e:
                                st.warning(f"Could not remove temp image: {e}")
                        return extracted_text
                else:
                    st.warning(f"Model {model} returned status: {response.status_code}")
            except Exception as e:
                st.warning(f"Error with model {model}: {str(e)}")
        
        # Clean up temporary image file if created
        if file_path.lower().endswith('.pdf') and os.path.exists(img_path):
            try:
                os.remove(img_path)
            except Exception as e:
                st.warning(f"Cleanup error: {e}")
                
        return "Could not extract text with any available model."
        
    except Exception as e:
        st.error(f"Error processing {file_path}: {e}")
        return ""

# -----------------------------
# Summarize Document
# -----------------------------
def summarize_text(text):
    """
    Uses Ollama to summarize the legal document text.
    """
    prompt = (
        f"Summarize the following legal document text in a comprehensive way:\n\n"
        f"{text[:3000]}...\n\n"
        "Provide a detailed summary that captures key points, parties involved, obligations, dates, and important clauses."
    )
    response = call_ollama(prompt, max_tokens=512)
    return response

# -----------------------------
# Extract Metadata
# -----------------------------
def extract_metadata(text):
    """
    Uses Ollama to extract detailed metadata from the legal document text.
    """
    prompt = (
        "Extract metadata from the following legal document text as a JSON object. Use the field guidelines below. "
        "Include each field only if it is found in the text.\n\n"
        "==== METADATA EXTRACTION GUIDELINES ====\n"
        "Include the following fields **only if they appear** in the text:\n"
        "- contract_type: The type of legal document (e.g., NDA, Service Agreement, Distribution Agreement, Employment Contract, License Agreement).\n"
        "- parties: The organizations or individuals involved.\n"
        "- effective_date: The date when the agreement takes effect (YYYY-MM-DD).\n"
        "- execution_date: The date when the document was signed.\n"
        "- termination_date: The date when the agreement ends or 'Indefinite'.\n"
        "- jurisdiction: The governing jurisdiction (e.g., France, EU, USA, Romanian Law).\n"
        "- governing_law: The legal framework (e.g., French Law, EU Regulations, Romanian Civil Code, GDPR Compliance).\n"
        "- version: The contract version or amendment indicator (e.g., V1, V2, Final, Draft).\n\n"
        "Additional Metadata (if present):\n"
        "- contract_status: Current status (Active, Expired, Terminated, Under Negotiation).\n"
        "- previous_version_reference: Reference to prior version(s) or version history.\n"
        "- key_obligations: Main responsibilities of the parties.\n"
        "- payment_obligations: Payment terms or financial obligations.\n"
        "- confidentiality_clause: Details of confidentiality or data protection obligations.\n"
        "- dispute_resolution: Mechanisms for resolving disputes (arbitration, litigation, etc.).\n"
        "- force_majeure: Conditions excusing performance (e.g., war, pandemic, government intervention).\n"
        "- exclusivity: Whether one party has exclusive rights.\n"
        "- non_compete: Restrictions on engaging with competitors.\n"
        "- ip_assignment: Ownership rights or licensing.\n"
        "- [Other relevant fields from General Contractual Obligations, Legal Clauses, Contract-Specific Terms, and Additional Business-Specific Labels if mentioned].\n\n"
        "==== ADDITIONAL FIELDS FROM SPECIFIC LABELS ====\n"
        "Also include these fields if they appear in the text:\n"
        "- brand_licensor: Entity granting rights over brand use.\n"
        "- licensee: Entity receiving brand/IP rights.\n"
        "- producers: Companies authorized to manufacture products.\n"
        "- partner_restaurants: Entities authorized to sell branded products.\n"
        "- sub_licensee: Third parties allowed to use IP under a sublicense.\n"
        "- competitor_restriction: Details on non-compete clauses or market exclusivity.\n"
        "- trademark_ownership: Ownership of trademarks and licensing terms.\n"
        "- trademark_registration_status: Registration status of trademarks (e.g., Pending, Approved).\n"
        "- trademark_usage_license: Details on trademark licensing (Exclusive/Non-exclusive, Geographic Scope).\n"
        "- know_how_transfer: Clauses related to transferring know-how.\n"
        "- trade_secrets_protection: Confidentiality terms around proprietary knowledge.\n"
        "- branding_rights: Rights regarding logos, slogans, and product names.\n"
        "- advertising_restrictions: Approval requirements for marketing materials.\n"
        "- royalty_fee_percentage: Percentage payable as a royalty fee.\n"
        "- revenue_share_model: Details on revenue sharing between parties.\n"
        "- late_payment_penalty: Penalties or interest for late payments.\n"
        "- revenue_collection_agent: Entity authorized to collect payments.\n"
        "- obligation_to_perform: Specific performance obligations defined in the contract.\n"
        "- service_standards: Quality control and performance measures.\n"
        "- product_quality_standards: Health and safety compliance details.\n"
        "- compliance_requirements: Legal compliance obligations (e.g., GDPR, Consumer Protection Laws).\n"
        "- audit_rights: Rights to inspect compliance with contract terms.\n"
        "- penalties_for_breach: Consequences for non-performance.\n"
        "- termination_notice_period: Notice period required for termination.\n"
        "- automatic_renewal: Whether the contract renews automatically.\n"
        "- grounds_for_termination: Conditions that allow contract termination.\n"
        "- post_termination_restrictions: Obligations after termination (e.g., non-compete clauses).\n"
        "- survival_clauses: Clauses that remain in effect post-termination.\n"
        "- exit_compensation: Penalties or fees for early termination.\n"
        "- exclusivity_agreement: Details on any exclusivity agreements.\n"
        "- market_restrictions: Geographic or sector-specific limitations.\n"
        "- competitor_collaboration_ban: Restrictions on partnering with competitors.\n"
        "- post_contract_restriction_period: Duration of non-compete obligations after contract ends.\n"
        "- data_processing_agreement: GDPR compliance details related to data processing.\n"
        "- third_party_disclosure_restrictions: Limits on sharing confidential information with third parties.\n"
        "- confidentiality_duration: Duration for which confidentiality obligations persist.\n"
        "- sensitive_data_definition: Definitions regarding sensitive or proprietary data.\n"
        "- security_measures: Measures such as encryption or access control details.\n"
        "- marketing_approval_requirement: Whether marketing materials require prior approval.\n"
        "- co_branding_agreements: Permissions for joint branding initiatives.\n"
        "- use_of_trademark_in_ads: Allowed usage of trademarks in advertising.\n"
        "- sales_channel_limitations: Restrictions on sales channels (online vs. offline).\n"
        "- influencer_advertising_restrictions: Terms protecting brand image in influencer campaigns.\n"
        "- reporting_requirements: Requirements for sales or performance reporting.\n"
        "- kpi_tracking: Key performance indicators mentioned in the contract.\n"
        "- performance_bonuses: Bonus structures tied to performance metrics.\n"
        "- inspection_rights: Rights to conduct inspections or audits.\n"
        "- equivalent_terms: Alternative phrasings for key legal terms (for AI contextual search).\n"
        "- legal_clause_paraphrasing: Simplified restatements of complex clauses.\n"
        "- contract_relationships: Mapping of obligations between parties.\n"
        "- clause_version_tracking: How clauses have evolved over time.\n\n"
        "==== FILENAME & DATE DETECTION ====\n"
        "- If the text references any filename(s) containing a date (e.g., 'Contract_2023-01-05_v2.pdf'), parse that date.\n"
        "- Compare multiple filenames to determine which is earliest or latest.\n"
        "- If the user asks 'What is the last contract version?', reference the filename with the latest date.\n"
        "- If the user's question references a specific year or date range (e.g., 2020), do NOT reference a file dated earlier than that if there is a matching or later date available.\n"
        "- Store any detected filename(s) under 'source_document' with the parsed date and a flag indicating whether it is the latest version, for example:\n"
        "  \"source_document\": {\"filename\": \"Contract_2023-01-05_v2.pdf\", \"parsed_date\": \"2023-01-05\", \"is_latest_version\": true}\n\n"
        "==== Q&A-STYLE ANSWERS ====\n"
        "After extracting metadata, you may receive questions regarding the contract such as:\n"
        "  - 'What is the last contract version?'\n"
        "  - 'Does the exclusivity clause remain the same?'\n"
        "  - 'Which addendum covers the new performance metric?'\n"
        "When answering:\n"
        "  - Provide short, concise answers.\n"
        "  - Reference the appropriate 'source_document' filename and date if applicable.\n"
        "  - If the user question references a specific year (e.g., 2020), avoid referencing a file dated before that year if a more recent file exists.\n\n"
        "==== FINAL OUTPUT STRUCTURE ====\n"
        "Your final output should be a valid JSON object with two main keys:\n"
        "  \"metadata\": { ... extracted metadata fields ... },\n"
        "  \"answers\": [\n"
        "    { \"question\": \"<user question>\", \"answer\": \"<answer referencing document name/date if applicable>\", \"source_document\": \"<filename if relevant>\" },\n"
        "    ... additional Q&A objects if needed ...\n"
        "  ]\n\n"
        "==== DOCUMENT TEXT ====\n"
        "Document Text (first section):\n"
        f"{text[:3000]}...\n\n"
        "Return ONLY a valid JSON object with the fields that are found. Omit any field that is not present. "
        "Do not add extra commentary or disclaimers."
    )

    response = call_ollama(prompt, max_tokens=512)
    
    try:
        metadata = json.loads(response)
        return metadata
    except:
        import re
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
        if json_match:
            try:
                metadata = json.loads(json_match.group(1))
                return metadata
            except:
                pass
        
        return {
            "document_type": "unknown",
            "extraction_error": "Could not parse metadata response"
        }

# -----------------------------
# Document Processing
# -----------------------------
def process_document(file_data, file_name):
    """
    Processes a document and extracts text, summary, and metadata.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as tmp_file:
        tmp_file.write(file_data.getbuffer())
        tmp_file_path = tmp_file.name

    st.info(f"Processing file: {file_name}")
    text = extract_text_with_ollama(tmp_file_path)
    
    if not text:
        st.warning(f"No text extracted from {file_name}.")
        os.remove(tmp_file_path)
        return None

    summary = summarize_text(text)
    metadata = extract_metadata(text)
    
    os.remove(tmp_file_path)
    
    return {
        "filename": file_name,
        "text": text,
        "summary": summary,
        "metadata": metadata
    }

# -----------------------------
# Utility: Convert Metadata to Text
# -----------------------------
def metadata_to_text(metadata_dict):
    """
    Converts metadata (a dict) to a textual representation for embedding in RAG.
    """
    if not metadata_dict or not isinstance(metadata_dict, dict):
        return ""
    lines = []
    for k, v in metadata_dict.items():
        if isinstance(v, (dict, list)):
            v = json.dumps(v)
        lines.append(f"{k}: {v}")
    return "\n".join(lines)

# -----------------------------
# RAG: Building a Vector Index
# -----------------------------
def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits the text into overlapping chunks of ~`chunk_size` words.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start += (chunk_size - overlap)
    return chunks

def build_vector_index(documents):
    """
    Builds a FAISS index of all text, summary, and metadata for RAG retrieval.
    Stores chunk metadata (doc_id, chunk_text, chunk_type) for later retrieval.
    """
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    chunk_metadata = []
    embeddings_list = []
    
    doc_id = 0
    for doc in documents:
        text_chunks = chunk_text(doc["text"], chunk_size=500, overlap=50)
        for chunk_id, chunk in enumerate(text_chunks):
            embedding = embed_model.encode([chunk])[0]
            embeddings_list.append(embedding)
            chunk_metadata.append({
                "doc_id": doc_id,
                "filename": doc["filename"],
                "chunk_id": chunk_id,
                "chunk_type": "text",
                "chunk_text": chunk
            })
        
        if doc["summary"]:
            summary_chunks = chunk_text(doc["summary"], chunk_size=200, overlap=20)
            for sc_id, chunk in enumerate(summary_chunks):
                embedding = embed_model.encode([chunk])[0]
                embeddings_list.append(embedding)
                chunk_metadata.append({
                    "doc_id": doc_id,
                    "filename": doc["filename"],
                    "chunk_id": sc_id,
                    "chunk_type": "summary",
                    "chunk_text": chunk
                })
        
        meta_text = metadata_to_text(doc["metadata"])
        if meta_text:
            meta_chunks = chunk_text(meta_text, chunk_size=200, overlap=20)
            for mc_id, chunk in enumerate(meta_chunks):
                embedding = embed_model.encode([chunk])[0]
                embeddings_list.append(embedding)
                chunk_metadata.append({
                    "doc_id": doc_id,
                    "filename": doc["filename"],
                    "chunk_id": mc_id,
                    "chunk_type": "metadata",
                    "chunk_text": chunk
                })
        
        doc_id += 1
    
    if not embeddings_list:
        return None, None
    
    embeddings_array = np.array(embeddings_list, dtype=np.float32)
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
    
    return index, chunk_metadata

def rag_search(query, index, chunk_metadata, documents, top_k=3):
    """
    Retrieves the most relevant chunk(s) from the FAISS index for the query
    using embeddings. Returns the combined chunk text for context.
    """
    if not index or not chunk_metadata:
        return None, None
    
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = embed_model.encode([query])[0]
    
    distances, indices = index.search(np.array([query_embedding], dtype=np.float32), top_k)
    
    relevant_texts = []
    for idx in indices[0]:
        chunk_info = chunk_metadata[idx]
        relevant_texts.append(f"[{chunk_info['chunk_type']}] {chunk_info['chunk_text']}")
    
    combined_text = "\n\n".join(relevant_texts)
    
    first_chunk_info = chunk_metadata[indices[0][0]]
    source_doc = documents[first_chunk_info["doc_id"]]
    
    return combined_text, source_doc

# -----------------------------
# RAG-based Answer
# -----------------------------
def answer_query(query, index, chunk_metadata, documents):
    """
    Answers a query by retrieving relevant chunk(s) from doc text, summary, or metadata
    and passing them to the LLM. Minimizes hallucinations by using only retrieved context.
    """
    if not documents:
        return "No documents available.", None
    
    combined_text, source_doc = rag_search(query, index, chunk_metadata, documents, top_k=3)
    if not combined_text:
        return "I couldn't find relevant information in the documents to answer your question.", None
    
    prompt = (
        "You are a legal contract assistant. Use ONLY the following context from the document text, summary, and metadata to answer.\n\n"
        f"Context:\n{combined_text}\n\n"
        f"Question: {query}\n\n"
        "Provide a clear, direct answer based ONLY on the above context. If the information isn't in the context, say so."
    )
    
    answer = call_ollama(prompt, max_tokens=1024, temperature=0.0)
    return answer.strip(), source_doc

# -----------------------------
# Streamlit App
# -----------------------------
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'index' not in st.session_state:
    st.session_state.index = None
if 'chunk_metadata' not in st.session_state:
    st.session_state.chunk_metadata = None

st.title("Legal Document Analysis & Q&A Assistant")

tab1, tab2 = st.tabs(["Document Processing", "Question Answering"])

with tab1:
    st.header("Upload and Process Documents")
    uploaded_files = st.file_uploader(
        "Choose legal document files (PDF, images, etc.)", 
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("Process Documents"):
        st.session_state.documents = []
        progress_bar = st.progress(0)
        for i, uploaded_file in enumerate(uploaded_files):
            doc = process_document(uploaded_file, uploaded_file.name)
            if doc:
                st.session_state.documents.append(doc)
            progress_bar.progress((i+1) / len(uploaded_files))
        
        st.success(f"Processed {len(st.session_state.documents)} documents.")
        
        st.session_state.index, st.session_state.chunk_metadata = build_vector_index(st.session_state.documents)
        if st.session_state.index:
            st.success("RAG index built successfully!")
        else:
            st.warning("No text found to build an index.")
    
    if st.session_state.documents:
        st.subheader("Processed Documents")
        for i, doc in enumerate(st.session_state.documents):
            with st.expander(f"{i+1}. {doc['filename']}"):
                st.markdown("### Summary")
                st.write(doc['summary'])
                st.markdown("### Metadata")
                st.json(doc['metadata'])
                if st.checkbox("Show Full Text", key=f"show_text_{i}"):
                    st.markdown("### Full Document Content")
                    st.text_area("Document Content", value=doc['text'], height=300)

with tab2:
    st.header("Ask Questions About Your Documents")
    if not st.session_state.documents:
        st.warning("Please process documents first before asking questions.")
    else:
        st.write(f"You can ask questions about {len(st.session_state.documents)} processed documents.")
        query = st.text_area("What would you like to know about these documents?", height=100)
        if st.button("Get Answer") and query:
            with st.spinner("Analyzing documents and generating answer..."):
                answer, source_doc = answer_query(query, st.session_state.index, st.session_state.chunk_metadata, st.session_state.documents)
            st.markdown("### Answer")
            st.write(answer)
            if source_doc:
                st.markdown("### Source Document")
                st.info(f"Answer based on: {source_doc['filename']}")

st.markdown("---")
st.markdown("⚖️ Legal Document Assistant powered by Ollama and Llama 3.2 Vision, with RAG (Text+Summary+Metadata) for accurate answers.")
