import os
import subprocess
import json
import tempfile
import streamlit as st
from ollama_ocr import OCRProcessor

# Set page configuration
st.set_page_config(
    page_title="Legal Document Q&A Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Call Ollama for Text Generation
# -----------------------------
def call_ollama(prompt, model="llama3.2-vision:11b", max_tokens=512, temperature=0.1):
    """
    Call Ollama API with the provided prompt
    """
    url = "http://localhost:11434/api/generate"
    import requests
    
    # Create the request payload
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        if response.status_code == 200:
            response_json = response.json()
            return response_json.get("response", "")
        else:
            st.error(f"API Error: Status code {response.status_code}")
            return ""
    except Exception as e:
        st.error(f"Error calling Ollama API: {str(e)}")
        return ""

# -----------------------------
# Functions for Document Processing
# -----------------------------
def extract_text_with_ollama(file_path):
    """
    Uses the ollama-ocr package to extract text from a file.
    """
    try:
        ocr = OCRProcessor(model_name="llama3.2-vision:11b")
        # Using plain text output format
        text = ocr.process_image(image_path=file_path, format_type="text")
        return text
    except Exception as e:
        st.error(f"Error processing {file_path}: {e}")
        return ""

def summarize_text(text):
    """
    Uses Ollama to summarize the legal document text.
    """
    prompt = f"Summarize the following legal document text in a comprehensive way:\n\n{text[:3000]}...\n\nProvide a detailed summary that captures key points, parties involved, obligations, dates, and important clauses."
    response = call_ollama(prompt, max_tokens=512)
    return response

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
    "- If the user’s question references a specific year or date range (e.g., 2020), do NOT reference a file dated earlier than that if there is a matching or later date available.\n"
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
    
    # Try to extract and parse JSON from the response
    try:
        # Check if response is already valid JSON
        metadata = json.loads(response)
        return metadata
    except:
        # Try to extract JSON part from text response
        import re
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
        if json_match:
            try:
                metadata = json.loads(json_match.group(1))
                return metadata
            except:
                pass
        
        # If all else fails, return basic metadata
        return {
            "document_type": "unknown",
            "extraction_error": "Could not parse metadata response"
        }

def process_document(file_data, file_name):
    """
    Processes a document and extracts text, summary, and metadata.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as tmp_file:
        tmp_file.write(file_data.getbuffer())
        tmp_file_path = tmp_file.name

    st.info(f"Processing file: {file_name}")
    # Extract text using OCR
    text = extract_text_with_ollama(tmp_file_path)
    
    if not text:
        st.warning(f"No text extracted from {file_name}.")
        os.remove(tmp_file_path)
        return None

    # Summarize the document text
    summary = summarize_text(text)
    
    # Extract metadata
    metadata = extract_metadata(text)
    
    # Clean up temporary file
    os.remove(tmp_file_path)
    
    return {
        "filename": file_name,
        "text": text,
        "summary": summary,
        "metadata": metadata
    }

# -----------------------------
# Simple text-based search function (no PyTorch/FAISS)
# -----------------------------
def search_documents(query, documents):
    """
    Simple keyword-based search to find relevant documents
    """
    query_terms = query.lower().split()
    document_scores = []
    
    for doc in documents:
        # Calculate simple relevance score based on term frequency
        text = doc['text'].lower()
        score = sum(1 for term in query_terms if term in text)
        document_scores.append((doc, score))
    
    # Sort by relevance score in descending order
    document_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return the most relevant document(s)
    if document_scores and document_scores[0][1] > 0:
        return document_scores[0][0]  # Return the most relevant document
    else:
        return None  # No relevant document found

def answer_query(query, documents):
    """
    Answers a query using the relevant document as context
    """
    # Find most relevant document
    relevant_doc = search_documents(query, documents)
    
    if not relevant_doc:
        return "I couldn't find relevant information in the documents to answer your question."
    
    # Use the document text as context for answering
    prompt = (
        f"Using the following legal document, answer this question accurately and cite specific sections:\n\n"
        f"DOCUMENT: {relevant_doc['filename']}\n"
        f"CONTENT: {relevant_doc['text'][:4000]}...\n\n"
        f"QUESTION: {query}\n\n"
        f"Provide a clear, direct answer based only on information in the document. "
        f"If the information isn't in the document, say so. "
        f"Cite specific sections or clauses when possible."
    )
    
    answer = call_ollama(prompt, max_tokens=1024, temperature=0.2)
    return answer, relevant_doc

# Store processed documents in session state
if 'documents' not in st.session_state:
    st.session_state.documents = []

# -----------------------------
# Streamlit UI
# -----------------------------
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
        # Clear previous documents
        st.session_state.documents = []
        
        progress_bar = st.progress(0)
        for i, uploaded_file in enumerate(uploaded_files):
            doc = process_document(uploaded_file, uploaded_file.name)
            if doc:
                st.session_state.documents.append(doc)
            progress_bar.progress((i+1)/len(uploaded_files))
        
        st.success(f"Processed {len(st.session_state.documents)} documents.")
    
    # Display processed documents
    if st.session_state.documents:
        st.subheader("Processed Documents")
        for i, doc in enumerate(st.session_state.documents):
            with st.expander(f"{i+1}. {doc['filename']}"):
                st.markdown("### Summary")
                st.write(doc['summary'])
                
                st.markdown("### Metadata")
                st.json(doc['metadata'])
                
                # Use checkbox instead of nested expander
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
                answer, source_doc = answer_query(query, st.session_state.documents)
            
            st.markdown("### Answer")
            st.write(answer)
            
            st.markdown("### Source Document")
            st.info(f"Answer based on: {source_doc['filename']}")

# Footer
st.markdown("---")
st.markdown("⚖️ Legal Document Assistant powered by Ollama and Llama 3.2 Vision")
