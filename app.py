# app.py
import os
import io
import time
import json
import base64
import traceback
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import pandas as pd

# Optional NLP
import spacy
from spacy.util import is_package

# ----------------------------
# Page & app configuration
# ----------------------------
st.set_page_config(page_title="INSIGHT IQ", layout="wide")

# ----------------------------
# Load .env (optional default)
# ----------------------------
load_dotenv()
DEFAULT_API_KEY = os.getenv("GOOGLE_API_KEY", None)

# ----------------------------
# Session state defaults
# ----------------------------
if "page" not in st.session_state:
    st.session_state.page = "API Key Setup" if not DEFAULT_API_KEY else "Upload Files"

# Store API key in session for the session lifetime
if "GOOGLE_API_KEY" not in st.session_state:
    st.session_state["GOOGLE_API_KEY"] = DEFAULT_API_KEY

# Files, indexes, caches
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = {}
if "combined_faiss_index" not in st.session_state:
    st.session_state.combined_faiss_index = None
if "last_api_call_time" not in st.session_state:
    st.session_state.last_api_call_time = 0.0
if "api_call_count" not in st.session_state:
    st.session_state.api_call_count = 0

# Local caches
CACHE_RESPONSES = {}
DOCUMENT_SUMMARIES = {}

# Minimum delay between LLM calls (seconds). Adjust as needed.
MIN_DELAY_SECONDS = 1.5

# ----------------------------
# Configure genai if key exists
# ----------------------------
def configure_genai_from_session():
    key = st.session_state.get("GOOGLE_API_KEY")
    if key:
        try:
            genai.configure(api_key=key)
            return True
        except Exception:
            # swallow and return False
            return False
    return False

if st.session_state.get("GOOGLE_API_KEY"):
    configure_genai_from_session()

# ----------------------------
# Load spaCy model (with graceful fallback)
# ----------------------------
nlp = None
try:
    # Try to load if available
    nlp = spacy.load("en_core_web_sm")
except Exception:
    try:
        # As fallback try package name (if wheel installed)
        if is_package("en_core_web_sm"):
            nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = None

# ----------------------------
# Utilities
# ----------------------------
def get_image_base64(image_path: str) -> Optional[str]:
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception:
        return None


def extract_entities(text: str) -> dict:
    """Extract a small set of market-related entities using spaCy (if installed)."""
    if nlp is None:
        return {}
    doc = nlp(text)
    # Return dict label -> list of strings (deduped)
    res = {}
    for ent in doc.ents:
        if ent.label_ in ["ORG", "MONEY", "PERCENT", "GPE"]:
            res.setdefault(ent.label_, []).append(ent.text)
    # reduce lists
    return {k: list(dict.fromkeys(v)) for k, v in res.items()}


def get_pdf_text(uploaded_file) -> str:
    """Safely read PDF bytes and extract text from pages; returns a single string."""
    try:
        uploaded_file.seek(0)
        pdf_bytes = uploaded_file.read()
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        pages_text = []
        for p in pdf_reader.pages:
            try:
                text = p.extract_text()
                if text:
                    pages_text.append(text)
            except Exception:
                # ignore problematic page
                continue
        return "\n".join(pages_text)
    except Exception as e:
        st.error(f"Failed to read PDF: {e}")
        return ""


def split_text_into_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    if not text:
        return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


# Exponential backoff wrapper
def call_with_backoff(func, *args, max_retries=3, initial_delay=1.0, **kwargs):
    attempt = 0
    delay = initial_delay
    while attempt <= max_retries:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                raise
            time.sleep(delay)
            delay *= 2


# Safe detection of Markdown table
def looks_like_markdown_table(text: str) -> bool:
    if not text:
        return False
    return ("|" in text and ("---" in text or ":--" in text or "--:" in text))


# ----------------------------
# FAISS & Embeddings helpers
# ----------------------------
def make_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Create embeddings configured with the session API key.
    Google library reads genai configured key; we try to construct embedding object.
    """
    if not st.session_state.get("GOOGLE_API_KEY"):
        raise RuntimeError("Google API key not configured. Please set it in 'API Key Setup'.")

    # Ensure genai configured
    configure_genai_from_session()

    # Create embeddings object (langchain wrapper). This will rely on configured genai api key.
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")


def safe_faiss_from_texts(texts: List[str]) -> FAISS:
    """
    Wrap FAISS.from_texts with retries and chunk size checks.
    """
    if not texts:
        raise ValueError("No texts to embed.")

    # Attempt direct create; if it fails (likely due to size), retry with smaller chunks
    try:
        embeddings = make_embeddings()
        return FAISS.from_texts(texts, embedding=embeddings)
    except Exception as e:
        # try splitting texts further and retry
        small_chunks = []
        for t in texts:
            small_chunks.extend(split_text_into_chunks(t, chunk_size=500, chunk_overlap=50))
        embeddings = make_embeddings()
        return FAISS.from_texts(small_chunks, embedding=embeddings)


# ----------------------------
# Core processing functions
# ----------------------------
def process_file(uploaded_file) -> Optional[FAISS]:
    """
    Extract text from uploaded PDF, split into chunks safely and create a FAISS index saved to disk.
    Returns FAISS vector store or None on error.
    """
    try:
        if not st.session_state.get("GOOGLE_API_KEY"):
            st.error("Please set your Google API key first on the 'API Key Setup' page.")
            return None

        raw_text = get_pdf_text(uploaded_file)
        if not raw_text.strip():
            st.error(f"No extractable text found in file: {uploaded_file.name}")
            return None

        # chunk size tuned to stay well within token limits of embeddings
        text_chunks = split_text_into_chunks(raw_text, chunk_size=1000, chunk_overlap=100)

        # create faiss vector store (with retries inside safe_faiss_from_texts)
        vector_store = call_with_backoff(safe_faiss_from_texts, text_chunks, max_retries=3, initial_delay=1.0)

        # persist locally (folder per file)
        file_name = os.path.splitext(uploaded_file.name)[0]
        folder_path = os.path.join("faiss_indexes", file_name)
        os.makedirs(folder_path, exist_ok=True)
        vector_store.save_local(folder_path)

        st.success(f"Processed and saved FAISS index for {uploaded_file.name}")
        return vector_store

    except Exception as e:
        tb = traceback.format_exc()
        st.error(f"Failed to process {uploaded_file.name}: {e}")
        st.write(f"Debug info:\n```\n{tb}\n```")
        return None


def load_faiss_for_file(file_name: str) -> Optional[FAISS]:
    folder_path = os.path.join("faiss_indexes", file_name)
    try:
        emb = make_embeddings()
        db = FAISS.load_local(folder_path, emb, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Error loading FAISS index for {file_name}: {e}")
        return None


def combine_faiss_indexes(vector_stores: List[FAISS]) -> Optional[FAISS]:
    if not vector_stores:
        return None
    combined = vector_stores[0]
    for vs in vector_stores[1:]:
        try:
            combined.merge_from(vs)
        except Exception:
            # if merge fails, ignore to continue
            continue
    return combined


# ----------------------------
# LLM Chains & analysis
# ----------------------------
def get_analysis_chain(prompt_template: str):
    """
    Returns a langchain QA chain using Gemini Chat model.
    """
    # model is ChatGoogleGenerativeAI wrapper
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0.3)
    return load_qa_chain(model, chain_type="stuff", prompt=PromptTemplate(template=prompt_template, input_variables=["context"]))


def analyze_document(file_name: str, query: str, prompt_template: str) -> Optional[str]:
    """
    Load FAISS, retrieve relevant docs and run the QA chain.
    """
    if not st.session_state.get("GOOGLE_API_KEY"):
        return "API key not configured."

    db = load_faiss_for_file(file_name)
    if db is None:
        return f"FAISS index for {file_name} could not be loaded."

    docs = db.similarity_search(query, k=4)
    chain = get_analysis_chain(prompt_template)
    try:
        # rate limiting between calls
        ensure_rate_limit()
        result = chain({"input_documents": docs}, return_only_outputs=True)
        return result.get("output_text", "")
    except Exception as e:
        return f"Error during analysis: {e}"


def comparative_analysis(file_name: str, query: str, domain: str) -> str:
    """
    Compose a domain-specific structured query and call Gemini directly for comparative analysis.
    """
    if not st.session_state.get("GOOGLE_API_KEY"):
        return "API key not configured."

    # Validate loading FAISS
    db = load_faiss_for_file(file_name)
    if db is None:
        return f"FAISS index for {file_name} could not be loaded."

    # structured prompts per domain
    if domain == "Mutual Funds":
        structured_query = (
            "Perform a comparative analysis of competitors in the Mutual funds sector based on the uploaded document. "
            "*Extract real data points directly from the document. Do not use hypothetical data.*\n\n"
            "Output the response in a structured tabular format with the following columns: Metric, This File, Primary Benchmark, Secondary Benchmark, Source (Page Number). "
            "Ensure clarity, completeness, and actionable insights. Include page numbers or table numbers for each data point. Mark missing data as 'N/A'.\n\n"
            "Focus on metrics: NAV, Performance (Returns %), 5 Year Avg forward P/E, 10 Year Avg forward P/E, Expense ratio (Regular plan), Return to Risk Ratio (March'14-March'24), Sector Allocation, Top holdings.\n"
        )
    elif domain == "Life Insurance":
        structured_query = (
            "Perform a comparative analysis of competitors in the Life Insurance sector based on the uploaded document. "
            "*Extract real data points directly from the document. Do not use hypothetical data.*\n\n"
            "Output as a Markdown table comparing competitor names as columns and metrics as rows. Mark missing data as 'N/A'."
        )
    else:
        return "Invalid domain selection."

    # Retrieve context (top docs)
    try:
        docs = db.similarity_search(query, k=6)
        context_text = "\n\n".join([d.page_content for d in docs])
    except Exception as e:
        return f"Error retrieving context from FAISS: {e}"

    # Build prompt
    prompt = (
        "You are a business analysis expert specialized in competitive analysis. Use only the data available in the context below. "
        "Cite sources with page numbers or section names if present. If not present, write 'Source: Document'.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Task:\n{structured_query}\n\n"
        "Return ONLY a Markdown table (or tables) and short notes if needed."
    )

    try:
        ensure_rate_limit()
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0.2)
        response = call_with_backoff(lambda p: model.invoke(p), prompt, max_retries=3, initial_delay=1.0)
        return getattr(response, "content", str(response))
    except Exception as e:
        return f"Error during Gemini analysis: {e}"


def summarize_document(text: str) -> str:
    if not text:
        return ""
    if text in DOCUMENT_SUMMARIES:
        return DOCUMENT_SUMMARIES[text]
    try:
        ensure_rate_limit()
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0.3)  # ✅ updated
        prompt = (
            "Summarize the following document while extracting the key points, metrics, and bulletable insights. "
            "Keep it concise and factual. Use only the content provided.\n\n"
            f"Document:\n{text}\n\nSummary:"
        )
        response = call_with_backoff(lambda p: model.invoke(p), prompt, max_retries=3, initial_delay=1.0)
        summary = getattr(response, "content", str(response))
        DOCUMENT_SUMMARIES[text] = summary
        return summary
    except Exception as e:
        return f"Error summarizing document: {e}"


# ----------------------------
# Rate limiting helper
# ----------------------------
def ensure_rate_limit():
    now = time.time()
    last = st.session_state.get("last_api_call_time", 0.0)
    elapsed = now - last
    if elapsed < MIN_DELAY_SECONDS:
        time.sleep(MIN_DELAY_SECONDS - elapsed)
    st.session_state["last_api_call_time"] = time.time()
    st.session_state["api_call_count"] = st.session_state.get("api_call_count", 0) + 1


# ----------------------------
# Chatbot functionality
# ----------------------------
def chatbot_response(user_input: str) -> str:
    if not st.session_state.get("uploaded_files"):
        return "No documents uploaded. Please upload documents first."

    if user_input in CACHE_RESPONSES:
        return CACHE_RESPONSES[user_input]

    if not st.session_state.get("combined_faiss_index"):
        return "No combined index found. Please process files and create the index."

    try:
        docs = st.session_state.combined_faiss_index.similarity_search(user_input, k=3)
        relevant_chunks = [d.page_content.strip() for d in docs]
        summarized_context = "\n\n".join(filter(None, relevant_chunks))
        if not summarized_context.strip():
            return "No relevant content found in uploaded documents."

        prompt = (
            "You are a helpful assistant that must answer ONLY using the provided context. Do not hallucinate.\n\n"
            f"Context:\n{summarized_context}\n\n"
            f"User Question: {user_input}\n\n"
            "Provide a concise answer (max ~100 words). If answer is not in context, reply 'Not available in documents.'"
        )
        ensure_rate_limit()
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0.2)  # ✅ updated
        response = call_with_backoff(lambda p: model.invoke(p), prompt, max_retries=3, initial_delay=1.0)
        text = getattr(response, "content", str(response))
        CACHE_RESPONSES[user_input] = text
        return text
    except Exception as e:
        return f"Error generating chatbot response: {e}"


# ----------------------------
# Storytelling helper
# ----------------------------
def generate_storytelling_insights(analysis_result: str, analysis_type: str, selected_domain: str, file_name: str) -> str:
    # keep prompt small to avoid tokens explosion
    short_context = analysis_result if len(analysis_result) < 4000 else analysis_result[:4000]
    if selected_domain == "Mutual Funds":
        if analysis_type == "Competitor Strategy":
            storytelling_prompt = f"Summarize competitor strategy analysis in 3-5 sentences focusing on actionable investor insights for {file_name}."
        elif analysis_type == "Market Trends":
            storytelling_prompt = f"In 3-5 sentences, describe major market trends from the analysis for {file_name}."
        elif analysis_type == "SWOT Analysis":
            storytelling_prompt = f"Provide a 3-5 sentence SWOT summary based on the analysis for {file_name}."
        else:
            storytelling_prompt = "Provide a concise 3-5 sentence summary of the analysis."
    elif selected_domain == "Life Insurance":
        storytelling_prompt = f"Provide a concise 3-5 sentence summary of the analysis for {file_name}."
    else:
        storytelling_prompt = "Provide a concise summary."

    prompt = storytelling_prompt + "\n\nContext:\n" + short_context
    try:
        ensure_rate_limit()
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0.25)
        response = call_with_backoff(lambda p: model.invoke(p), prompt, max_retries=2, initial_delay=1.0)
        return getattr(response, "content", str(response))
    except Exception as e:
        return f"Error generating storytelling insights: {e}"


# ----------------------------
# UI: Sidebar & Navigation
# ----------------------------

st.markdown(
    """
    <style>
    .stButton > button {
        width: 100%;
        margin-bottom: 5px;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ✅ Show logo first (top of sidebar)
logo_b64 = get_image_base64("INSIGHT IQ LOGO.png")
if logo_b64:
    st.sidebar.markdown(
        f"""
        <div style="display:flex;align-items:center;justify-content:center;margin-bottom:10px;">
            <img src="data:image/png;base64,{logo_b64}" width="180"/>
        </div>
        """,
        unsafe_allow_html=True
    )

# ✅ Navigation title centered below logo
st.sidebar.markdown(
    "<h3 style='text-align:center; margin-bottom:15px;'>------------ Navigation Bar ------------</h3>",
    unsafe_allow_html=True
)

# ✅ Navigation buttons
PAGES = {
    "API Key Setup": "🔑",
    "Upload Files": "📂",
    "Analysis": "📈",
    "Files": "📁",
    "Chatbot": "🤖",
    "Summary View": "📑",
    "Dashboard": "📊"
}

for page_name, icon in PAGES.items():
    if st.sidebar.button(f"{icon} {page_name}"):
        st.session_state.page = page_name

# ----------------------------
# Pages
# ----------------------------
if st.session_state.page == "API Key Setup":
    st.title("🔑 API Key Setup")
    st.write("Enter your Google Generative AI API key for this session. This will be stored only for the session (in Streamlit state).")
    api_key = st.text_input("Google API Key", value=st.session_state.get("GOOGLE_API_KEY", "") or "", type="password")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Save Key (session)"):
            if api_key and api_key.strip():
                st.session_state["GOOGLE_API_KEY"] = api_key.strip()
                ok = configure_genai_from_session()
                if ok:
                    st.success("API key saved to session and configured.")
                else:
                    st.error("Failed to configure genai with that key. Please verify.")
            else:
                st.error("Please enter a valid API key.")
    with col2:
        if st.button("Save to .env (optional)"):
            if api_key and api_key.strip():
                # append or replace .env entry
                env_path = ".env"
                try:
                    # read existing .env
                    lines = []
                    if os.path.exists(env_path):
                        with open(env_path, "r", encoding="utf-8") as f:
                            lines = f.readlines()
                    found = False
                    for i, ln in enumerate(lines):
                        if ln.startswith("GOOGLE_API_KEY="):
                            lines[i] = f'GOOGLE_API_KEY="{api_key.strip()}"\n'
                            found = True
                            break
                    if not found:
                        lines.append(f'GOOGLE_API_KEY="{api_key.strip()}"\n')
                    with open(env_path, "w", encoding="utf-8") as f:
                        f.writelines(lines)
                    st.success("Saved API key to .env. (You may need to restart the app for .env to be read on next cold start.)")
                    st.session_state["GOOGLE_API_KEY"] = api_key.strip()
                    configure_genai_from_session()
                except Exception as e:
                    st.error(f"Failed to save .env: {e}")
            else:
                st.error("Please enter a valid API key.")

    st.markdown("---")
    st.write("Current session API key set:" , "Yes" if st.session_state.get("GOOGLE_API_KEY") else "No")
    if st.session_state.get("GOOGLE_API_KEY"):
        if st.button("Clear session API key"):
            st.session_state["GOOGLE_API_KEY"] = None
            st.success("Cleared API key from session.")


elif st.session_state.page == "Upload Files":
    st.title("Upload Competitor Reports")
    st.write("### Step 1: Select the Document Domain")
    domain_mapping = {
        "Mutual Funds": "Investment strategies, fund performance, expense ratios, and marketing tactics.",
        "Life Insurance": "Risk management, policy innovations, customer engagement strategies."
    }

    if "selected_domain" not in st.session_state:
        st.session_state.selected_domain = "Select a domain"

    selected_domain = st.selectbox(
        "Select Document Domain",
        ["Select a domain"] + list(domain_mapping.keys()),
        index=list(domain_mapping.keys()).index(st.session_state.selected_domain) if st.session_state.selected_domain in domain_mapping else 0
    )
    if selected_domain != "Select a domain":
        st.session_state.selected_domain = selected_domain

    st.write("### Step 2: Upload PDF Reports")
    if selected_domain == "Select a domain":
        st.warning("Please select a domain before uploading PDFs.")
        file_uploader_disabled = True
    else:
        st.success(f"You selected *{selected_domain}*. Expected PDF content: {domain_mapping[selected_domain]}")
        file_uploader_disabled = False

    uploaded_files = st.file_uploader("Select PDFs", type=["pdf"], accept_multiple_files=True, disabled=file_uploader_disabled)
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.success(f"{len(uploaded_files)} file(s) uploaded successfully for {selected_domain} analysis.")

    if st.button("Submit & Process") and st.session_state.uploaded_files:
        # require API key
        if not st.session_state.get("GOOGLE_API_KEY"):
            st.error("Please set your Google API key first on the 'API Key Setup' page.")
        else:
            st.write("Processing files...")
            st.session_state.analysis_results = {}
            st.session_state.current_file_index = 0
            vector_stores = []
            for pdf_file in st.session_state.uploaded_files:
                with st.spinner(f"Processing {pdf_file.name}..."):
                    vs = process_file(pdf_file)
                    if vs:
                        vector_stores.append(vs)
            if vector_stores:
                st.session_state.combined_faiss_index = combine_faiss_indexes(vector_stores)
                st.success("All files processed and combined into a single index.")
            else:
                st.warning("No files were successfully processed to create the index.")

    if st.button("Proceed to Analysis"):
        st.session_state.page = "Analysis"
        st.rerun()



elif st.session_state.page == "Analysis":
    st.title("Run AI-Driven Analysis")
    if "selected_domain" not in st.session_state or st.session_state.selected_domain == "Select a domain":
        st.error("No domain selected. Please go back and upload files with a domain.")
        if st.button("Go Back to Upload Page"):
            st.session_state.page = "Upload Files"
            st.experimental_rerun()
    elif not st.session_state.get("uploaded_files"):
        st.warning("No files uploaded! Please upload relevant financial and market analysis documents.")
        if st.button("Go Back to Upload Page"):
            st.session_state.page = "Upload Files"
            st.rerun()

    else:
        selected_domain = st.session_state.selected_domain
        # domain prompts (kept succinct)
        domain_prompts = {
            "Mutual Funds": {
                "Competitor Strategy": "Analyze competitor strategies within the Mutual Funds sector. Use data from the document only.",
                "Market Trends": "Identify key market trends in Mutual Funds. Use data from the document only.",
                "SWOT Analysis": "Perform a SWOT analysis using only the document data.",
                "Comparative Analysis": "{selected_domain} comparative analysis request received. Present the output in a Markdown table, comparing key metrics. ONLY use data points found in the document."
            },
            "Life Insurance": {
                "Competitor Strategy": "Analyze competitor strategies within the Life Insurance sector. Use data from the document only.",
                "Market Trends": "Identify key market trends in Life Insurance. Use data from the document only.",
                "SWOT Analysis": "Perform a SWOT analysis using only the document data.",
                "Comparative Analysis": "{selected_domain} comparative analysis request received. Present the output in a Markdown table, comparing key metrics. ONLY use data points found in the document."
            }
        }
        analysis_options = list(domain_prompts[selected_domain].keys())
        analysis_type = st.selectbox("Select analysis type", analysis_options)

        if st.button("Run Analysis"):
            st.session_state.analysis_results = {}
            st.session_state.page = "Summary View"
            for pdf_file in st.session_state.uploaded_files:
                file_name = os.path.splitext(pdf_file.name)[0]
                folder_path = os.path.join("faiss_indexes", file_name)
                if not os.path.exists(folder_path):
                    st.error(f"FAISS index for {file_name} not found. Process the document first.")
                    continue
                query = domain_prompts[selected_domain][analysis_type].format(selected_domain=selected_domain)
                with st.spinner(f"Analyzing {pdf_file.name}..."):
                    try:
                        if analysis_type == "Comparative Analysis":
                            report = comparative_analysis(file_name=file_name, query=query, domain=selected_domain)
                            storytelling_insights = generate_storytelling_insights(report, analysis_type, selected_domain, file_name)
                        else:
                            report = analyze_document(file_name, query, f"Context: {{context}}")
                            storytelling_insights = generate_storytelling_insights(report, analysis_type, selected_domain, file_name)
                        st.session_state.setdefault("analysis_history", []).append({
                            "query": query,
                            "domain": selected_domain,
                            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "result": report,
                            "storytelling_insights": storytelling_insights
                        })
                        st.session_state.analysis_results[pdf_file.name] = {"report": report, "storytelling_insights": storytelling_insights}
                    except Exception as e:
                        st.error(f"Analysis failed for {pdf_file.name}: {e}")
            st.rerun()


    if st.button("Back to Upload Page"):
        st.session_state.page = "Upload Files"
        st.rerun()



elif st.session_state.page == "Dashboard":
    st.title("Dashboard 📊")
    st.metric("📊 Total Analyses", len(st.session_state.analysis_history))
    st.subheader("📜 Analysis History")
    if st.session_state.analysis_history:
        for analysis in st.session_state.analysis_history:
            st.markdown(f"*Query:* {analysis['query']}")
            st.markdown(f"*Timestamp:* {analysis['timestamp']}")
            st.markdown("**Result:**")
            if looks_like_markdown_table(analysis["result"]):
                st.markdown(analysis["result"], unsafe_allow_html=True)
            else:
                st.text_area("", value=str(analysis["result"]), height=120)
            if analysis.get("storytelling_insights"):
                st.markdown("**Summary:**")
                st.markdown(analysis["storytelling_insights"])
            st.markdown("---")
    else:
        st.info("No analysis history available.")


elif st.session_state.page == "Files":
    st.title("Uploaded Files")
    if st.session_state.uploaded_files:
        for file in st.session_state.uploaded_files:
            st.write(f"📄 {file.name}")
    else:
        st.info("No uploaded files yet.")


elif st.session_state.page == "Chatbot":
    st.title("Intel360 Chatbot 🤖")
    st.markdown("Ask about competitor analysis, insights, and AI-generated reports (answers will be based on uploaded documents).")
    user_input = st.text_input("Ask me anything about competitor analysis:")
    if user_input:
        with st.spinner("Contacting model..."):
            resp = chatbot_response(user_input)
            st.session_state.chat_history.append({"query": user_input, "response": resp})
            st.markdown("### 🤖 Chatbot Response")
            if looks_like_markdown_table(resp):
                st.markdown(resp, unsafe_allow_html=True)
            else:
                st.write(resp)
    st.subheader("🗂️ Chat History")
    for chat in st.session_state.chat_history:
        st.write(f"Q: {chat['query']}")
        st.markdown(f"A: {chat['response']}")
        st.markdown("---")


elif st.session_state.page == "Summary View":
    st.title("Summary View 📑")
    if not st.session_state.analysis_results:
        st.info("No analysis has been performed yet. Please run an analysis first.")
        if st.button("Back to Analysis Page"):
            st.session_state.page = "Analysis"
            st.rerun()

    else:
        file_names = list(st.session_state.analysis_results.keys())
        selected_file_name = st.selectbox("Select File to View Summary", file_names)
        report = st.session_state.analysis_results[selected_file_name]["report"]
        storytelling_insights = st.session_state.analysis_results[selected_file_name]["storytelling_insights"]
        st.subheader("✨ Summary")
        if storytelling_insights:
            st.markdown(storytelling_insights)
        else:
            st.write("No storytelling summary available.")
        st.subheader("Raw Analysis Result")
        if isinstance(report, str) and looks_like_markdown_table(report):
            st.markdown(report, unsafe_allow_html=True)
        else:
            st.text_area(f"Analysis Report for {selected_file_name}", value=str(report), height=300)
        if st.button("Back to Analysis Page"):
            st.session_state.page = "Analysis"
            st.rerun()


# end of app
