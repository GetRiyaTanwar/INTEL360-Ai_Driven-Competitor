import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import pandas as pd
import io
from dotenv import load_dotenv
import spacy
import base64



# Set page config - move this to the top as the first Streamlit command
st.set_page_config(page_title="INSIGHT IQ", layout="wide")

# Convert the image to base64
def get_image_base64(image_path):
    """Convert the image to base64 format for embedding in HTML."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        st.error("Logo file not found. Please ensure the correct path.")
        return None

# Directly use the uploaded image
logo_base64 = get_image_base64("INSIGHT IQ LOGO.png")

# Display the image in the sidebar if loaded successfully
if logo_base64:
    st.sidebar.markdown(f"""
        <div style="display: flex; flex-direction: column; align-items: center; margin-top: -40px; margin-bottom: -30px;">
            <img src="data:image/png;base64,{logo_base64}" alt="Logo" style="width: 200px; height: auto;">
        </div>
    """, unsafe_allow_html=True)



# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Session State Initialization
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Load NLP Model for Entity Recognition
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    """Extracts market-related entities from text using NLP."""
    doc = nlp(text)
    return {ent.label_: ent.text for ent in doc.ents if ent.label_ in ["ORG", "MONEY", "PERCENT", "GPE"]}

# Extract text from PDFs
def get_pdf_text(pdf_file):
    pdf_bytes = pdf_file.read()
    pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
    return "".join([page.extract_text() for page in pdf_reader.pages])

# Split text into chunks
def get_text_chunks(text):
    return RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000).split_text(text)

# Process and save FAISS index
def process_file(pdf_file):
    raw_text = get_pdf_text(pdf_file)
    text_chunks = get_text_chunks(raw_text)
    file_name = os.path.splitext(pdf_file.name)[0]
    folder_path = f"faiss_indexes/{file_name}"
    os.makedirs(folder_path, exist_ok=True)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(folder_path)
    st.success(f"Processed: {pdf_file.name}")

# AI Model for analysis
def get_analysis_chain(prompt_template):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    return load_qa_chain(model, chain_type="stuff", prompt=PromptTemplate(template=prompt_template, input_variables=["context"]))

# Document Analysis
def analyze_document(file_name, query, prompt_template):
    folder_path = f"faiss_indexes/{file_name}"
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    try:
        new_db = FAISS.load_local(folder_path, embeddings, allow_dangerous_deserialization=True)
    except RuntimeError:
        st.error(f"Error loading FAISS index for {file_name}.")
        return
    
    docs = new_db.similarity_search(query)
    chain = get_analysis_chain(prompt_template)
    response = chain({"input_documents": docs}, return_only_outputs=True)
    
    return response["output_text"]

# Chatbot Functionality
def chatbot_response(user_input):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    response = model.invoke(user_input)
    return response

# Main UI
st.title("INSIGHT IQ: AI-Driven Competitor Analysis 📊")

# Sidebar navigation
with st.sidebar:
    st.header("------------Navigation Bar-------------")

    if "page" not in st.session_state:
        st.session_state.page = "Upload Files"
    
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

    pages = {
        "Dashboard": "📊",
        "Upload Files": "📂",
        "Analysis": "📈",
        "Files": "📁",
        "Chatbot": "🤖"
    }

    for page, icon in pages.items():
        if st.button(f"{icon} {page}"):
            st.session_state.page = page

if st.session_state.page == "Upload Files":
    st.title("Upload Competitor Reports")
    
    # Step 1: Select Document Domain
    st.write("### Step 1: Select the Document Domain")
    domain_mapping = {
        "Healthcare": "Market growth, regulatory impact, emerging technologies.",
        "Life Insurance": "Risk management, policy innovations, customer engagement strategies.",
        "Mutual Funds": "Investment patterns, fund performance, economic impact.",
        "Lending or Diversified NBFCs": "Lending strategies, credit risk, financial stability."
    }

    selected_domain = st.selectbox("Select Document Domain", ["Select a domain"] + list(domain_mapping.keys()))

    # Step 2: Upload PDF (Disabled if no domain is selected)
    st.write("### Step 2: Upload PDF Reports")
    if selected_domain == "Select a domain":
        st.warning("Please select a domain before uploading PDFs.")
        file_uploader_disabled = True
    else:
        st.success(f"You selected **{selected_domain}**. Expected PDF content: {domain_mapping[selected_domain]}")
        file_uploader_disabled = False

    uploaded_files = st.file_uploader(
        "Select PDFs", 
        type=["pdf"], 
        accept_multiple_files=True,
        disabled=file_uploader_disabled
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded successfully for {selected_domain} analysis.")
        st.session_state["selected_domain"] = selected_domain  # Store domain in session state

    if st.button("Submit & Process") and uploaded_files:
        st.write("Processing files...")
        st.session_state.uploaded_files = uploaded_files
        for pdf_file in uploaded_files:
            process_file(pdf_file)


elif st.session_state.page == "Dashboard":
    st.title("Dashboard 📊")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("📊 Total Analyses", len(st.session_state.analysis_history))

    # Display Analysis History in Column Form
    st.subheader("📜 Analysis History")
    
    if st.session_state.analysis_history:
        for analysis in st.session_state.analysis_history:
            st.markdown(f"**Query:** {analysis['query']}")
            st.markdown(f"**Timestamp:** {analysis['timestamp']}")
            st.markdown(f"**Result:** {analysis['result']}")
            st.markdown("---")
    else:
        st.info("No analysis history available.")


elif st.session_state.page == "Analysis":
    st.title("Run AI-Driven Analysis")

    if not st.session_state.get("uploaded_files"):
        st.warning("No files uploaded! Please upload relevant financial and market analysis documents.")
    else:
        analysis_types = [
            "Competitor Strategy",
            "Market Trends",
            "SWOT Analysis",
            "Comparative Analysis"
        ]
        analysis_type = st.selectbox("Select Analysis Type", analysis_types)

        # Get the stored domain
        selected_domain = st.session_state.get("selected_domain", "Unknown Domain")

        # Domain-Specific Context
        domain_mapping = {
            "Healthcare": "Market growth, regulatory impact, emerging technologies, and patient care models.",
            "Life Insurance": "Risk management, policy innovations, customer engagement strategies, and claim settlement processes.",
            "Mutual Funds": "Investment patterns, fund performance, risk management strategies, and economic impact.",
            "Lending or Diversified NBFCs": "Lending strategies, credit risk, interest rate structures, and financial stability."
        }

        prompt_templates = {
            "Competitor Strategy": "Analyze competitor strategies within the {selected_domain} sector. Extract insights on business models, revenue streams, expansion plans, pricing strategies, competitive advantages, innovation adoption, partnerships, and customer acquisition strategies. Provide findings in bullet points with each point on a new line.",
            "Market Trends": "Extract key market trends affecting the {selected_domain} industry. Identify growth trends, demand-supply shifts, regulatory impacts, consumer behavior changes, adoption of new technologies, macroeconomic influences, and emerging competitors. Present insights in bullet points with each point on a new line.",
            "SWOT Analysis": "Perform a SWOT analysis for the {selected_domain} sector. Identify strengths (market leadership, financial stability, customer loyalty), weaknesses (high costs, regulatory challenges, operational inefficiencies), opportunities (new market expansions, technological adoption, industry growth), and threats (economic downturn, policy changes, increasing competition). Structure insights with separate bullet points under each category.",
            "Comparative Analysis": "Compare competitors within the {selected_domain} industry based on financial growth, business strategy, market positioning, and innovation. Compare market share, business model differences, competitive advantages, customer engagement strategies, investment in technology, and expansion strategies. Structure insights in bullet points with separate sections for each competitor."
        }

        if selected_domain not in domain_mapping:
            st.error("Domain selection is missing. Please upload files again with a domain.")
        else:
            st.success(f"Using selected domain: **{selected_domain}**")

        if st.button("Run Analysis"):
            for pdf_file in st.session_state.uploaded_files:
                file_name = os.path.splitext(pdf_file.name)[0]
                folder_path = f"faiss_indexes/{file_name}"

                if not os.path.exists(folder_path):
                    st.error(f"FAISS index for {file_name} not found. Process the document first.")
                    continue

                st.subheader(f"Analyzing: {pdf_file.name}")
                query = prompt_templates[analysis_type].format(selected_domain=selected_domain)

                with st.spinner("Analyzing..."):
                    try:
                        # Call the analyze_document function already defined in app.py
                        report = analyze_document(file_name, query, f"Context: {{context}}")
                        st.text_area(f"Analysis Report for {pdf_file.name}", value=report, height=300)

                        # Store analysis result in session state
                        st.session_state.setdefault("analysis_history", []).append({
                            "query": query,
                            "domain": selected_domain,
                            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "result": report
                        })

                    except Exception as e:
                        st.error(f"Analysis failed for {pdf_file.name}: {str(e)}")



elif st.session_state.page == "Files":
    st.title("Uploaded Files")
    for file in st.session_state.uploaded_files:
        st.write(f"📄 {file.name}")

elif st.session_state.page == "Chatbot":
    st.title("Intel360 Chatbot 🤖")
    st.markdown("**Ask about competitor analysis, insights, and AI-generated reports!**")
    user_input = st.text_input("Ask me anything about competitor analysis:")
    if user_input:
        response = chatbot_response(user_input)
        st.session_state.chat_history.append({"query": user_input, "response": response})
        st.text_area("Chatbot Response", value=response, height=150)
    
    st.subheader("🗂️ Chat History")
    for chat in st.session_state.chat_history:
        st.write(f"**Q:** {chat['query']}")
        st.write(f"**A:** {chat['response']}")
        st.markdown("---")