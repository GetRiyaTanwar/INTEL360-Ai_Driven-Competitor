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
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime
import tempfile

# Configure Streamlit page
st.set_page_config(page_title="Intel360 – AI-Driven Competitor Analysis", layout="wide")

# Load API Key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize session states
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()
if 'settings' not in st.session_state:
    st.session_state.settings = {
        'theme': 'Light',
        'language': 'English',
        'notification': True,
        'auto_analysis': False
    }
if 'last_analysis' not in st.session_state:
    st.session_state.last_analysis = None

# Existing helper functions (keep them as is)
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def analyze_competitor_strategies():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search("Analyze competitor strategies")
    chain = get_analysis_chain()
    response = chain({"input_documents": docs}, return_only_outputs=True)
    return response["output_text"]

def get_analysis_chain():
    prompt_template = """
    Analyze the provided competitor strategy context and extract key insights, summarize findings, and identify emerging trends. 
    Deliver actionable intelligence using:
    - Market Positioning
    - Strengths
    - Weaknesses
    - Business Opportunities
    - Industry Trends
    
    Provide:
    1. Executive Summary
    2. Emerging Trends
    3. Insights & Recommendations
    
    Context:
    {context}?
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Enhanced Settings
def show_settings():
    st.title("Settings ⚙️")
    
    # General Settings
    st.subheader("General Settings")
    st.session_state.settings['theme'] = st.selectbox(
        "Theme",
        options=['Light', 'Dark'],
        index=0 if st.session_state.settings['theme'] == 'Light' else 1
    )
    st.session_state.settings['language'] = st.selectbox(
        "Language",
        options=['English', 'Spanish', 'French'],
        index=0
    )
    
    # Notification Settings
    st.subheader("Notifications")
    st.session_state.settings['notification'] = st.toggle(
        "Enable Notifications",
        value=st.session_state.settings['notification']
    )
    st.session_state.settings['auto_analysis'] = st.toggle(
        "Auto-analyze new documents",
        value=st.session_state.settings['auto_analysis']
    )
    
    if st.button("Save Settings"):
        st.success("Settings saved successfully!")

# Enhanced Analytics
def show_analytics():
    st.title("Advanced Analytics 📈")
    
    tabs = st.tabs(["Competitor Analysis", "Market Trends", "SWOT Analysis"])
    
    with tabs[0]:
        st.subheader("Competitor Strategy Analysis")
        if st.button("Generate Analysis"):
            if len(st.session_state.uploaded_files) > 0:
                with st.spinner("Analyzing competitor strategies..."):
                    try:
                        report = analyze_competitor_strategies()
                        st.text_area("Analysis Report:", value=report, height=300)
                        st.session_state.last_analysis = datetime.now().strftime("%Y-%m-%d %H:%M")
                    except Exception as e:
                        st.error(f"Error generating analysis: {str(e)}")
            else:
                st.warning("Please upload documents first to perform analysis.")
    
    with tabs[1]:
        st.subheader("Market Trends")
        # Sample data for demonstration
        trend_data = pd.DataFrame({
            'Month': ['Jan', 'Feb', 'Mar', 'Apr'],
            'Market Share': [30, 35, 32, 38],
            'Growth Rate': [0, 16.7, -8.6, 18.8]
        })
        
        # Market share trend
        st.line_chart(trend_data.set_index('Month')['Market Share'])
        
        # Growth rate analysis
        st.subheader("Monthly Growth Rate")
        st.bar_chart(trend_data.set_index('Month')['Growth Rate'])
    
    with tabs[2]:
        st.subheader("SWOT Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Strengths")
            st.write("• Market Leadership")
            st.write("• Strong R&D")
            st.write("• Brand Recognition")
            
            st.markdown("### Opportunities")
            st.write("• Emerging Markets")
            st.write("• Digital Transformation")
            st.write("• Strategic Partnerships")
        
        with col2:
            st.markdown("### Weaknesses")
            st.write("• Cost Structure")
            st.write("• Legacy Systems")
            st.write("• Regional Limitations")
            
            st.markdown("### Threats")
            st.write("• New Competitors")
            st.write("• Regulatory Changes")
            st.write("• Market Volatility")

def main():
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Admin Dashboard")
        st.markdown("---")
        add_data_button = st.button("📂 Add Data")
        dashboard_button = st.button("📊 Dashboard")
        files_button = st.button("📁 Files")
        analytics_button = st.button("📈 Analytics")
        settings_button = st.button("⚙️ Settings")
    
    # Main content
    if add_data_button:
        st.title("Upload Competitor Reports 📄")
        pdf_docs = st.file_uploader("Select PDF Files", accept_multiple_files=True)
        
        if pdf_docs:
            st.session_state.uploaded_files.extend(pdf_docs)
        
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing documents..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Documents processed successfully!")
            else:
                st.warning("Please upload at least one file.")
    
    elif dashboard_button:
        st.title("Dashboard 📊")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Documents Analyzed", len(st.session_state.uploaded_files))
        with col2:
            st.metric("Last Analysis", st.session_state.last_analysis or "Never")
        with col3:
            st.metric("Active Competitors", len(st.session_state.uploaded_files))
    
    elif files_button:
        st.title("Uploaded Files 📂")
        if len(st.session_state.uploaded_files) > 0:
            for file in st.session_state.uploaded_files:
                col1, col2 = st.columns([3,1])
                with col1:
                    st.write(f"📄 {file.name}")
                with col2:
                    st.download_button("Download", file, file_name=file.name)
        else:
            st.warning("No files uploaded yet.")
    
    elif analytics_button:
        show_analytics()
    
    elif settings_button:
        show_settings()

if __name__ == "__main__":
    main()