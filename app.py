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

# ✅ Configure Streamlit page
st.set_page_config(page_title="Intel360 – AI-Driven Competitor Analysis", layout="wide")

# ✅ Load API Key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ✅ Sidebar: Admin Dashboard UI
with st.sidebar:
    st.markdown("## ⚙️ Admin Dashboard")
    st.markdown("---")
    st.markdown("### Menu")
    st.button("📂 Add Data")
    st.button("📊 Dashboard")
    st.button("📁 Files")
    st.button("📈 Analytics")
    st.button("⚙️ Settings")
    st.button("🚪 Logout")
    st.markdown("---")

# ✅ Extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# ✅ Split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# ✅ Convert text into a searchable vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# ✅ Load AI analysis model
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

# ✅ Run competitor analysis
def analyze_competitor_strategies():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search("Analyze competitor strategies")
    chain = get_analysis_chain()
    response = chain({"input_documents": docs}, return_only_outputs=True)
    return response["output_text"]

# ✅ Extract insights for visualization dynamically
def extract_insights_for_visualization():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search("Extract competitor insights for visualization")
    chain = get_analysis_chain()
    response = chain({"input_documents": docs}, return_only_outputs=True)
    
    # ✅ Process extracted insights into structured data
    extracted_text = response["output_text"]
    
    categories = {
        "Market Positioning": 0,
        "Strengths": 0,
        "Weaknesses": 0,
        "Opportunities": 0,
        "Trends": 0
    }
    
    # ✅ Example: Extract numerical values dynamically
    for category in categories.keys():
        if category.lower() in extracted_text.lower():
            categories[category] += 20  # Placeholder logic for now
    
    return categories

# ✅ Generate dynamic visualizations based on AI insights
def visualize_analysis():
    st.subheader("Competitor Insights Visualization")
    insights = extract_insights_for_visualization()

    categories = list(insights.keys())
    values = list(insights.values())

    plt.figure(figsize=(8, 5))
    sns.barplot(x=categories, y=values, palette="coolwarm")
    plt.xlabel("Categories")
    plt.ylabel("Percentage of Focus")
    plt.title("Competitor Strategy Breakdown")
    st.pyplot(plt)

# ✅ Streamlit UI
def main():
    st.title("Intel360: AI-Driven Competitor Analysis 📊")

    # Upload PDFs
    st.subheader("Upload Competitor Reports")
    pdf_docs = st.file_uploader("Select PDF Files", accept_multiple_files=True)
    
    if st.button("Submit & Process"):
        with st.spinner("Extracting and processing text..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("Documents processed successfully!")

    # Analysis & Visualization
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Run Competitor Analysis"):
            st.subheader("Competitor Strategy Analysis Report")
            with st.spinner("Analyzing competitor strategies..."):
                report = analyze_competitor_strategies()
                st.text_area("Analysis Report:", value=report, height=300)

    with col2:
        if st.button("Visualize Analysis"):
            visualize_analysis()

if __name__ == "__main__":
    main()
