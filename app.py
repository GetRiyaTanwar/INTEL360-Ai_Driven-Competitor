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

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Convert text into a searchable vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Load competitor analysis AI model
def get_analysis_chain():
    prompt_template = """
    Analyze the provided competitor strategy context and extract key insights, summarize findings, and identify emerging trends. 
    Deliver actionable intelligence using the following categories:
    - Market Positioning
    - Strengths
    - Weaknesses
    - Business Opportunities
    - Competitor Names
    - Product/Service Offerings
    - Geographic Regions
    - Industry Trends
    - Emerging Patterns

    Provide structured output including:
    1. Executive Summary
    2. Detailed Summary of Competitor Strategies
    3. Identified Emerging Trends
    4. Extracted Insights & Recommendations
    5. Supporting Data if available

    Context:
    {context}?
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Run competitor analysis
def analyze_competitor_strategies():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search("Analyze competitor strategies")
    chain = get_analysis_chain()
    response = chain({"input_documents": docs}, return_only_outputs=True)
    return response["output_text"]

# Generate competitor strategy visualization
def visualize_analysis():
    st.subheader("Competitor Insights Visualization")
    data = {"Market Positioning": 20, "Strengths": 30, "Weaknesses": 15, "Opportunities": 25, "Trends": 10}
    categories = list(data.keys())
    values = list(data.values())

    plt.figure(figsize=(8, 5))
    sns.barplot(x=categories, y=values, palette="coolwarm")
    plt.xlabel("Categories")
    plt.ylabel("Percentage of Focus")
    plt.title("Competitor Strategy Breakdown")
    st.pyplot(plt)

# Streamlit UI
def main():
    st.set_page_config(page_title="Intel360 – AI-Driven Competitor Analysis", layout="wide")
    st.title("Intel360: AI-Driven Competitor Analysis 📊")

    # Sidebar: Upload PDFs
    with st.sidebar:
        st.header("Upload Competitor Reports")
        st.write("Upload one or more competitor strategy documents (PDF format).")
        pdf_docs = st.file_uploader("Select PDF Files", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            with st.spinner("Extracting and processing text..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Documents processed successfully!")

    # Main Panel Layout
    col1, col2 = st.columns(2)

    # Competitor Analysis Button
    with col1:
        if st.button("Run Competitor Analysis"):
            st.subheader("Competitor Strategy Analysis Report")
            with st.spinner("Analyzing competitor strategies..."):
                report = analyze_competitor_strategies()
                st.text_area("Analysis Report:", value=report, height=300)

    # Visualization Button
    with col2:
        if st.button("Visualize Analysis"):
            visualize_analysis()

if __name__ == "__main__":
    main()
