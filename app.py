# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# import matplotlib.pyplot as plt
# import seaborn as sns
# from dotenv import load_dotenv

# # Load API key
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Extract text from PDFs
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# # Split text into manageable chunks
# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     return text_splitter.split_text(text)

# # Convert text into a searchable vector store
# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")

# # Load competitor analysis AI model
# def get_analysis_chain():
#     prompt_template = """
#     Analyze the provided competitor strategy context and extract key insights, summarize findings, and identify emerging trends. 
#     Deliver actionable intelligence using the following categories:
#     - Market Positioning
#     - Strengths
#     - Weaknesses
#     - Business Opportunities
#     - Competitor Names
#     - Product/Service Offerings
#     - Geographic Regions
#     - Industry Trends
#     - Emerging Patterns

#     Provide structured output including:
#     1. Executive Summary
#     2. Detailed Summary of Competitor Strategies
#     3. Identified Emerging Trends
#     4. Extracted Insights & Recommendations
#     5. Supporting Data if available

#     Context:
#     {context}?
#     """
#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
#     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# # Run competitor analysis
# def analyze_competitor_strategies():
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search("Analyze competitor strategies")
#     chain = get_analysis_chain()
#     response = chain({"input_documents": docs}, return_only_outputs=True)
#     return response["output_text"]

# # Generate competitor strategy visualization
# def visualize_analysis():
#     st.subheader("Competitor Insights Visualization")
#     data = {"Market Positioning": 20, "Strengths": 30, "Weaknesses": 15, "Opportunities": 25, "Trends": 10}
#     categories = list(data.keys())
#     values = list(data.values())

#     plt.figure(figsize=(8, 5))
#     sns.barplot(x=categories, y=values, palette="coolwarm")
#     plt.xlabel("Categories")
#     plt.ylabel("Percentage of Focus")
#     plt.title("Competitor Strategy Breakdown")
#     st.pyplot(plt)

# # Streamlit UI
# def main():
#     st.set_page_config(page_title="Intel360 – AI-Driven Competitor Analysis", layout="wide")
#     st.title("Intel360: AI-Driven Competitor Analysis 📊")

#     # Sidebar: Upload PDFs
#     with st.sidebar:
#         st.header("Upload Competitor Reports")
#         st.write("Upload one or more competitor strategy documents (PDF format).")
#         pdf_docs = st.file_uploader("Select PDF Files", accept_multiple_files=True)
        
#         if st.button("Submit & Process"):
#             with st.spinner("Extracting and processing text..."):
#                 raw_text = get_pdf_text(pdf_docs)
#                 text_chunks = get_text_chunks(raw_text)
#                 get_vector_store(text_chunks)
#                 st.success("Documents processed successfully!")

#     # Main Panel Layout
#     col1, col2 = st.columns(2)

#     # Competitor Analysis Button
#     with col1:
#         if st.button("Run Competitor Analysis"):
#             st.subheader("Competitor Strategy Analysis Report")
#             with st.spinner("Analyzing competitor strategies..."):
#                 report = analyze_competitor_strategies()
#                 st.text_area("Analysis Report:", value=report, height=300)

#     # Visualization Button
#     with col2:
#         if st.button("Visualize Analysis"):
#             visualize_analysis()

# if __name__ == "__main__":
#     main()




























# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# import matplotlib.pyplot as plt
# import seaborn as sns
# from dotenv import load_dotenv
# import pandas as pd
# from datetime import datetime
# import tempfile

# # Configure Streamlit page
# st.set_page_config(page_title="Intel360 – AI-Driven Competitor Analysis", layout="wide")

# # Load API Key
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Initialize session states
# if 'uploaded_files' not in st.session_state:
#     st.session_state.uploaded_files = []
# if 'processed_files' not in st.session_state:
#     st.session_state.processed_files = set()
# if 'temp_dir' not in st.session_state:
#     st.session_state.temp_dir = tempfile.mkdtemp()
# if 'settings' not in st.session_state:
#     st.session_state.settings = {
#         'theme': 'Light',
#         'language': 'English',
#         'notification': True,
#         'auto_analysis': False
#     }
# if 'last_analysis' not in st.session_state:
#     st.session_state.last_analysis = None

# # Existing helper functions (keep them as is)
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     return text_splitter.split_text(text)

# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")

# def analyze_competitor_strategies():
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search("Analyze competitor strategies")
#     chain = get_analysis_chain()
#     response = chain({"input_documents": docs}, return_only_outputs=True)
#     return response["output_text"]

# def get_analysis_chain():
#     prompt_template = """
#     Analyze the provided competitor strategy context and extract key insights, summarize findings, and identify emerging trends. 
#     Deliver actionable intelligence using:
#     - Market Positioning
#     - Strengths
#     - Weaknesses
#     - Business Opportunities
#     - Industry Trends
    
#     Provide:
#     1. Executive Summary
#     2. Emerging Trends
#     3. Insights & Recommendations
    
#     Context:
#     {context}?
#     """
#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
#     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# # Enhanced Settings
# def show_settings():
#     st.title("Settings ⚙️")
    
#     # General Settings
#     st.subheader("General Settings")
#     st.session_state.settings['theme'] = st.selectbox(
#         "Theme",
#         options=['Light', 'Dark'],
#         index=0 if st.session_state.settings['theme'] == 'Light' else 1
#     )
#     st.session_state.settings['language'] = st.selectbox(
#         "Language",
#         options=['English', 'Spanish', 'French'],
#         index=0
#     )
    
#     # Notification Settings
#     st.subheader("Notifications")
#     st.session_state.settings['notification'] = st.toggle(
#         "Enable Notifications",
#         value=st.session_state.settings['notification']
#     )
#     st.session_state.settings['auto_analysis'] = st.toggle(
#         "Auto-analyze new documents",
#         value=st.session_state.settings['auto_analysis']
#     )
    
#     if st.button("Save Settings"):
#         st.success("Settings saved successfully!")

# # Enhanced Analytics
# def show_analytics():
#     st.title("Advanced Analytics 📈")
    
#     tabs = st.tabs(["Competitor Analysis", "Market Trends", "SWOT Analysis"])
    
#     with tabs[0]:
#         st.subheader("Competitor Strategy Analysis")
#         if st.button("Generate Analysis"):
#             if len(st.session_state.uploaded_files) > 0:
#                 with st.spinner("Analyzing competitor strategies..."):
#                     try:
#                         report = analyze_competitor_strategies()
#                         st.text_area("Analysis Report:", value=report, height=300)
#                         st.session_state.last_analysis = datetime.now().strftime("%Y-%m-%d %H:%M")
#                     except Exception as e:
#                         st.error(f"Error generating analysis: {str(e)}")
#             else:
#                 st.warning("Please upload documents first to perform analysis.")
    
#     with tabs[1]:
#         st.subheader("Market Trends")
#         # Sample data for demonstration
#         trend_data = pd.DataFrame({
#             'Month': ['Jan', 'Feb', 'Mar', 'Apr'],
#             'Market Share': [30, 35, 32, 38],
#             'Growth Rate': [0, 16.7, -8.6, 18.8]
#         })
        
#         # Market share trend
#         st.line_chart(trend_data.set_index('Month')['Market Share'])
        
#         # Growth rate analysis
#         st.subheader("Monthly Growth Rate")
#         st.bar_chart(trend_data.set_index('Month')['Growth Rate'])
    
#     with tabs[2]:
#         st.subheader("SWOT Analysis")
#         col1, col2 = st.columns(2)
#         with col1:
#             st.markdown("### Strengths")
#             st.write("• Market Leadership")
#             st.write("• Strong R&D")
#             st.write("• Brand Recognition")
            
#             st.markdown("### Opportunities")
#             st.write("• Emerging Markets")
#             st.write("• Digital Transformation")
#             st.write("• Strategic Partnerships")
        
#         with col2:
#             st.markdown("### Weaknesses")
#             st.write("• Cost Structure")
#             st.write("• Legacy Systems")
#             st.write("• Regional Limitations")
            
#             st.markdown("### Threats")
#             st.write("• New Competitors")
#             st.write("• Regulatory Changes")
#             st.write("• Market Volatility")

# def main():
#     # Sidebar
#     with st.sidebar:
#         st.markdown("## ⚙️ Admin Dashboard")
#         st.markdown("---")
#         add_data_button = st.button("📂 Add Data")
#         dashboard_button = st.button("📊 Dashboard")
#         files_button = st.button("📁 Files")
#         analytics_button = st.button("📈 Analytics")
#         settings_button = st.button("⚙️ Settings")
    
#     # Main content
#     if add_data_button:
#         st.title("Upload Competitor Reports 📄")
#         pdf_docs = st.file_uploader("Select PDF Files", accept_multiple_files=True)
        
#         if pdf_docs:
#             st.session_state.uploaded_files.extend(pdf_docs)
        
#         if st.button("Submit & Process"):
#             if pdf_docs:
#                 with st.spinner("Processing documents..."):
#                     raw_text = get_pdf_text(pdf_docs)
#                     text_chunks = get_text_chunks(raw_text)
#                     get_vector_store(text_chunks)
#                     st.success("Documents processed successfully!")
#             else:
#                 st.warning("Please upload at least one file.")
    
#     elif dashboard_button:
#         st.title("Dashboard 📊")
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             st.metric("Documents Analyzed", len(st.session_state.uploaded_files))
#         with col2:
#             st.metric("Last Analysis", st.session_state.last_analysis or "Never")
#         with col3:
#             st.metric("Active Competitors", len(st.session_state.uploaded_files))
    
#     elif files_button:
#         st.title("Uploaded Files 📂")
#         if len(st.session_state.uploaded_files) > 0:
#             for file in st.session_state.uploaded_files:
#                 col1, col2 = st.columns([3,1])
#                 with col1:
#                     st.write(f"📄 {file.name}")
#                 with col2:
#                     st.download_button("Download", file, file_name=file.name)
#         else:
#             st.warning("No files uploaded yet.")
    
#     elif analytics_button:
#         show_analytics()
    
#     elif settings_button:
#         show_settings()

# if __name__ == "__main__":
#     main()





























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
from datetime import datetime
import pandas as pd

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Session State Initialization
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'settings' not in st.session_state:
    st.session_state.settings = {'theme': 'Light', 'language': 'English'}

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

# Load AI model for analysis
def get_analysis_chain(prompt_template):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Generalized analysis function
def analyze_documents(query, prompt_template):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(query)
    chain = get_analysis_chain(prompt_template)
    response = chain({"input_documents": docs}, return_only_outputs=True)
    st.session_state.analysis_history.append({"query": query, "result": response["output_text"], "timestamp": datetime.now()})
    return response["output_text"]

# Visualization
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

# Settings
def show_settings():
    st.header("Settings ⚙️")
    theme = st.selectbox("Theme", options=["Light", "Dark"], index=0 if st.session_state.settings['theme'] == 'Light' else 1)
    language = st.selectbox("Language", options=["English", "Spanish", "French"], index=0)
    if st.button("Save Settings"):
        st.session_state.settings['theme'] = theme
        st.session_state.settings['language'] = language
        st.success("Settings saved successfully!")

# Main UI
st.set_page_config(page_title="Intel360 – AI-Driven Competitor Analysis", layout="wide")
st.title("Intel360: AI-Driven Competitor Analysis 📊")



with st.sidebar:
    st.header("Navigation")
    
    # Initialize session state for page navigation
    if 'page' not in st.session_state:
        st.session_state.page = "Upload Files"

    # Navigation buttons
    if st.button("Dashboard"):
        st.session_state.page = "Dashboard"
    if st.button("Upload Files"):
        st.session_state.page = "Upload Files"
    if st.button("Analysis"):
        st.session_state.page = "Analysis"
    if st.button("Files"):
        st.session_state.page = "Files"
    if st.button("Settings"):
        st.session_state.page = "Settings"

# Display content based on the selected page
if st.session_state.page == "Upload Files":
    st.title("Upload Files Page")
elif st.session_state.page == "Dashboard":
    st.title("Dashboard Page")
elif st.session_state.page == "Analysis":
    st.title("Analysis Page")
elif st.session_state.page == "Files":
    st.title("Files Page")
elif st.session_state.page == "Settings":
    st.title("Settings Page")


# Pages
if st.session_state.page == "Upload Files":
    st.header("Upload Competitor Reports")
    pdf_docs = st.file_uploader("Select PDF Files", accept_multiple_files=True)
    if pdf_docs:
        st.session_state.uploaded_files.extend(pdf_docs)
    if st.button("Submit & Process"):
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        st.success("Documents processed successfully!")

elif st.session_state.page == "Dashboard":
    st.header("Dashboard")
    st.metric("Documents Analyzed", len(st.session_state.uploaded_files))
    st.metric("Total Analyses", len(st.session_state.analysis_history))
    st.subheader("Analysis History")
    for history in st.session_state.analysis_history:
        st.write(f"**Query:** {history['query']} | **Date:** {history['timestamp']}")
        st.text_area("Result", value=history['result'], height=150)

elif st.session_state.page == "Analysis":
    st.header("Document Analysis")
    analysis_type = st.selectbox("Select Analysis Type", ["Competitor Strategy", "Market Trends", "SWOT Analysis"])

    if st.button("Run Analysis"):
        if analysis_type == "Competitor Strategy":
            query = "Analyze competitor strategies"
            prompt_template = """
    Analyze the provided competitor strategy context and extract key insights, summarize findings, and identify emerging trends using Generative AI.  
    Process and analyze competitor-related documents such as financial reports, market research papers, and business strategy documents in PDF format.  

    If a single company's document is provided, generate a *comprehensive market analysis* for that company.  
    If multiple companies' documents are uploaded, perform a *comparative analysis* to evaluate market dynamics and competitive positioning.  

    Deliver actionable intelligence using the following categories:  
    - *Market Positioning* – Identify where the competitor stands in the industry.  
    - *Strengths* – Highlight competitive advantages, differentiators, and success factors.  
    - *Weaknesses* – Detect gaps, vulnerabilities, and areas for improvement.  
    - *Business Opportunities* – Explore potential growth areas, partnerships, and market gaps.  
    - *Competitor Names* – Recognize entities and major market players.  
    - *Product/Service Offerings* – List key products/services and their unique value propositions.  
    - *Geographic Regions* – Identify operational regions and expansion markets.  
    - *Industry Trends* – Capture current trends shaping the competitive landscape.  
    - *Emerging Patterns* – Detect recurring themes, disruptions, and strategic moves.  
    - *Comparative Insights (if multiple companies are uploaded)* – Highlight key differences and similarities between competitors.  

    Provide structured output including:  
    1. *Executive Summary* – High-level insights summarizing key competitor strategies.  
    2. *Market Entry Feasibility Report* – Evaluation of potential success in entering the market based on competitive insights.  
    3. *Detailed Competitor Analysis* – Breakdown of individual companies’ strategies and industry position.  
    4. *Identified Emerging Trends* – Highlight upcoming shifts and patterns in the industry.  
    5. *Comparative Insights & Competitive Benchmarking* – Directly compare multiple companies’ strategies, strengths, weaknesses, and market advantages.  
    6. *Extracted Insights & Recommendations* – Provide strategic guidance based on findings.  
    7. *Supporting Data & Evidence* – Include extracted statistics, tables, and citations from the document(s).  

    Context:  
    {context}?

            """
        elif analysis_type == "Market Trends":
            query = "Analyze in relation to market trends"
            prompt_template = """
            Compare the document context with current market trends and provide insights.
            Context:
            {context}
            """
        else:
            query = "SWOT Analysis"
            prompt_template = """
            Conduct a SWOT analysis based on the document context.
            Context:
            {context}
            """

        with st.spinner("Analyzing..."):
            report = analyze_documents(query, prompt_template)
            st.text_area("Analysis Report", value=report, height=300)

elif st.session_state.page == "Files":
    st.header("Uploaded Files")
    for file in st.session_state.uploaded_files:
        st.write(f"📄 {file.name}")

elif st.session_state.page == "Settings":
    show_settings()
















































# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# import matplotlib.pyplot as plt
# import seaborn as sns
# from dotenv import load_dotenv
# import pandas as pd
# from datetime import datetime
# import tempfile

# # Load API key
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Extract text from PDFs
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# # Split text into manageable chunks
# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     return text_splitter.split_text(text)

# # Convert text into a searchable vector store
# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")

# # Load competitor analysis AI model
# def get_analysis_chain():
#     prompt_template = """
#     Analyze the provided competitor strategy context and extract key insights, summarize findings, and identify emerging trends. 
#     Deliver actionable intelligence using the following categories:
#     - Market Positioning
#     - Strengths
#     - Weaknesses
#     - Business Opportunities
#     - Competitor Names
#     - Product/Service Offerings
#     - Geographic Regions
#     - Industry Trends
#     - Emerging Patterns

#     Provide structured output including:
#     1. Executive Summary
#     2. Detailed Summary of Competitor Strategies
#     3. Identified Emerging Trends
#     4. Extracted Insights & Recommendations
#     5. Supporting Data if available

#     Context:
#     {context}?
#     """
#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
#     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# # Run competitor analysis
# def analyze_competitor_strategies():
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search("Analyze competitor strategies")
#     chain = get_analysis_chain()
#     response = chain({"input_documents": docs}, return_only_outputs=True)
#     return response["output_text"]

# # Generate competitor strategy visualization
# def visualize_analysis():
#     st.subheader("Competitor Insights Visualization")
#     data = {"Market Positioning": 20, "Strengths": 30, "Weaknesses": 15, "Opportunities": 25, "Trends": 10}
#     categories = list(data.keys())
#     values = list(data.values())

#     plt.figure(figsize=(8, 5))
#     sns.barplot(x=categories, y=values, palette="coolwarm")
#     plt.xlabel("Categories")
#     plt.ylabel("Percentage of Focus")
#     plt.title("Competitor Strategy Breakdown")
#     st.pyplot(plt)

# # Streamlit UI
# def main():
#     st.set_page_config(page_title="Intel360 – AI-Driven Competitor Analysis", layout="wide")
#     st.title("Intel360: AI-Driven Competitor Analysis 📊")

#     # Sidebar: Navigation Buttons
#     with st.sidebar:
#         st.header("Navigation")
#         if st.button("Upload Files"):
#             page = "upload"
#         elif st.button("Dashboard"):
#             page = "dashboard"
#         elif st.button("Analytics"):
#             page = "analytics"
#         elif st.button("Files"):
#             page = "files"
#         elif st.button("Settings"):
#             page = "settings"
#         else:
#             page = "dashboard"

#     # Main Panel Layout
#     if page == "upload":
#         st.header("Upload Competitor Reports")
#         pdf_docs = st.file_uploader("Select PDF Files", accept_multiple_files=True)
#         if st.button("Submit & Process"):
#             with st.spinner("Extracting and processing text..."):
#                 raw_text = get_pdf_text(pdf_docs)
#                 text_chunks = get_text_chunks(raw_text)
#                 get_vector_store(text_chunks)
#                 st.success("Documents processed successfully!")

#     elif page == "dashboard":
#         st.header("Dashboard")
#         st.write("Document analysis history will appear here.")

#     elif page == "analytics":
#         st.header("Analytics")
#         if st.button("Run Competitor Analysis"):
#             st.subheader("Competitor Strategy Analysis Report")
#             with st.spinner("Analyzing competitor strategies..."):
#                 report = analyze_competitor_strategies()
#                 st.text_area("Analysis Report:", value=report, height=300)

#         if st.button("Visualize Analysis"):
#             visualize_analysis()

#     elif page == "files":
#         st.header("Uploaded Files")
#         st.write("List of uploaded files during this session.")

#     elif page == "settings":
#         st.header("Settings")
#         theme = st.selectbox("Select Theme", ["Light", "Dark"])
#         language = st.selectbox("Select Language", ["English", "Spanish", "French"])
#         if st.button("Save Settings"):
#             st.success("Settings saved successfully!")

# if __name__ == "__main__":
#     main()


















# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# import matplotlib.pyplot as plt
# import seaborn as sns
# from dotenv import load_dotenv
# import pandas as pd
# from datetime import datetime

# # Load API key
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Initialize session state
# if 'page' not in st.session_state:
#     st.session_state.page = 'Dashboard'

# # Extract text from PDFs
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# # Split text into manageable chunks
# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     return text_splitter.split_text(text)

# # Convert text into a searchable vector store
# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")

# # Load competitor analysis AI model
# def get_analysis_chain():
#     prompt_template = """
#     Analyze the provided competitor strategy context and extract key insights, summarize findings, and identify emerging trends. 
#     Deliver actionable intelligence using the following categories:
#     - Market Positioning
#     - Strengths
#     - Weaknesses
#     - Business Opportunities
#     - Competitor Names
#     - Product/Service Offerings
#     - Geographic Regions
#     - Industry Trends
#     - Emerging Patterns

#     Provide structured output including:
#     1. Executive Summary
#     2. Detailed Summary of Competitor Strategies
#     3. Identified Emerging Trends
#     4. Extracted Insights & Recommendations
#     5. Supporting Data if available

#     Context:
#     {context}?
#     """
#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
#     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# # Run competitor analysis
# def analyze_competitor_strategies():
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search("Analyze competitor strategies")
#     chain = get_analysis_chain()
#     response = chain({"input_documents": docs}, return_only_outputs=True)
#     return response["output_text"]

# # Generate competitor strategy visualization
# def visualize_analysis():
#     st.subheader("Competitor Insights Visualization")
#     data = {"Market Positioning": 20, "Strengths": 30, "Weaknesses": 15, "Opportunities": 25, "Trends": 10}
#     categories = list(data.keys())
#     values = list(data.values())

#     plt.figure(figsize=(8, 5))
#     sns.barplot(x=categories, y=values, palette="coolwarm")
#     plt.xlabel("Categories")
#     plt.ylabel("Percentage of Focus")
#     plt.title("Competitor Strategy Breakdown")
#     st.pyplot(plt)

# # Streamlit UI
# def main():
#     st.set_page_config(page_title="Intel360 – AI-Driven Competitor Analysis", layout="wide")
#     st.title("Intel360: AI-Driven Competitor Analysis 📊")

#     # Sidebar: Navigation
#     with st.sidebar:
#         st.header("Navigation")
#         if st.button("Dashboard"):
#             st.session_state.page = 'Dashboard'
#         if st.button("Upload Files"):
#             st.session_state.page = 'Upload Files'
#         if st.button("Analytics"):
#             st.session_state.page = 'Analytics'
#         if st.button("Files"):
#             st.session_state.page = 'Files'
#         if st.button("Settings"):
#             st.session_state.page = 'Settings'



#         # Main Panel Layout
#     if st.session_state.page == 'Upload Files':
#         st.header("Upload Competitor Reports")
#         pdf_docs = st.file_uploader("Select PDF Files", accept_multiple_files=True)
#         if st.button("Submit & Process"):
#             with st.spinner("Extracting and processing text..."):
#                 raw_text = get_pdf_text(pdf_docs)
#                 text_chunks = get_text_chunks(raw_text)
#                 get_vector_store(text_chunks)
#                 st.success("Documents processed successfully!")

#     elif st.session_state.page == 'Dashboard':
#         st.header("Dashboard")
#         st.write("Document analysis history will appear here.")





#     # elif st.session_state.page == 'Analytics':
#     #     st.header("Document Analytics")
#     # analysis_type = st.selectbox("Select Analysis Type", ["Competitor Strategy", "Market Trends", "SWOT Analysis"])

#     # if st.button("Run Analysis"):
#     #     if analysis_type == "Competitor Strategy":
#     #         query = "Analyze competitor strategies"
#     #         prompt_template = """
#     #         Provide a competitor strategy analysis focusing on market positioning, strengths, weaknesses, and emerging trends.
#     #         Context:
#     #         {context}
#     #         """
#     #     elif analysis_type == "Market Trends":
#     #         query = "Analyze in relation to market trends"
#     #         prompt_template = """
#     #         Compare the document context with current market trends and provide insights.
#     #         Context:
#     #         {context}
#     #         """
#     #     else:
#     #         query = "SWOT Analysis"
#     #         prompt_template = """
#     #         Conduct a SWOT analysis based on the document context.
#     #         Context:
#     #         {context}
#     #         """

#     #     with st.spinner("Analyzing..."):
#     #         report = analyze_documents(query, prompt_template)
#     #         st.text_area("Analysis Report", value=report, height=300)






    
    
#     # elif st.session_state.page == 'Analytics':
#     #     st.header("Analytics")
#     #     if st.button("Run Competitor Analysis"):
#     #         st.subheader("Competitor Strategy Analysis Report")
#     #         with st.spinner("Analyzing competitor strategies..."):
#     #             report = analyze_competitor_strategies()
#     #             st.text_area("Analysis Report:", value=report, height=300)

#     #     if st.button("Visualize Analysis"):
#     #         visualize_analysis()











#     elif st.session_state.page == 'Analytics':
#          st.header("Document Analytics")

#     analysis_options = {
#         "Competitor Strategy": {
#             "query": "Analyze competitor strategies",
#             "prompt": """
#                 Provide a competitor strategy analysis focusing on market positioning, strengths, weaknesses, and emerging trends.
#                 Context:
#                 {context}
#             """
#         },
#         "Market Trends": {
#             "query": "Analyze in relation to market trends",
#             "prompt": """
#                 Compare the document context with current market trends and provide insights.
#                 Context:
#                 {context}
#             """
#         },
#         "SWOT Analysis": {
#             "query": "SWOT Analysis",
#             "prompt": """
#                 Conduct a SWOT analysis based on the document context.
#                 Context:
#                 {context}
#             """
#         }
#     }

#     analysis_type = st.selectbox("Select Analysis Type", list(analysis_options.keys()))

#     if st.button("Run Analysis"):
#         selected_analysis = analysis_options[analysis_type]
#         with st.spinner("Analyzing..."):
#             report = analyze_documents(selected_analysis["query"], selected_analysis["prompt"])
#             st.text_area("Analysis Report", value=report, height=300)










#     elif st.session_state.page == 'Files':
#         st.header("Uploaded Files")
#         st.write("List of uploaded files during this session.")

#     elif st.session_state.page == 'Settings':
#         st.header("Settings")
#         theme = st.selectbox("Select Theme", ["Light", "Dark"])
#         language = st.selectbox("Select Language", ["English", "Spanish", "French"])
#         if st.button("Save Settings"):
#             st.success("Settings saved successfully!")

# if __name__ == "__main__":
#     main()    















