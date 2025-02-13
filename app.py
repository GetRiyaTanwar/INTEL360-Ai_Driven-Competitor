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
# from datetime import datetime
# import pandas as pd
# from PyPDF2 import PdfReader
# import io
# from io import BytesIO
# import re
# from langchain.chains import StuffDocumentsChain
# from langchain.chains.question_answering import load_qa_chain

# # Load API key
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Session State Initialization
# if 'uploaded_files' not in st.session_state:
#     st.session_state.uploaded_files = []
# if 'analysis_history' not in st.session_state:
#     st.session_state.analysis_history = []
# if 'settings' not in st.session_state:
#     st.session_state.settings = {'theme': 'Light', 'language': 'English'}

# # Extract text from PDFs
# # def get_pdf_text(pdf):
# #     # Ensure pdf is a BytesIO object
# #     pdf_stream = io.BytesIO(pdf.read())  
# #     pdf_reader = PdfReader(pdf_stream)  

# #     text = ""
# #     for page in pdf_reader.pages:
# #         text += page.extract_text() or ""

# #     return text







# def get_pdf_text(pdf_file):
#     """Extracts text from a PDF file."""
#     from PyPDF2 import PdfReader

#     if pdf_file.size == 0:
#         return None  # Prevents processing an empty file

#     pdf_stream = BytesIO(pdf_file.getvalue())  # Convert to BytesIO object
#     pdf_reader = PdfReader(pdf_stream)

#     text = ""
#     for page in pdf_reader.pages:
#         text += page.extract_text() or ""  # Extract text safely

#     return text.strip() if text else None  # Return None if no text found









# # Split text into manageable chunks
# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     return text_splitter.split_text(text)

# # Convert text into a searchable vector store
# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")




# # def generate_comparative_analysis(uploaded_files):
# #     """
# #     This function takes a list of uploaded documents, extracts key insights, and creates
# #     a comparative analysis in a tabular format.
# #     """
# #     comparative_data = []

# #     for pdf_file in uploaded_files:
# #         context = get_pdf_text(pdf_file)  # Extract text
# #         query = "Extract competitor analysis data"
# #         prompt_template = """
# #         Analyze the document and extract key competitor insights including:
# #         - Market Position
# #         - Strengths
# #         - Weaknesses
# #         - Business Opportunities
# #         - Product Offerings
# #         - Geographic Reach

# #         Context:
# #         {context}
# #         """

# #         analysis_result = analyze_documents(query, prompt_template)

# #         comparative_data.append({
# #             "Document": pdf_file.name,
# #             "Market Position": extract_market_position(analysis_result),
# #             "Strengths": extract_strengths(analysis_result),
# #             "Weaknesses": extract_weaknesses(analysis_result),
# #             "Opportunities": extract_opportunities(analysis_result),
# #             "Product Offerings": extract_product_offerings(analysis_result),
# #             "Geographic Reach": extract_geographic_reach(analysis_result)
# #         })

# #     # Convert results into a DataFrame for easy visualization
# #     return pd.DataFrame(comparative_data)





# # Load AI model for analysis
# # def get_analysis_chain(prompt_template):
# #     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
# #     prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
# #     return load_qa_chain(model, chain_type="stuff", prompt=prompt)












# # Generalized analysis function
# def analyze_documents(query, prompt_template):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search(query)
#     chain = get_analysis_chain(prompt_template)
#     response = chain({"input_documents": docs}, return_only_outputs=True)
#     st.session_state.analysis_history.append({"query": query, "result": response["output_text"], "timestamp": datetime.now()})
#     return response["output_text"]

# # Visualization
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












# def generate_comparative_analysis(reports):
#     """
#     Generates a comparative analysis table by extracting key competitor insights
#     from multiple reports and organizing them into a structured format.
    
#     :param reports: List of dictionaries with extracted reports, each containing:
#                     {"Document": <file_name>, "Report": <analysis_result>}
#     :return: DataFrame with comparative insights
#     """
#     comparative_data = []

#     for report in reports:
#         document_name = report["Document"]
#         analysis_result = report["Report"]

#         # Extract key insights using helper functions (or regex if needed)
#         comparative_data.append({
#             "Document": document_name,
#             "Market Position": extract_market_position(analysis_result),
#             "Strengths": extract_strengths(analysis_result),
#             "Weaknesses": extract_weaknesses(analysis_result),
#             "Opportunities": extract_opportunities(analysis_result),
#             "Product Offerings": extract_product_offerings(analysis_result),
#             "Geographic Reach": extract_geographic_reach(analysis_result)
#         })

#     # Convert results into a DataFrame for easy visualization
#     return pd.DataFrame(comparative_data)


# def extract_market_position(analysis_result):
#     """Extracts market position details from the analysis report."""
#     match = re.search(r"Market Position:\s*(.+)", analysis_result, re.IGNORECASE)
#     return match.group(1).strip() if match else "Not Found"

# def extract_strengths(analysis_result):
#     """Extracts strengths from the analysis report."""
#     match = re.search(r"Strengths:\s*(.+)", analysis_result, re.IGNORECASE)
#     return match.group(1).strip() if match else "Not Found"

# def extract_weaknesses(analysis_result):
#     """Extracts weaknesses from the analysis report."""
#     match = re.search(r"Weaknesses:\s*(.+)", analysis_result, re.IGNORECASE)
#     return match.group(1).strip() if match else "Not Found"

# def extract_opportunities(analysis_result):
#     """Extracts business opportunities from the analysis report."""
#     match = re.search(r"Opportunities:\s*(.+)", analysis_result, re.IGNORECASE)
#     return match.group(1).strip() if match else "Not Found"

# def extract_product_offerings(analysis_result):
#     """Extracts product offerings from the analysis report."""
#     match = re.search(r"Product Offerings:\s*(.+)", analysis_result, re.IGNORECASE)
#     return match.group(1).strip() if match else "Not Found"

# def extract_geographic_reach(analysis_result):
#     """Extracts geographic reach from the analysis report."""
#     match = re.search(r"Geographic Reach:\s*(.+)", analysis_result, re.IGNORECASE)
#     return match.group(1).strip() if match else "Not Found"









# # Settings
# def show_settings():
#     st.header("Settings ⚙️")
#     theme = st.selectbox("Theme", options=["Light", "Dark"], index=0 if st.session_state.settings['theme'] == 'Light' else 1)
#     language = st.selectbox("Language", options=["English", "Spanish", "French"], index=0)
#     if st.button("Save Settings"):
#         st.session_state.settings['theme'] = theme
#         st.session_state.settings['language'] = language
#         st.success("Settings saved successfully!")

# # Main UI
# st.set_page_config(page_title="Intel360 – AI-Driven Competitor Analysis", layout="wide")
# st.title("Intel360: AI-Driven Competitor Analysis 📊")



# with st.sidebar:
#     st.header("Navigation")
    
#     # Initialize session state for page navigation
#     if 'page' not in st.session_state:
#         st.session_state.page = "Upload Files"

#     # Navigation buttons
#     if st.button("📊Dashboard"):
#         st.session_state.page = "Dashboard"
#     if st.button("📂Upload Files"):
#         st.session_state.page = "Upload Files"
#     if st.button("📈Analysis"):
#         st.session_state.page = "Analysis"
#     if st.button("📁Files"):
#         st.session_state.page = "Files"
#     if st.button("⚙️Settings"):
#         st.session_state.page = "Settings"

# # Display content based on the selected page
# if st.session_state.page == "Upload Files":
#     st.title("Upload Files Page")
# elif st.session_state.page == "Dashboard":
#     st.title("Dashboard Page")
# elif st.session_state.page == "Analysis":
#     st.title("Analysis Page")
# elif st.session_state.page == "Files":
#     st.title("Files Page")
# elif st.session_state.page == "Settings":
#     st.title("Settings Page")



# def process_file(pdf_file):
#     raw_text = get_pdf_text(pdf_file)
#     text_chunks = get_text_chunks(raw_text)
#     get_vector_store(text_chunks)
#     st.success(f"Processed: {pdf_file.name}")

# if st.session_state.page == "Upload Files":
#     st.header("Upload Competitor Reports")
#     pdf_docs = st.file_uploader("Select up to 3 PDF Files", accept_multiple_files=True, type=["pdf"], key="uploader")
    
#     if pdf_docs:
#         if len(pdf_docs) > 3:
#             st.warning("You can only upload a maximum of 3 files.")
#         else:
#             st.session_state.uploaded_files = pdf_docs
    
#     if st.button("Submit & Process") and pdf_docs:
#         for pdf_file in pdf_docs:
#             st.subheader(f"Processing: {pdf_file.name}")
#             process_file(pdf_file)








# elif st.session_state.page == "Dashboard":
#     st.header("Dashboard")
#     st.metric("Documents Analyzed", len(st.session_state.uploaded_files))
#     st.metric("Total Analyses", len(st.session_state.analysis_history))
#     st.subheader("Analysis History")
#     for history in st.session_state.analysis_history:
#         st.write(f"*Query:* {history['query']} | *Date:* {history['timestamp']}")
#         st.text_area("Result", value=history['result'], height=150)















# # elif st.session_state.page == "Analysis":
# #     st.header("Document Analysis")
# #     analysis_type = st.selectbox("Select Analysis Type", ["Competitor Strategy", "Market Trends", "SWOT Analysis", "Comparative Analysis"])

# #     if st.button("Run Analysis"):
# #         reports = []
# #         if analysis_type == "Comparative Analysis" and len(st.session_state.uploaded_files) < 2:
# #             st.warning("Comparative Analysis requires at least 2 uploaded documents.")
# #         else:
# #             for pdf_file in st.session_state.uploaded_files:
# #                 st.subheader(f"Analyzing: {pdf_file.name}")
# #                 if analysis_type == "Competitor Strategy":
# #                     query = "Analyze competitor strategies"
# #                     prompt_template = """
# #                     Analyze the provided competitor strategy context and extract key insights, summarize findings, and identify emerging trends using Generative AI. 
# #                     Context:
# #                     {context}
# #                     """
# #                 elif analysis_type == "Market Trends":
# #                     query = "Analyze in relation to market trends"
# #                     prompt_template = """
# #                     Compare the document context with current market trends and provide insights.
# #                     Context:
# #                     {context}
# #                     """
# #                 elif analysis_type == "SWOT Analysis":
# #                     query = "SWOT Analysis"
# #                     prompt_template = """
# #                     Conduct a SWOT analysis based on the document context.
# #                     Context:
# #                     {context}
# #                     """
                
# #                 with st.spinner("Analyzing..."):
# #                     report = analyze_documents(query, prompt_template)
# #                     reports.append(report)
# #                     st.text_area(f"Analysis Report for {pdf_file.name}", value=report, height=300)
            
# #             if analysis_type == "Comparative Analysis" and len(st.session_state.uploaded_files) > 1:
# #                 with st.spinner("Generating Comparative Analysis..."):
# #                     comparative_report = generate_comparative_analysis(st.session_state.uploaded_files)
# #                     st.subheader("Comparative Analysis Report")
# #                     st.dataframe(comparative_report)











# # elif st.session_state.page == "Analysis":
# #     st.header("Document Analysis")
# #     analysis_type = st.selectbox(
# #         "Select Analysis Type", 
# #         ["Competitor Strategy", "Market Trends", "SWOT Analysis", "Comparative Analysis"]
# #     )

# #     if st.button("Run Analysis"):
# #         reports = []
        
# #         # Ensure there are uploaded files
# #         if not st.session_state.uploaded_files:
# #             st.warning("Please upload at least one document before running analysis.")
# #         elif analysis_type == "Comparative Analysis" and len(st.session_state.uploaded_files) < 2:
# #             st.warning("Comparative Analysis requires at least 2 uploaded documents.")
# #         else:
# #             for pdf_file in st.session_state.uploaded_files:
# #                 st.subheader(f"Analyzing: {pdf_file.name}")

# #                 # Define query and prompt template based on analysis type
# #                 if analysis_type == "Competitor Strategy":
# #                     query = "Analyze competitor strategies"
# #                     prompt_template = """
# #                     Analyze the provided competitor strategy context and extract key insights, summarize findings, and identify emerging trends using Generative AI. 
# #                     Context:
# #                     {context}
# #                     """
# #                 elif analysis_type == "Market Trends":
# #                     query = "Analyze in relation to market trends"
# #                     prompt_template = """
# #                     Compare the document context with current market trends and provide insights.
# #                     Context:
# #                     {context}
# #                     """
# #                 elif analysis_type == "SWOT Analysis":
# #                     query = "SWOT Analysis"
# #                     prompt_template = """
# #                     Conduct a SWOT analysis based on the document context.
# #                     Context:
# #                     {context}
# #                     """
# #                 else:
# #                     # Skip individual processing for Comparative Analysis
# #                     continue  

# #                 with st.spinner("Analyzing..."):
# #                     report = analyze_documents(query, prompt_template)
# #                     reports.append(report)
# #                     st.text_area(f"Analysis Report for {pdf_file.name}", value=report, height=300)

# #             # Handle Comparative Analysis separately
# #             if analysis_type == "Comparative Analysis":
# #                 with st.spinner("Generating Comparative Analysis..."):
# #                     comparative_report = generate_comparative_analysis(st.session_state.uploaded_files)
# #                     st.subheader("Comparative Analysis Report")
# #                     st.dataframe(comparative_report)















# elif st.session_state.page == "Analysis":
#     st.header("Document Analysis")
#     analysis_type = st.selectbox(
#         "Select Analysis Type", 
#         ["Competitor Strategy", "Market Trends", "SWOT Analysis", "Comparative Analysis"]
#     )

#     if st.button("Run Analysis"):
#         reports = []
        
#         # Ensure there are uploaded files
#         if not st.session_state.uploaded_files:
#             st.warning("Please upload at least one document before running analysis.")
#         elif analysis_type == "Comparative Analysis" and len(st.session_state.uploaded_files) < 2:
#             st.warning("Comparative Analysis requires at least 2 uploaded documents.")
#         else:
#             for pdf_file in st.session_state.uploaded_files:
#                 st.subheader(f"Analyzing: {pdf_file.name}")

#                 if analysis_type != "Comparative Analysis":
#                     if analysis_type == "Competitor Strategy":
#                         query = "Analyze competitor strategies"
#                         prompt_template = """
#                         Analyze the provided competitor strategy context and extract key insights.
#                         Context:
#                         {context}
#                         """
#                     elif analysis_type == "Market Trends":
#                         query = "Analyze in relation to market trends"
#                         prompt_template = """
#                         Compare the document context with current market trends.
#                         Context:
#                         {context}
#                         """
#                     elif analysis_type == "SWOT Analysis":
#                         query = "SWOT Analysis"
#                         prompt_template = """
#                         Conduct a SWOT analysis based on the document context.
#                         Context:
#                         {context}
#                         """

#                     with st.spinner("Analyzing..."):
#                         report = analyze_documents(query, prompt_template)
#                         reports.append(report)
#                         st.text_area(f"Analysis Report for {pdf_file.name}", value=report, height=300)

#             if analysis_type == "Comparative Analysis":
#                 with st.spinner("Generating Comparative Analysis..."):
#                     comparative_report = generate_comparative_analysis(st.session_state.uploaded_files)
#                     st.subheader("Comparative Analysis Report")
#                     st.dataframe(comparative_report)



































# elif st.session_state.page == "Files":
#     st.header("Uploaded Files")
#     for file in st.session_state.uploaded_files:
#         st.write(f"📄 {file.name}")

# elif st.session_state.page == "Settings":
#     show_settings()
























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
import io


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
def get_pdf_text(pdf_file):
    # Read the file content as bytes
    pdf_bytes = pdf_file.read()
    
    # Wrap the bytes into a file-like object
    pdf_file_like = io.BytesIO(pdf_bytes)
    
    # Now use PdfReader with the file-like object
    pdf_reader = PdfReader(pdf_file_like)
    
    # Your existing code to extract text
    raw_text = ""
    for page in pdf_reader.pages:
        raw_text += page.extract_text()
    
    return raw_text

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



def process_file(pdf_file):
    raw_text = get_pdf_text(pdf_file)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
    st.success(f"Processed: {pdf_file.name}")




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
    if st.button("📊Dashboard"):
        st.session_state.page = "Dashboard"
    if st.button("📂Upload Files"):
        st.session_state.page = "Upload Files"
    if st.button("📈Analysis"):
        st.session_state.page = "Analysis"
    if st.button("📁Files"):
        st.session_state.page = "Files"
    if st.button("⚙️Settings"):
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
# if st.session_state.page == "Upload Files":
#     st.header("Upload Competitor Reports")
#     pdf_docs = st.file_uploader("Select PDF Files", accept_multiple_files=True)
#     if pdf_docs:
#         st.session_state.uploaded_files.extend(pdf_docs)
#     if st.button("Submit & Process"):
#         raw_text = get_pdf_text(pdf_docs)
#         text_chunks = get_text_chunks(raw_text)
#         get_vector_store(text_chunks)
#         st.success("Documents processed successfully!")



if st.session_state.page == "Upload Files":
    st.header("Upload Competitor Reports")
    pdf_docs = st.file_uploader("Select up to 3 PDF Files", accept_multiple_files=True, type=["pdf"], key="uploader")
    
    if pdf_docs:
        if len(pdf_docs) > 3:
            st.warning("You can only upload a maximum of 3 files.")
        else:
            st.session_state.uploaded_files = pdf_docs
    
    if st.button("Submit & Process") and pdf_docs:
        for pdf_file in pdf_docs:
            st.subheader(f"Processing: {pdf_file.name}")
            process_file(pdf_file)






elif st.session_state.page == "Dashboard":
    st.header("Dashboard")
    st.metric("Documents Analyzed", len(st.session_state.uploaded_files))
    st.metric("Total Analyses", len(st.session_state.analysis_history))
    st.subheader("Analysis History")
    for history in st.session_state.analysis_history:
        st.write(f"Query: {history['query']} | Date: {history['timestamp']}")
        st.text_area("Result", value=history['result'], height=150)

elif st.session_state.page == "Analysis":
    st.header("Document Analysis")
    analysis_type = st.selectbox("Select Analysis Type", ["Competitor Strategy", "Market Trends", "SWOT Analysis", "Comparative Analysis"])

    if st.button("Run Analysis"):
        reports = []
        if analysis_type == "Comparative Analysis" and len(st.session_state.uploaded_files) < 2:
            st.warning("Comparative Analysis requires at least 2 uploaded documents.")
        else:
            for pdf_file in st.session_state.uploaded_files:
                st.subheader(f"Analyzing: {pdf_file.name}")
                if analysis_type == "Competitor Strategy":
                    query = "Analyze competitor strategies"
                    prompt_template = """
                    Analyze the provided competitor strategy context and extract key insights, summarize findings, and identify emerging trends using Generative AI. 
                    Context:
                    {context}
                    """
                elif analysis_type == "Market Trends":
                    query = "Analyze in relation to market trends"
                    prompt_template = """
                    Compare the document context with current market trends and provide insights.
                    Context:
                    {context}
                    """
                elif analysis_type == "SWOT Analysis":
                    query = "SWOT Analysis"
                    prompt_template = """
                    Conduct a SWOT analysis based on the document context.
                    Context:
                    {context}
                    """
                
                with st.spinner("Analyzing..."):
                    report = analyze_documents(query, prompt_template)
                    reports.append(report)
                    st.text_area(f"Analysis Report for {pdf_file.name}", value=report, height=300)
            
            if analysis_type == "Comparative Analysis" and len(st.session_state.uploaded_files) > 1:
                with st.spinner("Generating Comparative Analysis..."):
                    comparative_report = generate_comparative_analysis(st.session_state.uploaded_files)
                    st.subheader("Comparative Analysis Report")
                    st.dataframe(comparative_report)

elif st.session_state.page == "Files":
    st.header("Uploaded Files")
    for file in st.session_state.uploaded_files:
        st.write(f"📄 {file.name}")

elif st.session_state.page == "Settings":
    show_settings()

















