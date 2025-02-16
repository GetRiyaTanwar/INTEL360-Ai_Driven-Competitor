import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import pandas as pd
import io
from dotenv import load_dotenv
import spacy
import base64

# Set page config
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


# Load logo image
logo_base64 = get_image_base64("INSIGHT IQ LOGO.png")
if logo_base64:
    st.sidebar.markdown(
        f"""
        <div style="display: flex; flex-direction: column; align-items: center; margin-top: -40px; margin-bottom: -30px;">
            <img src="data:image/png;base64,{logo_base64}" alt="Logo" style="width: 200px; height: auto;">
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    def extract_entities(text):
        """Extracts market-related entities from text using NLP."""
        doc = nlp(text)
        return {ent.label_: ent.text for ent in doc.ents if ent.label_ in ["ORG", "MONEY", "PERCENT", "GPE"]}

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
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0.3)
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


def comparative_analysis(file_name, query, domain):
    folder_path = f"faiss_indexes/{file_name}"

    try:
        # Load the FAISS index
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local(folder_path, embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(query)
    except Exception as e:
        return f"Error loading FAISS index or similarity search failed: {str(e)}"

    # Define structured queries for each domain
    structured_query = ""
    if domain == "Mutual Funds":  # Removed unnecessary spaces
        structured_query = """
        Perform a comparative analysis of competitors in the Mutual funds sector based on the following parameters from the upload document :


        Output the response in a structured tabular format with competitor names as columns and comparison metrics as rows using Markdown table format.  Ensure clarity, completeness, and actionable insights in your analyses.
        """
    elif domain == "Life Insurance":
        structured_query = """
        Perform a comparative analysis of competitors in the Life Insurance sector based on the uploaded document 


        Output the response in a structured tabular format with competitor names as columns and comparison metrics as rows using Markdown table format.  Ensure clarity, completeness, and actionable insights in your analyses.
        """
    else:
        return "Invalid domain selection."

    try:
        # Prepare the prompt for Gemini 2.0 Flash
        prompt_template = PromptTemplate.from_template(
            "You are a business analysis expert specialized in competitive analysis. You are excellent at presenting insights in a structured, tabular format using Markdown tables. Ensure clarity, completeness, and actionable insights in your analyses.\n\n{text}"
        )
        formatted_prompt = prompt_template.format(text=structured_query)

        # Call Gemini 2.0 Flash through Langchain
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0.3)
        response = model.invoke(formatted_prompt)  # Invoke the model directly
        return response.content  # Access content attribute to get the string
    except Exception as e:
        return f"Error during Gemini analysis: {str(e)}"


# Chatbot Functionality
def chatbot_response(user_input):
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0.3)  # Using Gemini 2.0 Flash
    response = model.invoke(user_input)
    return response


# Sidebar Navigation
st.sidebar.header("------------Navigation Bar-------------")
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
    if st.sidebar.button(f"{icon} {page}"):
        st.session_state.page = page

# Domain prompts dictionary
domain_prompts = {
    "Mutual Funds": {
        "Competitor Strategy": """Analyze competitor strategies within the Mutual Funds sector. Extract insights and structure the output.  Focus on investment strategies, fund performance, expense ratios, and marketing tactics.

Response Format:
•⁠  ⁠Present each section as bullet points.
•⁠  ⁠Include quantitative metrics such as AUM (Assets Under Management), expense ratios, and fund returns.
•⁠  ⁠Provide comparative insights between different fund houses and their strategies.
•⁠  ⁠Highlight trends using tables or charts for clarity (e.g., market share trends, fund performance comparison charts).""",

        "Market Trends": """Identify key market trends in the Mutual Funds sector.  Focus on asset allocation trends, investor preferences, and regulatory changes.

Response Format:
•⁠  ⁠Present insights in concise bullet points.
•⁠  ⁠Include relevant data visualizations for trend comparison, such as asset allocation shifts over time and growth in specific fund categories.""",

        "SWOT Analysis": """Perform a SWOT analysis of key mutual fund competitors.  Analyze their strengths, weaknesses, opportunities, and threats in the current market environment.

Response Format:
•⁠  ⁠Structure the SWOT analysis in a tabular format.
•⁠  ⁠Provide comparative insights on key competitors, highlighting their competitive advantages and vulnerabilities.""",

        "Comparative Analysis": """{selected_domain} comparative analysis request received. Present the output in a Markdown table, comparing key metrics such as fund performance, expense ratios, and AUM across different fund houses."""
    },

    "Life Insurance": {
        "Competitor Strategy": """Analyze competitor strategies within the Life Insurance sector. Extract insights and structure the output.  Focus on product offerings, pricing strategies, and distribution channels.

Response Format:
•⁠  ⁠Present insights using bullet points.
•⁠  ⁠Include quantitative metrics such as premium growth rates and claim settlement ratios if available.
•⁠  ⁠Highlight trends using charts or tables, such as market share trends and customer acquisition costs.""",

        "Market Trends": """Analyze key market trends in the Life Insurance industry.  Focus on changing consumer needs, regulatory developments, and technological advancements.

Response Format:
•⁠  ⁠Use data visualizations where possible.
•⁠  ⁠Compare industry growth rates and market penetration, highlighting key drivers of change.""",

        "SWOT Analysis": """Perform a SWOT analysis of life insurance competitors:

Response Format:
•⁠  ⁠Present SWOT analysis in a structured table.
•⁠  ⁠Provide insights into emerging competitive threats, such as new entrants or disruptive technologies.""",

        "Comparative Analysis": """{selected_domain} comparative analysis request received. Present the output in a Markdown table, comparing key metrics such as premium rates, policy features, and claim settlement ratios across different insurance providers."""
    }
}

# Page Handling
if "page" not in st.session_state:
    st.session_state.page = "Upload Files"

if st.session_state.page == "Upload Files":
    st.title("Upload Competitor Reports")
    st.write("### Step 1: Select the Document Domain")

    domain_mapping = {
        "Mutual Funds": "Investment strategies, fund performance, expense ratios, and marketing tactics.",
        "Life Insurance": "Risk management, policy innovations, customer engagement strategies."
    }

    # Initialize session state for selected_domain if not set
    if "selected_domain" not in st.session_state:
        st.session_state.selected_domain = "Select a domain"

    # Dropdown for selecting domain
    selected_domain = st.selectbox(
        "Select Document Domain",
        ["Select a domain"] + list(domain_mapping.keys()),
        index=list(domain_mapping.keys()).index(st.session_state.selected_domain)
        if st.session_state.selected_domain in domain_mapping else 0
    )

    # Save the selected domain
    if selected_domain != "Select a domain":
        st.session_state.selected_domain = selected_domain

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
        st.session_state.uploaded_files = uploaded_files
        st.success(f"{len(uploaded_files)} file(s) uploaded successfully for {selected_domain} analysis.")

    if st.button("Submit & Process") and uploaded_files:
        st.write("Processing files...")
        st.session_state.uploaded_files = uploaded_files
        for pdf_file in uploaded_files:
            process_file(pdf_file)

    # Button to switch to Analysis Page
    if st.button("Proceed to Analysis"):
        st.session_state.page = "Analysis"
        st.rerun()  # Refresh the page

elif st.session_state.page == "Analysis":
    st.title("Run AI-Driven Analysis")

    # Ensure a domain is selected
    if "selected_domain" not in st.session_state or st.session_state.selected_domain == "Select a domain":
        st.error("No domain selected. Please go back and upload files with a domain.")
        if st.button("Go Back to Upload Page"):
            st.session_state.page = "Upload Files"
            st.rerun()
    else:
        selected_domain = st.session_state.selected_domain

    # Ensure files are uploaded
    if not st.session_state.get("uploaded_files"):
        st.warning("No files uploaded! Please upload relevant financial and market analysis documents.")
        if st.button("Go Back to Upload Page"):
            st.session_state.page = "Upload Files"
            st.rerun()
    else:
        # Domain & Analysis Selection
        selected_domain = st.session_state.selected_domain

        # Use the actual dictionary keys for selectbox options
        analysis_options = list(domain_prompts[selected_domain].keys())  #Get the keys
        analysis_type = st.selectbox("Select analysis type", analysis_options)

        if st.button("Run Analysis"):
            for pdf_file in st.session_state.uploaded_files:
                file_name = os.path.splitext(pdf_file.name)[0]
                folder_path = f"faiss_indexes/{file_name}"

                if not os.path.exists(folder_path):
                    st.error(f"FAISS index for {file_name} not found. Process the document first.")
                    continue

                st.subheader(f"Analyzing: {pdf_file.name}")
                query = domain_prompts[selected_domain][analysis_type].format(selected_domain=selected_domain)

                with st.spinner("Analyzing..."):
                    try:
                        if analysis_type == "Comparative Analysis":
                            # Pass the selected_domain to the function
                            report = comparative_analysis(file_name=file_name, query=query, domain=selected_domain)
                            # Display the report as markdown
                            st.markdown(f"## Analysis Report for {pdf_file.name}")
                            st.markdown(report, unsafe_allow_html=True)  # Use st.markdown for tables
                        else:
                            report = analyze_document(file_name, query, f"Context: {{context}}")
                            st.text_area(f"Analysis Report for {pdf_file.name}", value=report, height=300)

                        # Store analysis history in session state
                        st.session_state.setdefault("analysis_history", []).append({
                            "query": query,
                            "domain": selected_domain,
                            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "result": report
                        })

                    except Exception as e:
                        st.error(f"Analysis failed for {pdf_file.name}: {str(e)}")

    # Button to go back
    if st.button("Back to Upload Page"):
        st.session_state.page = "Upload Files"
        st.rerun()


elif st.session_state.page == "Dashboard":
    st.title("Dashboard 📊")
    st.metric("📊 Total Analyses", len(st.session_state.analysis_history))
    st.subheader("📜 Analysis History")

    if st.session_state.analysis_history:
        for analysis in st.session_state.analysis_history:
            st.markdown(f"**Query:** {analysis['query']}")
            st.markdown(f"**Timestamp:** {analysis['timestamp']}")
            st.markdown(f"**Result:** {analysis['result']}")
            st.markdown("---")
    else:
        st.info("No analysis history available.")

elif st.session_state.page == "Files":
    st.title("Uploaded Files")
    for file in st.session_state.uploaded_files:
        st.write(f"📄 {file.name}")

if st.session_state.page == "Chatbot":
    st.title("Intel360 Chatbot 🤖")
    st.markdown("**Ask about competitor analysis, insights, and AI-generated reports!**")

    user_input = st.text_input("Ask me anything about competitor analysis:")

    if user_input:
        response = chatbot_response(user_input)
        st.session_state.chat_history.append({"query": user_input, "response": response})
        st.markdown("### 🤖 Chatbot Response")
        st.markdown(f"{response}")  # Displaying response in markdown for better formatting

    # Display chat history
    st.subheader("🗂️ Chat History")
    for chat in st.session_state.chat_history:
        st.write(f"**Q:** {chat['query']}")
        st.markdown(f"**A:** {chat['response']}")
        st.markdown("---")