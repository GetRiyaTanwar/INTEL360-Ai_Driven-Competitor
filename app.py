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
import time  # Import time module for rate limiting

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
else:
    st.sidebar.write("Logo not found") # Display a message if the logo isn't found.

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
if 'current_file_index' not in st.session_state:
    st.session_state.current_file_index = 0  # Start with the first file
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'combined_faiss_index' not in st.session_state:
    st.session_state.combined_faiss_index = None  # Store combined FAISS index
if 'last_api_call_time' not in st.session_state:
    st.session_state.last_api_call_time = 0
if 'api_call_count' not in st.session_state:
    st.session_state.api_call_count = 0

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

    return vector_store # Return the vector store



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
        structured_query = f"""
        Perform a comparative analysis of competitors in the Mutual funds sector based on the uploaded document.  *Extract real data points directly from the document. Do not use hypothetical data.*

        Output the response in a structured tabular format with the following columns: Metric, HDFC Mid-Cap Opportunities Fund (this file), NIFTY Midcap 150 TRI (Primary Benchmark), NIFTY 50 TRI (Secondary Benchmark), Source (Page Number). Using Markdown table format.  Ensure clarity, completeness, and actionable insights in your analyses.  Include the specific data points and their source (page number or section) from the original document. If data is missing from document , mark it as missing in tabular format.
        Focus specifically on metrics such as:
        * NAV
        *	Performance (Returns %)
        *	5 Year Average forward P/E
        *	10 Year Average forward P/E
        *	Expense ratio (Regular plan)
        *	Return to Risk Ratio(March'14-March'24)
        * Sector Allocation (Financials, Top holdings).
        """

    elif domain == "Life Insurance":
        structured_query = f"""
        Perform a comparative analysis of competitors in the Life Insurance sector based on the uploaded document. *Extract real data points directly from the document. Do not use hypothetical data.*


        Output the response in a structured tabular format with competitor names as columns and comparison metrics as rows using Markdown table format. Only provide the information available within the document. Ensure clarity, completeness, and actionable insights in your analyses.  Include the specific data points and their source (page number or section) from the original document.  If data is missing from document , mark it as missing in tabular format.
        """
    else:
        return "Invalid domain selection."

    try:
        # Prepare the prompt for Gemini 2.0 Flash
        prompt_template = PromptTemplate.from_template(
            "You are a business analysis expert specialized in competitive analysis. You are excellent at presenting insights in a structured, tabular format using Markdown tables. Ensure clarity, completeness, and actionable insights in your analyses, using ONLY the data provided in the document. Cite the source of each data point (page number, table number, etc.). If a specific data point is not available in the document, clearly mark it as 'Not Available' or 'N/A'.\n\n{text}"
        )
        formatted_prompt = prompt_template.format(text=structured_query)

        # Call Gemini 2.0 Flash through Langchain
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0.3)
        response = model.invoke(formatted_prompt)  # Invoke the model directly
        return response.content  # Access content attribute to get the string
    except Exception as e:
        return f"Error during Gemini analysis: {str(e)}"



cache = {}
document_summaries = {}


def combine_faiss_indexes(vector_stores):
    """Combines multiple FAISS indexes into one."""
    if not vector_stores:
        return None

    combined_index = vector_stores[0]  # Start with the first index
    for i in range(1, len(vector_stores)):
        combined_index.merge_from(vector_stores[i])

    return combined_index


def summarize_document(text):
    """Summarizes the given document content and extracts key points (only called once per document)."""
    if text in cache:  # Check if already summarized
        return cache[text]

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = (
        "Summarize the following document while extracting all key points. "
        "Ensure the summary is concise yet retains important details.\n\n"
        f"Document:\n{text}\n\nSummary:"
    )
    summary = model.invoke(prompt).content
    cache[text] = summary  # Store summary to avoid redundant API calls
    return summary

def preprocess_uploaded_documents():
    """Processes uploaded documents, extracts key points, and stores summaries."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_stores = []  # To store individual FAISS indexes

    for pdf_file in st.session_state.uploaded_files:
        file_name = os.path.splitext(pdf_file.name)[0]
        folder_path = f"faiss_indexes/{file_name}"

        if os.path.exists(folder_path):
            try:
                vector_store = FAISS.load_local(folder_path, embeddings, allow_dangerous_deserialization=True)
                vector_stores.append(vector_store)  # Add to the list of vector stores

                # Retrieve chunks for summarization
                docs = vector_store.similarity_search("", k=5)  # Reduce k to lower API usage
                text_content = "\n".join([doc.page_content.strip() for doc in docs])

                # Summarize document (only if not already done)
                if file_name not in document_summaries:
                    document_summaries[file_name] = summarize_document(text_content)

            except Exception as e:
                st.error(f"Error processing {file_name}: {e}")

    # Combine FAISS indexes after processing all files
    if vector_stores:
        st.session_state.combined_faiss_index = combine_faiss_indexes(vector_stores)

MIN_DELAY_SECONDS = 1.5  # Increase delay (try 1.5 or 2 seconds)

def chatbot_response(user_input):
    """Generates a chatbot response based on uploaded documents."""
    if not st.session_state.get("uploaded_files"):
        return "No documents uploaded! Please upload a document first."

    # Check if response is already cached
    if user_input in cache:
        return cache[user_input]

    # If a combined FAISS index is available, use it
    if not st.session_state.combined_faiss_index:
        st.warning("No combined index found. Please re-upload and process the documents.")
        return "Please re-upload and process documents to create an index."

    try:
        docs = st.session_state.combined_faiss_index.similarity_search(user_input, k=3)  # k=3:  Aggressively limit the number of chunks.
        relevant_chunks = [doc.page_content.strip() for doc in docs]
    except Exception as e:
        st.error(f"Error searching combined FAISS index: {e}")
        return "An error occurred while searching the documents."

    summarized_context = "\n".join(filter(None, relevant_chunks))

    # If no content, avoid an unnecessary API call
    if not summarized_context:
        return "I couldn't find an exact answer in the documents."

    prompt_template = (
        "You are an AI assistant that answers questions strictly based on the provided document context. "
        "Use only the given context to generate a well-explained answer. Do NOT generate information outside the document.\n\n"
        "Context:\n{context}\n\nUser Question: {user_input}\n\n"
        "Provide a detailed and easy-to-understand response in simple terms. Answer should not exceed 50 words."  # Limit response length.
    )
    prompt = prompt_template.format(context=summarized_context, user_input=user_input)

    # Rate Limiting
    current_time = time.time()

    # Initialize last_api_call_time if it's the first call
    if st.session_state.last_api_call_time == 0:
        st.session_state.last_api_call_time = current_time  # Initialize on first call

    time_since_last_call = current_time - st.session_state.last_api_call_time
    if time_since_last_call < MIN_DELAY_SECONDS:
        sleep_time = MIN_DELAY_SECONDS - time_since_last_call
        time.sleep(sleep_time) # Pause execution

    # Debugging: Print information before API call
    print(f"API Call Count: {st.session_state.api_call_count}")
    print(f"Time since last API call: {time_since_last_call:.2f} seconds")

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    try:
        st.session_state.api_call_count += 1  # Increment API call count
        response = model.invoke(prompt).content

        st.session_state.last_api_call_time = time.time()  # Update last call time

    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "An error occurred while generating the response.  Please try again in a moment."

    # Cache response to prevent redundant API calls
    cache[user_input] = response

    return response



# Function to generate Storytelling Insights
def generate_storytelling_insights(analysis_result, analysis_type, selected_domain, file_name):
    """Generates compelling storytelling insights based on the analysis."""
    # Example Storytelling Prompts (customize based on domain and analysis type)

    if selected_domain == "Mutual Funds":
        if analysis_type == "Competitor Strategy":
            storytelling_prompt = f"""Summarize this competitor strategy analysis in 3-5 sentences, highlighting key findings and implications for investment decisions. Focus on actionable items and strategies to inform potential investors about {file_name}. """
        elif analysis_type == "Market Trends":
            storytelling_prompt = f"""In 3-5 sentences, describe the major market trends revealed in the analysis, and what these trends might mean for the future of fund performance and investor strategies related to {file_name}."""
        elif analysis_type == "SWOT Analysis":
            storytelling_prompt = f"""Explain the major strengths, weaknesses, opportunities, and threats to fund using SWOT analysis. Summarize the SWOT analysis in 3-5 sentences, and how these factors influence potential investment strategies for fund {file_name}."""
        elif analysis_type == "Comparative Analysis":
            storytelling_prompt = f"""Highlight the key insights from the comparative analysis, focusing on the fund's strengths and weaknesses relative to its competitors. Explain in 3-5 sentences"""
        else:
            storytelling_prompt = "Summarize key insights."
    elif selected_domain == "Life Insurance":
        if analysis_type == "Competitor Strategy":
            storytelling_prompt = f"""Summarize competitor's product offerings, customer acquisition and retention strategies, new product development, etc in 3-5 sentences. Focus on key points to showcase the company's current position."""
        elif analysis_type == "Market Trends":
            storytelling_prompt = f"""Summarize major trends that has been observed or predicted that have and/or will effect company's position. Use 3-5 sentences."""
        elif analysis_type == "SWOT Analysis":
            storytelling_prompt = f"""Explain the major strengths, weaknesses, opportunities, and threats to company. Summarize the SWOT analysis in 3-5 sentences, and how these factors influence potential investment strategies for {file_name}."""
        elif analysis_type == "Comparative Analysis":
            storytelling_prompt = f"""Summarize the strength and/or weekness of the company vs other similar or close competitor. Use 3-5 sentences."""
        else:
            storytelling_prompt = "Summarize key insights."

    else:
        storytelling_prompt = "Provide a concise summary." #Generic Summary

    # Generate Storytelling Insights using Gemini
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0.3)
    storytelling_response = model.invoke(storytelling_prompt + "\n" + analysis_result)
    return storytelling_response.content


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
    "Chatbot": "🤖",
    "Summary View": "📑" # Change to "Summary View"
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

        "Comparative Analysis": """{selected_domain} comparative analysis request received. Present the output in a Markdown table, comparing key metrics such as fund performance, expense ratios, and AUM across different fund houses.  ONLY use data points found in the document."""
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

        "Comparative Analysis": """{selected_domain} comparative analysis request received. Present the output in a Markdown table, comparing key metrics such as premium rates, policy features, and claim settlement ratios across different insurance providers. ONLY use data points found in the document."""
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
        st.success(f"You selected *{selected_domain}*. Expected PDF content: {domain_mapping[selected_domain]}")
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
        st.session_state.analysis_results = {}  # Reset analysis results
        st.session_state.current_file_index = 0 #Revert File index back to zero when new files are uploaded
        vector_stores = []  # Collect individual vector stores
        for pdf_file in uploaded_files:
            vector_store = process_file(pdf_file)  # Process each file and get its vector store
            if vector_store:
                vector_stores.append(vector_store)

        # Combine the vector stores into a single FAISS index
        if vector_stores:
            st.session_state.combined_faiss_index = combine_faiss_indexes(vector_stores)
            st.success("All files processed and combined into a single index.")
        else:
            st.warning("No files were successfully processed to create the index.")


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
            st.session_state.analysis_results = {} #Reset the list
            st.session_state.page = "Summary View" # Switch to Summary View

            for i, pdf_file in enumerate(st.session_state.uploaded_files):
                # Analyze Each File
                file_name = os.path.splitext(pdf_file.name)[0]
                folder_path = f"faiss_indexes/{file_name}"

                if not os.path.exists(folder_path):
                    st.error(f"FAISS index for {file_name} not found. Process the document first.")
                    continue

                query = domain_prompts[selected_domain][analysis_type].format(selected_domain=selected_domain)

                with st.spinner(f"Analyzing {pdf_file.name}..."):
                    try:
                        if analysis_type == "Comparative Analysis":
                            # Pass the selected_domain to the function
                            report = comparative_analysis(file_name=file_name, query=query, domain=selected_domain)
                            # Store results
                            storytelling_insights = generate_storytelling_insights(report, analysis_type, selected_domain, file_name)
                            # Store analysis history in session state
                            st.session_state.setdefault("analysis_history", []).append({
                                "query": query,
                                "domain": selected_domain,
                                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "result": report,
                                "storytelling_insights": storytelling_insights  # Store storytelling insights
                            })
                            # Store values
                            st.session_state.analysis_results[pdf_file.name] = {"report": report, "storytelling_insights": storytelling_insights}

                        else:
                            report = analyze_document(file_name, query, f"Context: {{context}}")
                            storytelling_insights = generate_storytelling_insights(report, analysis_type, selected_domain, file_name)
                             # Store analysis history in session state
                            st.session_state.setdefault("analysis_history", []).append({
                                "query": query,
                                "domain": selected_domain,
                                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "result": report,
                                "storytelling_insights": storytelling_insights  # Store storytelling insights
                            })
                            # Store values
                            st.session_state.analysis_results[pdf_file.name] = {"report": report, "storytelling_insights": storytelling_insights}

                    except Exception as e:
                        st.error(f"Analysis failed for {pdf_file.name}: {str(e)}")
            st.rerun()


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
            st.markdown(f"*Query:* {analysis['query']}")
            st.markdown(f"*Timestamp:* {analysis['timestamp']}")
            st.markdown(f"*Result:* {analysis['result']}")
            if 'storytelling_insights' in analysis:
                st.markdown(f"*Summary:* {analysis['storytelling_insights']}")
            st.markdown("---")
    else:
        st.info("No analysis history available.")

elif st.session_state.page == "Files":
    st.title("Uploaded Files")
    for file in st.session_state.uploaded_files:
        st.write(f"📄 {file.name}")

elif st.session_state.page == "Chatbot":
    st.title("Intel360 Chatbot 🤖")
    st.markdown("Ask about competitor analysis, insights, and AI-generated reports!")

    user_input = st.text_input("Ask me anything about competitor analysis:")

    if user_input:
        response = chatbot_response(user_input)
        st.session_state.chat_history.append({"query": user_input, "response": response})
        st.markdown("### 🤖 Chatbot Response")
        st.markdown(f"{response}")  # Displaying response in markdown for better formatting

    # Display chat history
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
        # File Selection Dropdown
        file_names = list(st.session_state.analysis_results.keys())
        selected_file_name = st.selectbox("Select File to View Summary", file_names)

        # Display Summary and Results
        report = st.session_state.analysis_results[selected_file_name]["report"]
        storytelling_insights = st.session_state.analysis_results[selected_file_name]["storytelling_insights"]

        st.subheader("✨ Summary")
        st.markdown(storytelling_insights)

        st.subheader("Raw Analysis Result")

        # Adjust display based on whether it is comparative analysis or normal analysis result
        if "table" in report.lower():
            st.markdown(report, unsafe_allow_html=True)  # Use st.markdown for tables
        else:
            st.text_area(f"Analysis Report for {selected_file_name}", value=report, height=300)

        if st.button("Back to Analysis Page"):
            st.session_state.page = "Analysis"
            st.rerun()