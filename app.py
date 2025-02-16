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
import openai
from langchain.chat_models import ChatOpenAI 

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
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI Model
openai_model = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)


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
    if domain == "Healthcare":
        structured_query = """
        Perform a **comparative analysis** of competitors in the Healthcare sector based on the following parameters:

        - **Business Models**: Key offerings, revenue models, target customers  
        - **Revenue Streams**: Subscription services, licensing, one-time payments  
        - **Expansion Plans**: Future growth, targeted regions, partnerships  
        - **Pricing Strategies**: Discount models, competitive pricing, value-based pricing  
        - **Competitive Advantages**: Key differentiators, technology, patents  
        - **Innovation Adoption**: AI, telemedicine, wearables, blockchain  
        - **Partnerships**: Collaborations, alliances, acquisitions  
        - **Customer Acquisition Strategies**: Marketing, influencer engagement, referral programs  

        Output the response in a **structured tabular format** with competitor names as rows and comparison metrics as columns.
        """
    elif domain == "Life Insurance":
        structured_query = """
        Perform a **comparative analysis** of competitors in the Life Insurance sector based on the following parameters:

        - **Business Models**: Types of insurance (term, whole life, ULIP), target customer segments  
        - **Revenue Streams**: Premium collections, investment returns, policy surrender charges  
        - **Expansion Plans**: New product launches, market expansion, acquisitions  
        - **Pricing Strategies**: Premium models (fixed, variable), bundling options, discounts  
        - **Competitive Advantages**: Unique selling points like fast claims processing, digital tools  
        - **Innovation Adoption**: AI risk assessment, blockchain claims management, health monitoring wearables  
        - **Partnerships**: Collaborations with banks, health apps, wellness platforms  
        - **Customer Acquisition Strategies**: Digital marketing, influencers, referral programs  

        Output the response in a **structured tabular format** with competitor names as rows and comparison metrics as columns.
        """
    else:
        return "Invalid domain selection."

    # Use OpenAI GPT-3.5 Turbo for analysis
    try:
        # Send the structured query to the model
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a business analysis expert."},
                {"role": "user", "content": structured_query}
            ]
        )
        return response["choices"][0]["message"]["content"]

    except Exception as e:
        return f"Error during OpenAI analysis: {str(e)}"
    

cache = {}
document_summaries = {}

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

    for pdf_file in st.session_state.uploaded_files:
        file_name = os.path.splitext(pdf_file.name)[0]
        folder_path = f"faiss_indexes/{file_name}"

        if os.path.exists(folder_path):
            try:
                vector_store = FAISS.load_local(folder_path, embeddings, allow_dangerous_deserialization=True)

                # Retrieve chunks for summarization
                docs = vector_store.similarity_search("", k=5)  # Reduce `k` to lower API usage
                text_content = "\n".join([doc.page_content.strip() for doc in docs])

                # Summarize document (only if not already done)
                if file_name not in document_summaries:
                    document_summaries[file_name] = summarize_document(text_content)

            except Exception as e:
                st.error(f"Error processing {file_name}: {e}")

def chatbot_response(user_input):
    if not st.session_state.get("uploaded_files"):
        return "No documents uploaded! Please upload a document first."

    # Check if response is already cached
    if user_input in cache:
        return cache[user_input]

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    relevant_chunks = []

    # Search FAISS index first to find exact matches
    for pdf_file in st.session_state.uploaded_files:
        file_name = os.path.splitext(pdf_file.name)[0]
        folder_path = f"faiss_indexes/{file_name}"

        if os.path.exists(folder_path):
            try:
                vector_store = FAISS.load_local(folder_path, embeddings, allow_dangerous_deserialization=True)
                docs = vector_store.similarity_search(user_input, k=3)  # Retrieve only top 3 chunks
                
                # Extract unique relevant chunks
                unique_chunks = list(set([doc.page_content.strip() for doc in docs if doc.page_content.strip()]))
                relevant_chunks.extend(unique_chunks)

            except Exception as e:
                st.error(f"Error searching FAISS index for {file_name}: {e}")

    # If no exact match is found, use document summaries instead
    if not relevant_chunks:
        relevant_chunks = [document_summaries.get(os.path.splitext(pdf.name)[0], "") for pdf in st.session_state.uploaded_files]

    summarized_context = "\n".join(filter(None, relevant_chunks))

    # If still no content, avoid an unnecessary API call
    if not summarized_context:
        return "I couldn't find an exact answer, but I can try to infer from related document content. Let me know if you need more details."

    # If a relevant answer is already found in the retrieved text, return it directly
    if len(relevant_chunks) == 1 and len(relevant_chunks[0].split()) < 50:
        cache[user_input] = relevant_chunks[0]  # Cache short direct answers
        return relevant_chunks[0]

    # Optimized prompt to minimize token consumption
    prompt_template = (
        "You are an AI assistant that answers questions strictly based on the provided document context. "
        "Use only the given context to generate a well-explained answer. Do NOT generate information outside the document.\n\n"
        "Context:\n{context}\n\nUser Question: {user_input}\n\n"
        "Provide a precise and easy-to-understand response in simple terms."
    )
    prompt = prompt_template.format(context=summarized_context, user_input=user_input)

    # Call AI model **only if absolutely necessary**
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    response = model.invoke(prompt).content

    # Cache response to prevent redundant API calls
    cache[user_input] = response

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
    "Healthcare": {
        "Competitor Strategy": """Analyze competitor strategies within the Healthcare sector. Extract insights and structure the output under these headings: 

- **Business Models:** Key service offerings, target customers, and revenue approaches.
- **Revenue Streams:** Breakdown of income sources (e.g., subscriptions, services, licensing).
- **Expansion Plans:** Future growth strategies, regions targeted, and partnerships.
- **Pricing Strategies:** Competitive pricing models, discounts, and value-based pricing.
- **Competitive Advantages:** Unique features, technological innovations, or patents.
- **Innovation Adoption:** Use of AI, telemedicine, wearables, or advanced diagnostics.
- **Partnerships:** Key collaborations and their impact on market presence.
- **Customer Acquisition Strategies:** Marketing channels, campaigns, and engagement tactics.

**Response Format:**
- Present each section as bullet points.
- Include quantitative metrics (e.g., revenue figures, market share percentages, customer growth rates) if available.
- Provide comparative insights (e.g., which competitor leads in innovation adoption).
- Highlight trends using tables or charts for clarity (e.g., market share trends, pricing comparison charts).""",

        "Market Trends": """Identify key market trends in the Healthcare sector. Focus on emerging innovations, regulatory shifts, and financial growth patterns. Provide detailed analysis on: 

- **Industry Growth:** Yearly market expansion and emerging trends.
- **Regulatory Changes:** Impact of new healthcare laws and compliance requirements.
- **Emerging Technologies:** AI, wearables, and telemedicine adoption.
- **Market Dynamics:** Competitive shifts, mergers, and acquisitions.

**Response Format:**
- Present insights in concise bullet points.
- Include relevant data visualizations for trend comparison.""",

        "SWOT Analysis": """Perform a SWOT analysis of key healthcare competitors. Analyze the following: 

- **Strengths:** Technological advantages, market leadership, revenue sources.
- **Weaknesses:** Regulatory challenges, limited scalability, operational inefficiencies.
- **Opportunities:** Market expansion, emerging technology integration, new consumer segments.
- **Threats:** Competitor strategies, regulatory risks, economic downturns.

**Response Format:**
- Structure the SWOT analysis in a tabular format.
- Provide comparative insights on key competitors.""",

        "Comparative Analysis": """{selected_domain} comparative analysis request received."""
    },

    "Life Insurance": {
        "Competitor Strategy": """Analyze competitor strategies within the Life Insurance sector. Extract insights and structure the output under these headings: 

- **Business Models:** Types of insurance offered (e.g., term, whole life, ULIP), customer segments targeted, and distribution channels.
- **Revenue Streams:** Breakdown of income sources such as premiums, investment returns, and policy surrender charges.
- **Expansion Plans:** Future growth strategies including new product launches, market expansions, or acquisitions.
- **Pricing Strategies:** Premium pricing models, discounts for long-term policies, or bundling options.
- **Competitive Advantages:** Unique value propositions such as faster claims processing, personalized coverage, or digital tools.
- **Innovation Adoption:** AI for risk assessment, blockchain for claims management, or wearables for health monitoring.
- **Partnerships:** Collaborations with banks, health apps, or wellness platforms.
- **Customer Acquisition Strategies:** Marketing campaigns, use of influencers, referral programs, or loyalty schemes.

**Response Format:**
- Present insights using bullet points.
- Include quantitative metrics such as premium growth rates and claim settlement ratios.
- Highlight trends using charts or tables.""",

        "Market Trends": """Analyze key market trends in the Life Insurance industry, including: 

- **Growth in digital insurance adoption.**
- **The rise of AI-driven risk assessment.**
- **Impact of changing regulations on policies.**
- **Shift towards personalized and flexible policy structures.**

**Response Format:**
- Use data visualizations where possible.
- Compare industry growth rates and market penetration.""",

        "SWOT Analysis": """Perform a SWOT analysis of life insurance competitors: 

- **Strengths:** Market leadership, technological capabilities, customer trust.
- **Weaknesses:** High claim ratios, pricing disadvantages, policy limitations.
- **Opportunities:** Growth in digital insurance, AI-powered underwriting.
- **Threats:** Regulatory hurdles, economic downturns, new fintech competitors.

**Response Format:**
- Present SWOT analysis in a structured table.
- Provide insights into emerging competitive threats.""",

        "Comparative Analysis": """{selected_domain} comparative analysis request received."""
    }
}



# Page Handling
if "page" not in st.session_state:
    st.session_state.page = "Upload Files"

if st.session_state.page == "Upload Files":
    st.title("Upload Competitor Reports")
    st.write("### Step 1: Select the Document Domain")
    
    domain_mapping = {
        "Healthcare": "Market growth, regulatory impact, emerging technologies.",
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
        analysis_type = st.selectbox("Select analysis type", list(domain_prompts[selected_domain].keys()))

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
                            report = comparative_analysis(file_name, query, selected_domain)
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