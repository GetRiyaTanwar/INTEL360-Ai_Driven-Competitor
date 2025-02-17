# INSIGHT IQ: AI-Powered Competitive Analysis Tool 🚀


## Overview

INSIGHT IQ is a cutting-edge application designed to streamline competitive analysis using the power of AI.  It allows users to upload competitor reports (PDFs), extract insights using advanced NLP techniques, and generate comprehensive summaries and comparative analyses.  Leveraging the latest advancements in Large Language Models (LLMs), INSIGHT IQ provides actionable intelligence to inform strategic decision-making.

## Key Features

*   **Intuitive PDF Upload:**  Easily upload multiple competitor reports via a drag-and-drop interface.
*   **Intelligent Text Extraction:** Employs PyPDF2 to efficiently extract text data from PDFs.
*   **Advanced Text Chunking:** Utilizes LangChain to break down documents into manageable chunks for improved processing.
*   **Vector Embedding & Indexing:**  FAISS (Facebook AI Similarity Search) creates a searchable index for rapid information retrieval.
*   **AI-Driven Analysis:** Leverages Google's Generative AI models (Gemini) for detailed summaries, trend insights, and entity recognition.
*   **Structured Comparative Analysis:**  Generates comparative reports in a structured, tabular format (Markdown) for easy understanding.
*   **Storytelling Insights:**  Provides concise summaries of key findings in a narrative style.
*   **Interactive Chatbot:**  An AI-powered chatbot enables users to ask questions and receive answers based on the uploaded documents.
*   **Interactive Dashboard:** Visualize and export analysis results.
*   **Streamlit-Based UI:**  A user-friendly interface built with Streamlit for seamless interaction.
*   **Domain Specialization:** Supports analysis for specific domains like Mutual Funds and Life Insurance with tailored prompts.

## Technology Stack

INSIGHT IQ is built on a robust technology stack that leverages the best tools for AI-powered analysis and user interface development.

*   **Frontend:**
    *   **Streamlit:** Used for creating the interactive user interface and dashboard.
*   **Backend:**
    *   **LangChain:** Facilitates AI-driven analysis and workflow management.
    *   **FAISS:** Enables efficient vector search for relevant information retrieval.
*   **Gen-AI:**
    *   **Google Gemini:** Powers summarization, insight generation, and entity recognition.
*   **Libraries:**
    *   **Matplotlib/Seaborn:** For visual analytics and creating informative charts.
    *   **Pandas:** For data manipulation and analysis, especially for structured comparative reports.
    *   **PyPDF2:** Used for extracting text from PDF documents.

## Workflow & Process

The application follows a streamlined workflow to ensure efficient and accurate analysis:

1.  **Upload PDFs:** Users upload competitor reports with an intuitive drag-and-drop interface.
2.  **Text Extraction:** Data is extracted from multiple PDFs using PyPDF2, supporting multiple formats.
3.  **Text Chunking:** LangChain breaks documents into manageable chunks to improve processing.
4.  **Vector Embedding:** FAISS creates a searchable index to enable rapid information retrieval.
5.  **AI Analysis:** Google Gemini generates detailed summaries and trend insights with entity recognition.
6.  **Visualization:** Matplotlib and Seaborn present results through interactive charts, highlighting patterns and insights clearly.

## Setup and Installation

Follow these steps to set up and run INSIGHT IQ on your local machine:

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/GetRiyaTanwar/INTEL360-Ai_Driven-Competitor.git
    cd INTEL360-Ai_Driven-Competitor
    ```

2.  **Create a Virtual Environment (Recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Google Cloud Credentials:**

    *   **Enable the Gemini API:** Go to the [Google Cloud Console](https://console.cloud.google.com/) and enable the Gemini API.
    *   **Create API Key:** Create an API key and store it securely.
    *   **Set Environment Variable:** Set the `GOOGLE_API_KEY` environment variable:

        ```bash
        export GOOGLE_API_KEY="YOUR_API_KEY"  # Linux/macOS
        set GOOGLE_API_KEY="YOUR_API_KEY"  # Windows
        ```
        Or create a `.env` file in the root directory with the following content:
        ```
        GOOGLE_API_KEY="YOUR_API_KEY"
        ```

        ***Important:***  Do not commit your API key to the repository!

5.  **Download spaCy Model:**

    ```bash
    python -m spacy download en_core_web_sm
    ```

6.  **Run the Streamlit Application:**

    ```bash
    streamlit run app.py
    ```

    This will open the application in your web browser.


## Code Structure

*   `app.py`: Main Streamlit application file containing the UI and core logic.
*   `requirements.txt`: List of Python dependencies.
*   `faiss_indexes/`: Directory to store FAISS indexes (created dynamically).
*   `INSIGHT IQ LOGO.png`: The application logo image.
*   `.env`: (Optional) File to store environment variables (like the API key).  Make sure to add this file to your `.gitignore`.

