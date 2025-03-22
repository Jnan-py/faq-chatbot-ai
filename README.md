# FAQ Chatbot AI

FAQ Chatbot AI is an interactive chatbot application built with Streamlit that allows users to upload FAQ files (JSON, TXT, PDF, DOCX), analyze their content, and chat with an AI assistant to ask questions based on the FAQ data. The app leverages modern AI and vector search technologies using Google Generative AI, Pinecone, and LangChain integrations to provide insightful answers and detailed reports.

## Features

- **User Authentication:** Sign up and login functionality with SQLite database.
- **File Upload and Management:** Upload FAQ files in various formats (JSON, TXT, PDF, DOCX). The files are processed and stored along with their vector representations.
- **Vector Storage:** Uses Pinecone to create and manage vector indexes for FAQ content.
- **Intelligent Chat Interface:** Engage with a chatbot that uses the FAQ context to answer queries, perform web searches when needed, and generate detailed analysis reports.
- **Dynamic Query Handling:** Uses Google Generative AI and various tools to modify queries, perform web searches, and classify query relevancy.
- **Persistent Chat History:** Conversations are stored in a database to maintain context across sessions.

## Technologies Used

- **[Streamlit](https://streamlit.io/):** For building the interactive web interface.
- **[SQLite](https://www.sqlite.org/):** For local user and conversation data storage.
- **[PyPDF2](https://pypi.org/project/PyPDF2/):** For processing PDF files.
- **[python-docx](https://pypi.org/project/python-docx/):** For processing DOCX files.
- **[python-dotenv](https://pypi.org/project/python-dotenv/):** For loading environment variables.
- **[Google Generative AI](https://developers.generativeai.google/):** For generating responses and query modifications.
- **[Pinecone](https://www.pinecone.io/):** For vector storage and similarity search.
- **[LangChain Integrations](https://python.langchain.com/):** For embedding and vector store management.
- **[Phi Tools](https://github.com/phi-ai):** For additional agents and web tools such as DuckDuckGo and SerpApi integrations.

## Prerequisites

- Python 3.7 or higher
- A Pinecone API Key for creating and querying vector indexes.
- A Google API Key for using Google Generative AI.
- A SerpAPI API Key for performing web searches.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Jnan-py/faq-chatbot-ai.git
   cd faq-chatbot-ai
   ```

2. **Create a virtual environment and activate it:**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use: venv\Scripts\activate
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Create a `.env` file in the project root and add your API keys:**

   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   SERPAPI_API_KEY=your_serpapi_api_key_here
   ```

## Usage

1. **Run the application:**

   ```bash
   streamlit run main.py
   ```

2. **Authentication:**
   - When the app starts, use the **Login/Signup** screen to create a new account or login with an existing account.
3. **Upload FAQ Files:**
   - Enter your Pinecone API Key in the sidebar.
   - Use the file upload interface to add FAQ files. Provide a unique name that will be used as the vector index name.
4. **Chat with the Bot:**

   - Once a file is loaded, navigate to the **Chatbot** tab.
   - Ask questions related to the uploaded FAQ. The app will analyze the FAQ context and provide direct answers or perform web searches if needed.
   - All conversation history is stored and displayed for context.

5. **View Uploaded Files:**
   - In the **Uploaded Files** tab, review the content of files you have uploaded.

## Project Structure

```
faq-chatbot-ai/
├── main.py                # Main application code
├── requirements.txt      # List of dependencies
├── .env                  # Environment variables file (not committed)
├── README.md             # README file
└── app.db                # Database
```

## Contributing

Contributions are welcome!
