# Ragchatbypdf

Chat with PDF using Gemini + HuggingFace (RAG)
A Retrieval-Augmented Generation (RAG) application that lets you chat with your PDF documents using Google Gemini and HuggingFace embeddings — built with LangChain and Streamlit.

🚀 Features

Upload one or multiple PDF files
Automatically chunks and indexes PDF content into a FAISS vector store
Uses HuggingFace all-MiniLM-L6-v2 for local, cost-free embeddings
Powered by Google Gemini 2.0 Flash for intelligent answers
Conversational interface with context-aware responses
Handles multi-PDF queries seamlessly


🏗️ Architecture
User Question
      │
      ▼
HuggingFace Embeddings (all-MiniLM-L6-v2)
      │
      ▼
FAISS Vector Store ──► Top-k Retrieval (k=3)
      │
      ▼
Prompt Template (Context + Question)
      │
      ▼
Google Gemini 2.0 Flash
      │
      ▼
Answer
RAG Pipeline:

PDF text is extracted and split into chunks (2000 chars, 200 overlap)
Chunks are embedded and stored in a local FAISS index
At query time, the top 3 most relevant chunks are retrieved
Retrieved context + user question are passed to Gemini for a grounded answer


🛠️ Tech Stack
ComponentTechnologyUIStreamlitEmbeddingsHuggingFace all-MiniLM-L6-v2Vector StoreFAISS (CPU)LLMGoogle Gemini 2.0 FlashOrchestrationLangChain (LCEL)PDF ParsingPyPDF2Text SplittingRecursiveCharacterTextSplitter

⚙️ Setup & Installation
1. Clone the repository
bashgit clone <your-repo-url>
cd <your-repo-folder>
2. Create a virtual environment
bashpython -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
3. Install dependencies
bashpip install -r requirement.txt
4. Configure environment variables
Create a .env file in the project root:
envGOOGLE_API_KEY=your_google_api_key_here

🔑 Get your Google API key from Google AI Studio

5. Run the app
bashstreamlit run app.py

📁 Project Structure
├── app.py               # Main Streamlit application
├── requirement.txt      # Python dependencies
├── .env                 # API keys (not committed to git)
├── faiss_index/
│   ├── index.faiss      # FAISS vector index (auto-generated)
│   └── index.pkl        # FAISS metadata (auto-generated)
└── README.md

📖 Usage

Upload PDFs — Use the sidebar to upload one or more PDF files.
Process — Click "Submit & Process" to extract text, chunk it, and build the FAISS index.
Ask Questions — Type any question in the text input. The app retrieves relevant context from the PDFs and answers using Gemini.


⚠️ You must process at least one PDF before asking questions.


🔧 Configuration
You can tune the following parameters in app.py:
ParameterLocationDefaultDescriptionchunk_sizeget_text_chunks()2000Characters per text chunkchunk_overlapget_text_chunks()200Overlap between consecutive chunkskget_conversational_chain()3Number of chunks retrieved per querytemperatureChatGoogleGenerativeAI0.3LLM response creativity (0=focused)modelChatGoogleGenerativeAIgemini-2.0-flashGemini model variant

📦 Dependencies
streamlit
langchain-community>=0.2.0
langchain-core>=0.2.0
langchain-google-genai>=2.0.0
langchain-text-splitters
faiss-cpu
sentence-transformers
PyPDF2
python-dotenv
google-generativeai

🔒 Security Notes

Never commit your .env file or expose your GOOGLE_API_KEY.
Add .env and faiss_index/ to your .gitignore:

gitignore.env
faiss_index/
__pycache__/
*.pyc

allow_dangerous_deserialization=True is required by FAISS for loading local indexes. Only load indexes you created yourself.


🐛 Troubleshooting
"answer is not available in the context"
The question may not match content in the uploaded PDFs. Try rephrasing or check that the correct PDFs were processed.
FAISS index not found error
You must click "Submit & Process" in the sidebar before asking questions.
Google API errors
Verify your GOOGLE_API_KEY is set correctly in .env and has access to the Gemini API.
Slow first run
The HuggingFace model (all-MiniLM-L6-v2) is downloaded on first use (~90MB). Subsequent runs are fast.

🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

📄 License
MIT
