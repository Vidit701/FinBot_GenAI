# FinBot – AI-Powered Financial Assistant

FinBot is an AI-driven financial assistant that helps you **analyze, query, and understand your personal banking transactions**.  
It combines **semantic search**, **FAISS indexing**, and **LLM-powered natural language understanding** to provide clear, contextual answers about your spending patterns.

---

## Features

- **Interactive Chat UI** built with [Streamlit](https://streamlit.io/)  
- **Natural Language Queries**: Ask questions like  
  - "Show me my recent restaurant expenses"  
  - "How much did I spend on travel last month?"  
  - "Compare my grocery expenses between January and February"  
- **Semantic Transaction Search** using [SentenceTransformers](https://www.sbert.net/) + [FAISS](https://github.com/facebookresearch/faiss)  
- **LLM-Powered Responses** via [Groq LLaMA models](https://groq.com/) (through `langchain_groq`)  
- **Persistent Logs**:
  - Transaction embeddings stored in `data/`  
  - Bot conversations saved in `conversation_logs/`  
  - System logs in `logs/`

---

## Project Structure

FinBot_GenAI/
├── app.py # Main Streamlit app
├── requirements.txt # Dependencies
├── docs/
│ └── Statement.xlsx # Example transaction data
├── data/ # Embeddings + FAISS index
├── logs/ # Log files
└── conversation_logs/ # Saved conversation transcripts


---

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/Vidit701/FinBot_GenAI.git
   cd FinBot_GenAI


