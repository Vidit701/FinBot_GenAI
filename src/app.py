import streamlit as st
import pandas as pd
import numpy as np
import faiss
import os
import logging
import pickle
from datetime import datetime
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
from pathlib import Path

# Setup logging
logging.basicConfig(
    filename=f'logs/finbot_{datetime.now().strftime("%Y%m%d")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('FinBot')

# Load environment variables
load_dotenv()

# Initialize session states
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if "user_question" not in st.session_state:
    st.session_state.user_question = ""

class FinBot:
    def __init__(self):
        logger.info("Initializing FinBot...")
        try:
            # Load data
            self.df = pd.read_excel("docs/Statement.xlsx")
            logger.info(f"Loaded {len(self.df)} transactions")
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded")
            
            # Create or load embeddings and FAISS index
            self.load_or_create_embeddings()
            
            # Initialize Groq LLM
            self.llm = ChatGroq(
                temperature=0.7,
                groq_api_key=os.getenv('GROQ_API_KEY'),
                model_name="llama-3.1-8b-instant"
            )
            
            # Initialize system context
            self.system_context = SystemMessage(content="""You are a helpful financial assistant analyzing bank transactions.
                For each query:
                1. Use the provided transaction data to give accurate insights
                2. Consider the dates, amounts, transaction details and types in your analysis
                3. Provide specific examples from the transactions when relevant
                4. If asked about spending patterns, analyze the amounts and frequencies
                5. For comparisons, look at similar transactions and their timing""")
            
            # Create necessary directories
            Path("logs").mkdir(exist_ok=True)
            Path("conversation_logs").mkdir(exist_ok=True)
            Path("data").mkdir(exist_ok=True)
            
        except Exception as e:
            logger.error(f"Initialization error: {str(e)}", exc_info=True)
            raise

    def load_or_create_embeddings(self):
        """Load existing embeddings or create new ones if they don't exist"""
        embeddings_path = Path("data/embeddings.pkl")
        index_path = Path("data/faiss_index.index")
        
        if embeddings_path.exists() and index_path.exists():
            logger.info("Loading existing embeddings and index...")
            try:
                # Load embeddings
                with open(embeddings_path, 'rb') as f:
                    self.df['text_for_embedding'] = pickle.load(f)
                
                # Load FAISS index
                self.index = faiss.read_index(str(index_path))
                logger.info("Existing embeddings and index loaded successfully")
                return
            except Exception as e:
                logger.warning(f"Error loading existing embeddings: {e}. Creating new ones...")
        
        # Create new embeddings
        logger.info("Creating new embeddings...")
        self.df['text_for_embedding'] = self.df['Transaction Details'] + ' - ' + self.df['Transaction Type']
        embeddings = self.embedding_model.encode(self.df['text_for_embedding'].tolist())
        
        # Create and save FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        # Save embeddings and index
        try:
            with open(embeddings_path, 'wb') as f:
                pickle.dump(self.df['text_for_embedding'], f)
            faiss.write_index(self.index, str(index_path))
            logger.info("New embeddings and index saved successfully")
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
    
    def search_similar_transactions(self, query_text, k=5):
        logger.debug(f"Searching for: {query_text}")
        try:
            query_embedding = self.embedding_model.encode([query_text]).astype('float32')
            distances, indices = self.index.search(query_embedding, k)
            
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                results.append({
                    'transaction': self.df.iloc[idx]['Transaction Details'],
                    'amount': self.df.iloc[idx]['Amount £'],
                    'date': self.df.iloc[idx]['Transaction Date'],
                    'type': self.df.iloc[idx]['Transaction Type']
                })
            return results
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}", exc_info=True)
            raise
    
    def save_conversation(self, user_input, bot_response, relevant_txns):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_logs/conversation_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"User: {user_input}\n\n")
                f.write(f"Bot: {bot_response}\n\n")
                f.write("Relevant Transactions:\n")
                for txn in relevant_txns:
                    f.write(f"- {txn['transaction']} (£{txn['amount']:.2f})\n")
            logger.info(f"Conversation saved to {filename}")
            
        except Exception as e:
            logger.error(f"Save error: {str(e)}", exc_info=True)
    
    def process_query(self, user_input):
        logger.info(f"Processing query: {user_input}")
        try:
            relevant_txns = self.search_similar_transactions(user_input)
            
            context = f"""
            Based on the transaction data:
            
            Relevant transactions for your reference:
            {[f"{t['transaction']} on {t['date']} for £{t['amount']:.2f} ({t['type']})" for t in relevant_txns]}
            
            Total transactions in database: {len(self.df)}
            Date range: {self.df['Transaction Date'].min()} to {self.df['Transaction Date'].max()}
            """
            
            messages = [self.system_context, HumanMessage(content=user_input), SystemMessage(content=context)]
            response = self.llm.generate([messages])
            bot_response = response.generations[0][0].text
            
            # Save conversation
            self.save_conversation(user_input, bot_response, relevant_txns)
            
            return bot_response
            
        except Exception as e:
            logger.error(f"Query processing error: {str(e)}", exc_info=True)
            raise

# Streamlit UI
try:
    st.set_page_config(page_title="FinBot", page_icon="💰", layout="wide")
    
    st.title("💰 FinBot - Your Financial Assistant")
    st.write("Ask me anything about your transactions!")

    # Initialize FinBot
    if 'finbot' not in st.session_state:
        with st.spinner("Initializing FinBot..."):
            st.session_state.finbot = FinBot()
            st.success("FinBot is ready!")

    # Chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_input = st.text_input(
            "Your question:", 
            value=st.session_state.user_question,
            key="user_input", 
            placeholder="e.g., Show me my recent restaurant expenses"
        )

    with col2:
        if st.button("Ask", key="ask_button"):
            if user_input:
                with st.spinner("Thinking..."):
                    try:
                        bot_response = st.session_state.finbot.process_query(user_input)
                        st.session_state.conversation_history.append({
                            "user": user_input,
                            "bot": bot_response,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        st.session_state.user_question = ""
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        logger.error(f"UI error: {str(e)}", exc_info=True)

    # Display conversation history
    st.write("### 💬 Conversation History")
    for message in reversed(st.session_state.conversation_history):
        with st.container():
            st.write(f"**You:** {message['user']}")
            st.write(f"**FinBot:** {message['bot']}")
            st.write(f"*{message['timestamp']}*")
            st.write("---")

except Exception as e:
    logger.critical(f"Critical error: {str(e)}", exc_info=True)
    st.error("A critical error occurred. Please check the logs.")