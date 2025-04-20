import os
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import requests
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# Set Google API key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

def initialize_components():
    """Initialize all required components for the RAG system."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, max_tokens=500)
    return embeddings, llm

def load_and_process_document(pdf_path: str):
    """Load and process the PDF document."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_documents(documents)
    return texts

def create_rag_chain(embeddings, llm, texts):
    """Create the RAG chain with the given components."""
    # Create vector store
    vectorstore = FAISS.from_documents(documents=texts, embedding=embeddings)
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    # Define system prompt
    system_prompt = (
        "You are a comprehensive mental health and wellness assistant. "
        "You can answer questions about the document, suggest songs, tell jokes, "
        "recommend meditation techniques, suggest books, and search online. "
        "For document questions, use the retrieved context. "
        "For other requests, use your general knowledge and capabilities. "
        "Always be friendly, helpful, and provide detailed, practical advice.\n\n{context}"
    )
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Create chains
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

def suggest_song(mood: str) -> str:
    """Suggest a song based on mood."""
    try:
        response = llm.invoke(
            f"Suggest a song for someone feeling {mood}. "
            "Include the artist name, a brief reason why this song would be good for this mood, "
            "and a link to listen to it if possible."
        )
        return response.content
    except Exception as e:
        return f"Sorry, I couldn't suggest a song right now. Error: {str(e)}"

def tell_joke() -> str:
    """Tell a random joke."""
    try:
        response = llm.invoke(
            "Tell me a short, clean joke that would be appropriate for a mental health context. "
            "Make it uplifting and positive."
        )
        return response.content
    except Exception as e:
        return f"Sorry, I couldn't think of a joke right now. Error: {str(e)}"

def suggest_meditation(duration: str = "5 minutes", focus: str = "general") -> str:
    """Suggest meditation techniques."""
    try:
        response = llm.invoke(
            f"Suggest a {duration} meditation technique focusing on {focus}. "
            "Include step-by-step instructions, benefits, and tips for beginners."
        )
        return response.content
    except Exception as e:
        return f"Sorry, I couldn't suggest a meditation technique right now. Error: {str(e)}"

def suggest_books(topic: str = "mental health") -> str:
    """Suggest books on a specific topic."""
    try:
        response = llm.invoke(
            f"Suggest 3 books about {topic}. "
            "For each book, include the author, a brief description, "
            "and why it would be helpful. Focus on practical, actionable content."
        )
        return response.content
    except Exception as e:
        return f"Sorry, I couldn't suggest books right now. Error: {str(e)}"

def search_online(query: str) -> str:
    """Search for information online."""
    try:
        response = llm.invoke(
            f"Search for information about: {query}. "
            "Provide a comprehensive answer with recent information, "
            "include sources if possible, and highlight the most relevant points."
        )
        return response.content
    except Exception as e:
        return f"Sorry, I couldn't search for that information right now. Error: {str(e)}"

def process_query(query: str, rag_chain) -> Dict[str, Any]:
    """Process a user query and determine the appropriate response."""
    query = query.lower()
    
    # Check for song requests
    if "song" in query or "music" in query:
        mood = query.replace("song", "").replace("music", "").strip()
        return {
            "type": "song",
            "response": suggest_song(mood)
        }
    
    # Check for joke requests
    elif "joke" in query:
        return {
            "type": "joke",
            "response": tell_joke()
        }
    
    # Check for meditation requests
    elif "meditate" in query or "meditation" in query:
        duration = "5 minutes"  # Default duration
        focus = "general"  # Default focus
        if "minute" in query:
            duration = query.split("minute")[0].strip() + " minutes"
        return {
            "type": "meditation",
            "response": suggest_meditation(duration, focus)
        }
    
    # Check for book suggestions
    elif "book" in query or "read" in query:
        topic = query.replace("book", "").replace("read", "").strip()
        if not topic:
            topic = "mental health"
        return {
            "type": "books",
            "response": suggest_books(topic)
        }
    
    # Check for online search requests
    elif "search" in query or "find" in query or "look up" in query:
        search_query = query.replace("search", "").replace("find", "").replace("look up", "").strip()
        return {
            "type": "search",
            "response": search_online(search_query)
        }
    
    # Otherwise, use RAG for document-based or general questions
    else:
        try:
            response = rag_chain.invoke({"input": query})
            return {
                "type": "answer",
                "response": response['answer']
            }
        except Exception as e:
            return {
                "type": "error",
                "response": f"Sorry, I couldn't process your request. Error: {str(e)}"
            }

def main():
    # Initialize components
    embeddings, llm = initialize_components()
    
    # Load and process document
    pdf_path = os.path.join(os.path.dirname(__file__), "Atlas-of-the-Heart-by-by-Bren-23.pdf")
    texts = load_and_process_document(pdf_path)
    
    # Create RAG chain
    rag_chain = create_rag_chain(embeddings, llm, texts)
    
    print("\nWelcome to the Enhanced Mental Health Assistant!")
    print("I can help you with:")
    print("1. Answering questions about mental health")
    print("2. Suggesting songs based on your mood")
    print("3. Telling uplifting jokes")
    print("4. Recommending meditation techniques")
    print("5. Suggesting helpful books")
    print("6. Searching for information online")
    print("\nExamples:")
    print("- 'Suggest a song for when I'm feeling anxious'")
    print("- 'Tell me a joke'")
    print("- 'Suggest a 10-minute meditation for stress'")
    print("- 'Recommend books about anxiety'")
    print("- 'Search for information about mindfulness'")
    print("\nType 'quit' to exit")
    
    while True:
        query = input("\nHow can I help you today? ")
        if query.lower() == 'quit':
            break
        
        result = process_query(query, rag_chain)
        print(f"\n{result['response']}")

if __name__ == "__main__":
    main()
