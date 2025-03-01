"""
Advanced RAG example with multiple document sources and metadata filtering.
This script demonstrates how to extend the basic RAG system with more advanced features.
"""

import os
import chainlit as cl
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from dotenv import load_dotenv
import glob

# Load environment variables
load_dotenv()

# Check if API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY is not set in the .env file")

# Initialize global variables
vector_store = None
retriever = None
qa_chain = None

# Define the prompt template
QA_CHAIN_PROMPT = PromptTemplate.from_template(
    """
    You are a helpful AI assistant that provides information based on the documents you have access to.
    
    Context information is below:
    {context}
    
    Question: {question}
    
    Provide a comprehensive answer to the question based on the provided context.
    If the context doesn't contain relevant information to answer the question, 
    just say "I don't have enough information to answer this question."
    
    Include citations to the source documents in your answer using the format [Source: document_name].
    """
)

def process_documents():
    """Process multiple documents with metadata and create a vector store."""
    global vector_store, retriever, qa_chain
    
    # Get all text files in the data directory
    file_paths = glob.glob("data/*.txt")
    
    if not file_paths:
        print("No documents found in the data directory.")
        return
    
    # Initialize document list
    documents = []
    
    # Process each file
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        
        # Determine document category based on filename
        # This is a simple example - you could use more sophisticated methods
        if "chainlit" in file_name.lower():
            category = "chainlit"
        elif "rag" in file_name.lower():
            category = "rag"
        elif "llm" in file_name.lower():
            category = "llm"
        else:
            category = "general"
        
        # Read the file
        with open(file_path, "r") as f:
            raw_text = f.read()
        
        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        texts = text_splitter.split_text(raw_text)
        
        # Create Document objects with metadata
        for i, text_chunk in enumerate(texts):
            doc = Document(
                page_content=text_chunk,
                metadata={
                    "source": file_name,
                    "category": category,
                    "chunk_id": i,
                }
            )
            documents.append(doc)
    
    print(f"Processed {len(documents)} document chunks from {len(file_paths)} files.")
    
    # Create a vector store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    
    # Create a retriever with search options
    retriever = vector_store.as_retriever(
        search_type="mmr",  # Maximum Marginal Relevance
        search_kwargs={
            "k": 5,  # Number of documents to retrieve
            "fetch_k": 10,  # Number of documents to consider before reranking
            "lambda_mult": 0.5,  # Diversity of results (0 = max diversity, 1 = min diversity)
        }
    )
    
    # Create a QA chain
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True,
    )

@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session."""
    # Send a welcome message
    await cl.Message(
        content="Welcome to the Advanced Chainlit RAG Demo! I can answer questions based on multiple documents with metadata filtering."
    ).send()
    
    # Process documents and set up the retrieval system
    process_documents()
    
    # Store the chain and vector store in the user session
    cl.user_session.set("qa_chain", qa_chain)
    cl.user_session.set("vector_store", vector_store)
    
    # Create category filter options
    actions = [
        cl.Action(name="all", value="all", label="All Categories", payload={}),
        cl.Action(name="chainlit", value="chainlit", label="Chainlit", payload={}),
        cl.Action(name="rag", value="rag", label="RAG", payload={}),
        cl.Action(name="llm", value="llm", label="LLM", payload={}),
        cl.Action(name="general", value="general", label="General", payload={}),
    ]
    
    await cl.Message(
        content="You can filter documents by category:",
        actions=actions,
    ).send()

@cl.action_callback("all")
async def on_all(action):
    """Reset category filter to include all documents."""
    vector_store = cl.user_session.get("vector_store")
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.5},
    )
    
    # Update the QA chain with the new retriever
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True,
    )
    
    cl.user_session.set("qa_chain", qa_chain)
    await cl.Message(content=f"Now searching across all document categories.").send()

@cl.action_callback("chainlit")
@cl.action_callback("rag")
@cl.action_callback("llm")
@cl.action_callback("general")
async def on_category_filter(action):
    """Filter documents by category."""
    category = action.value
    vector_store = cl.user_session.get("vector_store")
    
    # Create a filtered retriever
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 10,
            "lambda_mult": 0.5,
            "filter": {"category": category},
        }
    )
    
    # Update the QA chain with the new retriever
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True,
    )
    
    cl.user_session.set("qa_chain", qa_chain)
    await cl.Message(content=f"Now searching only in the '{category}' category.").send()

@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages."""
    # Get the QA chain from the user session
    qa_chain = cl.user_session.get("qa_chain")
    
    # Send a thinking message
    thinking_msg = cl.Message(content="Thinking...", author="System")
    await thinking_msg.send()
    
    # Call the QA chain
    response = await qa_chain.ainvoke({"query": message.content})
    answer = response["result"]
    source_docs = response["source_documents"]
    
    # Update the thinking message with the answer
    thinking_msg.content = answer
    thinking_msg.author = "Assistant"
    await thinking_msg.update()
    
    # Send source documents as elements
    if source_docs:
        elements = []
        for i, doc in enumerate(source_docs):
            source_text = doc.page_content
            source_name = doc.metadata.get("source", f"Source {i+1}")
            category = doc.metadata.get("category", "unknown")
            
            elements.append(
                cl.Text(
                    name=f"{source_name} (Category: {category})",
                    content=source_text,
                    display="side",
                )
            )
        thinking_msg.elements = elements
        await thinking_msg.update()

if __name__ == "__main__":
    # This is used when running locally with `chainlit run advanced_rag.py`
    pass