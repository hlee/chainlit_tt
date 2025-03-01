import os
import chainlit as cl
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

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
    """
)

def process_documents():
    """Process the documents and create a vector store."""
    global vector_store, retriever, qa_chain
    
    # Load the document
    with open("data/sample_data.txt", "r") as f:
        raw_text = f.read()
    
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    texts = text_splitter.split_text(raw_text)
    
    # Create a vector store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts, embeddings)
    
    # Create a retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
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
        content="Welcome to the Chainlit RAG Demo! I can answer questions about the Chainlit framework and RAG systems."
    ).send()
    
    # Process documents and set up the retrieval system
    process_documents()
    
    # Store the chain in the user session
    cl.user_session.set("qa_chain", qa_chain)

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
            elements.append(
                cl.Text(
                    name=f"Source {i+1}",
                    content=source_text,
                    display="side",
                )
            )
        thinking_msg.elements = elements
        await thinking_msg.update()

if __name__ == "__main__":
    # This is used when running locally with `chainlit run app.py`
    # Not used when deployed with Chainlit Cloud
    pass