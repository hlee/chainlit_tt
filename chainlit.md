# Chainlit RAG Demo 🔍

Welcome to the Chainlit RAG (Retrieval-Augmented Generation) Demo!

## About this demo

This application demonstrates how to build a simple RAG system using:
- **Chainlit**: For the chat interface
- **LangChain**: For document processing and retrieval
- **OpenAI**: For embeddings and text generation
- **FAISS**: For vector storage and similarity search

## How to use

1. Type your question in the chat input
2. The system will:
   - Search for relevant information in the knowledge base
   - Generate a response based on the retrieved information
   - Show the source documents used to generate the response

## Sample questions to try

- "What is Chainlit?"
- "How does RAG work?"
- "What are the benefits of using RAG?"
- "What are the key features of Chainlit?"

## Behind the scenes

When you ask a question, the system:
1. Converts your question into a vector embedding
2. Searches the vector store for similar content
3. Retrieves the most relevant document chunks
4. Sends these chunks as context to the language model
5. Returns the generated response along with the source documents

Feel free to explore and ask questions about Chainlit and RAG systems!