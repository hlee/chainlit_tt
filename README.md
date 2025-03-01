# Chainlit RAG Tutorial

This is a simple tutorial project demonstrating how to build a Retrieval-Augmented Generation (RAG) system using Chainlit and LangChain.

## Project Overview

This project implements a simple question-answering system that:

1. Loads and processes documents from the `data` directory
2. Creates a vector store using FAISS for efficient retrieval
3. Uses OpenAI embeddings and ChatGPT for generating responses
4. Provides a chat interface using Chainlit

## Prerequisites

- Python 3.8 or higher
- OpenAI API key

## Setup Instructions

1. Clone this repository:
   ```
   git clone <repository-url>
   cd chainlit_tt
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   - Open the `.env` file
   - Replace `your_openai_api_key_here` with your actual OpenAI API key

4. Run the application:
   ```
   chainlit run app.py
   ```

5. Open your browser and navigate to `http://localhost:8000` to interact with the application

## How It Works

### Document Processing

The application reads the sample document from `data/sample_data.txt` and splits it into smaller chunks using LangChain's `RecursiveCharacterTextSplitter`. This allows the system to work with large documents by breaking them down into manageable pieces.

```python
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
```

### Vector Store Creation

The chunks are then converted into vector embeddings using OpenAI's embedding model and stored in a FAISS vector store. This allows for efficient similarity search when retrieving relevant context for a query.

```python
# Create a vector store
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_texts(texts, embeddings)

# Create a retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
```

### Retrieval-Augmented Generation

When a user asks a question, the system:
1. Retrieves the most relevant document chunks from the vector store
2. Passes these chunks as context to the language model
3. Generates a response based on the provided context

```python
# Create a QA chain
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    return_source_documents=True,
)
```

### Chainlit Integration

Chainlit provides the chat interface and handles the user interaction. The application uses Chainlit's session management to maintain the state of the QA chain and displays source documents alongside the generated responses.

```python
@cl.on_chat_start
async def on_chat_start():
    # Initialize the chat session
    # ...

@cl.on_message
async def on_message(message: cl.Message):
    # Handle incoming messages
    # ...
```

## Extending the Project

Here are some ways you can extend this project:

1. **Add more documents**: Place additional text files in the `data` directory and modify the `process_documents` function to load them all.

2. **Use different document types**: Integrate PDF, Word, or other document types using LangChain document loaders.

3. **Implement metadata filtering**: Add metadata to your documents and implement filtering based on document properties.

4. **Customize the UI**: Use Chainlit's UI customization options to enhance the user experience.

5. **Deploy the application**: Deploy your Chainlit application to a cloud provider for public access.

## Resources

- [Chainlit Documentation](https://docs.chainlit.io)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [OpenAI API Documentation](https://platform.openai.com/docs/introduction)