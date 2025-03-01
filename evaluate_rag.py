"""
Evaluation script for the RAG system.
This script demonstrates how to evaluate the performance of your RAG implementation.
"""

import os
import json
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.evaluation import QAEvalChain

# Load environment variables
load_dotenv()

# Check if API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY is not set in the .env file")

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

# Sample evaluation questions with expected answers
EVAL_QA_PAIRS = [
    {
        "question": "What is Chainlit?",
        "answer": "Chainlit is a Python framework for building conversational AI applications. It provides a user-friendly interface for interacting with language models and other AI components."
    },
    {
        "question": "How does RAG work?",
        "answer": "RAG works by indexing documents, retrieving relevant documents when a query is received, and then using those documents as context for the language model to generate a response."
    },
    {
        "question": "What are the limitations of LLMs?",
        "answer": "LLMs have limitations including knowledge cutoff (they don't know about events after their training), reasoning limitations, contextual understanding challenges, and potential bias from training data."
    },
    {
        "question": "What is prompt engineering?",
        "answer": "Prompt engineering is the practice of designing effective inputs (prompts) to guide LLMs toward generating desired outputs. This involves crafting instructions, examples, and context to elicit the best possible responses."
    },
    {
        "question": "What is a question that shouldn't have an answer?",
        "answer": "I don't have enough information to answer this question."
    }
]

def setup_rag_system():
    """Set up the RAG system for evaluation."""
    # Get all text files in the data directory
    data_files = [f for f in os.listdir("data") if f.endswith(".txt")]
    
    if not data_files:
        print("No documents found in the data directory.")
        return None
    
    # Load and process documents
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    
    for file_name in data_files:
        file_path = os.path.join("data", file_name)
        with open(file_path, "r") as f:
            raw_text = f.read()
        
        texts = text_splitter.split_text(raw_text)
        documents.extend(texts)
    
    # Create a vector store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(documents, embeddings)
    
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
    
    return qa_chain

def evaluate_rag_system(qa_chain):
    """Evaluate the RAG system using predefined QA pairs."""
    if not qa_chain:
        print("RAG system setup failed. Cannot evaluate.")
        return
    
    # Prepare evaluation data
    eval_data = []
    predictions = []
    
    # Get predictions for each question
    for i, qa_pair in enumerate(EVAL_QA_PAIRS):
        question = qa_pair["question"]
        print(f"Processing question {i+1}/{len(EVAL_QA_PAIRS)}: {question}")
        
        # Get prediction from the RAG system
        result = qa_chain({"query": question})
        prediction = result["result"]
        
        # Store the prediction
        predictions.append(prediction)
        
        # Prepare evaluation data
        eval_data.append({
            "question": question,
            "answer": qa_pair["answer"],
            "prediction": prediction
        })
    
    # Create an evaluation chain
    eval_llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    eval_chain = QAEvalChain.from_llm(eval_llm)
    
    # Evaluate the predictions
    graded_outputs = eval_chain.evaluate(
        [{"question": d["question"], "answer": d["answer"]} for d in eval_data],
        [{"text": d["prediction"]} for d in eval_data]
    )
    
    # Process and display the results
    results = []
    correct_count = 0
    
    for i, output in enumerate(graded_outputs):
        correctness = output.get("text", "").strip().lower()
        is_correct = "correct" in correctness or "yes" in correctness
        
        if is_correct:
            correct_count += 1
        
        result = {
            "question": eval_data[i]["question"],
            "expected_answer": eval_data[i]["answer"],
            "prediction": eval_data[i]["prediction"],
            "evaluation": output.get("text", ""),
            "is_correct": is_correct
        }
        
        results.append(result)
        
        # Print the result
        print(f"\nQuestion {i+1}: {result['question']}")
        print(f"Expected: {result['expected_answer']}")
        print(f"Predicted: {result['prediction']}")
        print(f"Evaluation: {result['evaluation']}")
        print(f"Correct: {result['is_correct']}")
    
    # Calculate and display the overall accuracy
    accuracy = correct_count / len(EVAL_QA_PAIRS) * 100
    print(f"\nOverall Accuracy: {accuracy:.2f}% ({correct_count}/{len(EVAL_QA_PAIRS)})")
    
    # Save the evaluation results to a file
    with open("evaluation_results.json", "w") as f:
        json.dump({
            "results": results,
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_questions": len(EVAL_QA_PAIRS)
        }, f, indent=2)
    
    print("\nEvaluation results saved to evaluation_results.json")

def main():
    """Main function to run the evaluation."""
    print("Setting up the RAG system...")
    qa_chain = setup_rag_system()
    
    print("\nEvaluating the RAG system...")
    evaluate_rag_system(qa_chain)

if __name__ == "__main__":
    main()