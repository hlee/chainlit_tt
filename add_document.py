"""
Utility script to add new documents to the data directory.
This script demonstrates how to process and add new documents to your RAG system.
"""

import os
import argparse
from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_file(file_path, output_dir="data"):
    """
    Process a text file and save it to the data directory.
    
    Args:
        file_path (str): Path to the text file to process
        output_dir (str): Directory to save the processed file
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return False
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Get the file name
    file_name = os.path.basename(file_path)
    output_path = os.path.join(output_dir, file_name)
    
    # Read the file
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Optional: Split the content into chunks for preview
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = text_splitter.split_text(content)
    
    # Save the file to the data directory
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"Successfully processed and saved file to {output_path}")
    print(f"Document was split into {len(chunks)} chunks for RAG processing")
    
    return True

def main():
    """Main function to parse arguments and process files."""
    parser = argparse.ArgumentParser(description="Process and add documents to the RAG system")
    parser.add_argument("file_path", help="Path to the text file to process")
    parser.add_argument("--output-dir", default="data", help="Directory to save the processed file")
    
    args = parser.parse_args()
    
    process_file(args.file_path, args.output_dir)

if __name__ == "__main__":
    main()