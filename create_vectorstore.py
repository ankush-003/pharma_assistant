# from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document
import json
from pathlib import Path
from pprint import pprint
from genson import SchemaBuilder
from typing import Dict, Any, List
import os
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from utils.llms import get_ollama_embedding
from uuid import uuid4
from dotenv import load_dotenv

def json_to_langchain_documents(json_data: Dict[str, Any], filename: str) -> List[Document]:
    """
    Convert a JSON object with drug information into Langchain Documents.
    
    Args:
        json_data (Dict[str, Any]): A dictionary containing drug information
        filename (str): Name of the source file
    
    Returns:
        List[Document]: A list of Langchain Document objects
    """
    # Create a list to store documents
    documents = []
    
    # Key mapping to make document metadata more readable
    key_mapping = {
        "DESCRIPTION:": "description",
        "CLINICAL PHARMACOLOGY:": "clinical_pharmacology",
        "INDICATIONS AND USAGE:": "indications",
        "CONTRAINDICATIONS:": "contraindications",
        "WARNINGS:": "warnings",
        "PRECAUTIONS:": "precautions",
        "ADVERSE REACTIONS:": "adverse_reactions",
        "OVERDOSAGE:": "overdosage",
        "DOSAGE AND ADMINISTRATION:": "dosage",
        "HOW SUPPLIED:": "how_supplied",
        "PACKAGE LABEL.PRINCIPAL DISPLAY PANEL": "package_label",
        "INGREDIENTS AND APPEARANCE": "ingredients",
        "product_name": "product_name"
    }
    
    # Combine all text sections into a single page content
    page_content = "\n\n".join([
        f"{key}: {value}" 
        for key, value in json_data.items() 
        if value  # Only include non-empty values
    ])
    
    # Create metadata dictionary
    metadata = {
        "source": filename,
        **{
            key_mapping.get(k, k): v 
            for k, v in json_data.items() 
            if k in key_mapping
        }
    }
    
    # Create a Langchain Document
    document = Document(
        page_content=page_content,
        metadata=metadata
    )
    
    documents.append(document)
    
    return documents

def process_json_directory(directory_path: str) -> List[Document]:
    """
    Process all JSON files in a given directory and convert them to Langchain Documents.
    
    Args:
        directory_path (str): Path to the directory containing JSON files
    
    Returns:
        List[Document]: A list of Langchain Documents
    """
    # List to store all documents
    all_documents = []
    
    # Ensure the directory path exists
    directory = Path(directory_path)
    if not directory.is_dir():
        raise ValueError(f"The path {directory_path} is not a valid directory")
    
    # Iterate through all JSON files in the directory
    for json_file in directory.glob('*.json'):
        try:
            # Load JSON data
            with json_file.open('r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Convert JSON to Langchain Documents
            documents = json_to_langchain_documents(json_data, json_file.name)
            
            # Add to the list of all documents
            all_documents.extend(documents)
            
            print(f"Processed: {json_file.name}")
        
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
    
    return all_documents

if __name__ == "__main__":
    load_dotenv()

    directory_path = './datasets/microlabs_usa'
    
    # Process all JSON files in the directory
    documents = process_json_directory(directory_path)
    
    # Print summary
    print(f"\nTotal Documents Processed: {len(documents)}")
    
    # Print details of the first few documents
    for i, doc in enumerate(documents[:3], 1):
        print(f"\nDocument {i}:")
        print("Metadata:")
        print(doc.metadata)
        print("\nContent (first 500 characters):")
        print(doc.page_content[:500] + "...")
        print("-" * 50)

    print(f"Total Documents Processed: {len(documents)}")

    embeddings = get_ollama_embedding("all-minilm:l6-v2", os.getenv("BASE_URL"))

    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    uuids = [str(uuid4()) for _ in range(len(documents))]

    vector_store.add_documents(documents=documents, ids=uuids)
    vector_store.save_local("products")