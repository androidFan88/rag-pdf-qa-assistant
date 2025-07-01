# main_script.py
import os
import re
import fitz  # PyMuPDF
import chromadb
import vertexai
from vertexai.language_models import TextEmbeddingModel
from tqdm import tqdm

# --- CONFIGURATION ---
# TODO: Update with your Google Cloud Project ID
GCP_PROJECT_ID = "google-cloud-project-id"  # Replace with your actual project ID
GCP_LOCATION = "us-central1"  # Or your preferred GCP region

# --- PATHS AND CONSTANTS ---
# Create a folder named 'pdfs' in the same directory as this script
# and place your PDF files inside it.
PDF_DIRECTORY = os.path.join(os.path.dirname(__file__), 'pdfs')
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), 'chroma_db')
CHROMA_COLLECTION_NAME = "pdf_embeddings_collection"

# --- MODEL AND CHUNKING PARAMETERS ---
EMBEDDING_MODEL_NAME = "gemini-embedding-001"
CHUNK_SIZE = 1000  # Size of text chunks in characters
CHUNK_OVERLAP = 150  # Number of characters to overlap between chunks


def clean_text(text: str) -> str:
    """
    Cleans the extracted text by removing excessive whitespace and non-standard characters.
    Args:
        text: The raw text extracted from a PDF.
    Returns:
        Cleaned text.
    """
    # Replace multiple newlines and spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # You can add more specific cleaning rules here if needed
    # For example, removing headers/footers if they follow a pattern.
    return text.strip()


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts and cleans text from a single PDF file.
    Args:
        pdf_path: The full path to the PDF file.
    Returns:
        A single string containing all the cleaned text from the PDF.
    """
    print(f"   - Extracting text from: {os.path.basename(pdf_path)}")
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
        return clean_text(full_text)
    except Exception as e:
        print(f"      Error extracting text from {os.path.basename(pdf_path)}: {e}")
        return ""


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Splits a long text into smaller, overlapping chunks.
    Args:
        text: The text to be chunked.
        chunk_size: The maximum size of each chunk (in characters).
        chunk_overlap: The number of characters to overlap between consecutive chunks.
    Returns:
        A list of text chunks.
    """
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks


def get_gemini_embeddings(model: TextEmbeddingModel, text_chunks: list[str]) -> list[list[float]]:
    """
    Generates embeddings for a list of text chunks using the specified Gemini model.
    Handles potential API errors.
    Args:
        model: The initialized TextEmbeddingModel from Vertex AI.
        text_chunks: A list of text strings to embed.
    Returns:
        A list of embedding vectors.
    """
    embeddings = []
    # The gemini-embedding-001 model supports one text instance per request.
    # We iterate through the chunks and get embeddings one by one.
    print(f"   - Generating {len(text_chunks)} embeddings...")
    for chunk in tqdm(text_chunks, desc="      Embedding Chunks"):
        try:
            # The API expects a list, even for a single item.
            response = model.get_embeddings([chunk])
            # The response is a list of one embedding object.
            embeddings.append(response[0].values)
        except Exception as e:
            print(f"      Could not get embedding for chunk. Error: {e}")
            # Optionally, you could add a placeholder or skip this chunk.
            # For this script, we'll skip it.
            continue
    return embeddings


def main():
    """
    Main function to orchestrate the PDF processing and embedding pipeline.
    """
    print("--- Starting PDF to ChromaDB Embedding Pipeline ---")

    # 1. Initialize Vertex AI
    print(f"\n[Step 1/5] Initializing Vertex AI for project '{GCP_PROJECT_ID}'...")
    try:
        vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
        # Load the pre-trained embedding model
        embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
        print("   - Vertex AI initialized successfully.")
    except Exception as e:
        print(f"   Error initializing Vertex AI: {e}")
        print("   Please ensure you have authenticated with 'gcloud auth application-default login'")
        print("   and that the project ID is correct.")
        return

    # 2. Initialize ChromaDB
    print("\n[Step 2/5] Initializing ChromaDB...")
    if not os.path.exists(CHROMA_DB_PATH):
        os.makedirs(CHROMA_DB_PATH)

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
    print(f"   - ChromaDB client initialized. Using collection: '{CHROMA_COLLECTION_NAME}'")

    # 3. List PDF files
    print("\n[Step 3/5] Finding PDF files to process...")
    if not os.path.exists(PDF_DIRECTORY):
        os.makedirs(PDF_DIRECTORY)
        print(f"   - Created 'pdfs' directory. Please add your PDF files there and run again.")
        return

    pdf_files = [f for f in os.listdir(PDF_DIRECTORY) if f.lower().endswith('.pdf')]

    if not pdf_files:
        print("   - No PDF files found in the 'pdfs' directory.")
        return

    print(f"   - Found {len(pdf_files)} PDF(s) to process.")

    # 4. Process each PDF
    print("\n[Step 4/5] Processing each PDF file...")
    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_DIRECTORY, pdf_file)
        print(f"\nProcessing: {pdf_file}")

        # Extract and chunk text
        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text:
            print("   - Skipping file due to extraction error or empty content.")
            continue

        text_chunks = chunk_text(raw_text, CHUNK_SIZE, CHUNK_OVERLAP)
        if not text_chunks:
            print("   - No text chunks generated after processing.")
            continue
        print(f"   - Extracted and split text into {len(text_chunks)} chunks.")

        # Generate embeddings
        embeddings = get_gemini_embeddings(embedding_model, text_chunks)
        if not embeddings:
            print("   - No embeddings were generated for this file. Skipping.")
            continue

        # 5. Store in ChromaDB
        print(f"   - Storing {len(embeddings)} embeddings in ChromaDB...")

        # Create unique IDs for each chunk to store in the database
        # Format: filename_chunk_index
        ids = [f"{pdf_file}_{i}" for i in range(len(embeddings))]

        # Create metadata to store the source filename
        metadatas = [{"source": pdf_file} for _ in range(len(embeddings))]

        try:
            collection.add(
                embeddings=embeddings,
                documents=text_chunks,
                metadatas=metadatas,
                ids=ids
            )
            print("   - Successfully stored embeddings.")
        except Exception as e:
            print(f"      Error storing embeddings in ChromaDB: {e}")

    print("\n[Step 5/5] --- Pipeline Finished ---")
    print(f"Total documents in collection: {collection.count()}")


if __name__ == '__main__':
    main()
