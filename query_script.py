import os
import chromadb
import vertexai
from google.oauth2 import service_account
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel, GenerationConfig

# --- CONFIGURATION ---
# TODO: Update with your Google Cloud Project ID
GCP_PROJECT_ID = "google-cloud-project-id" # Replace with your actual project ID
GCP_LOCATION = "us-central1"  # Or your preferred GCP region

# --- PATHS AND CONSTANTS ---
# Place the downloaded service account key file in the same directory as this script.
SERVICE_ACCOUNT_KEY_PATH = os.path.join(os.path.dirname(__file__), 'service_account_key.json')
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), 'chroma_db')
CHROMA_COLLECTION_NAME = "pdf_embeddings_collection"

# --- MODEL PARAMETERS ---
EMBEDDING_MODEL_NAME = "gemini-embedding-001"
GENERATION_MODEL_NAME = "gemini-2.0-flash-001"
TOP_N_RESULTS = 10


def main():
    """
    Main function to run the RAG query assistant.
    """
    print("--- Digestive Health Q&A Assistant ---")

    # 1. Initialize Vertex AI using a Service Account Key
    print("\n[Step 1/3] Initializing Vertex AI with Service Account...")
    try:
        if not os.path.exists(SERVICE_ACCOUNT_KEY_PATH):
            print(f"   ERROR: Service account key file not found at '{SERVICE_ACCOUNT_KEY_PATH}'")
            print(f"   Please download the key and place it in the project folder.")
            return

        credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_KEY_PATH)
        vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION, credentials=credentials)

        # Load the models
        embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
        generation_model = GenerativeModel(GENERATION_MODEL_NAME)
        print("   - Vertex AI and models initialized successfully.")
    except Exception as e:
        print(f"   Error initializing Vertex AI: {e}")
        return

    # 2. Initialize ChromaDB
    print("\n[Step 2/3] Connecting to ChromaDB...")
    if not os.path.exists(CHROMA_DB_PATH):
        print(f"   Error: ChromaDB database not found at '{CHROMA_DB_PATH}'")
        print(f"   Please run the ingestion script first.")
        return

    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
        print(f"   - Connected to ChromaDB. Collection '{CHROMA_COLLECTION_NAME}' has {collection.count()} documents.")
    except Exception as e:
        print(f"   Error connecting to ChromaDB: {e}")
        return

    print("\n[Step 3/3] Ready to answer your questions.")
    print("Type 'quit' to exit the assistant.")

    # 3. Start the interactive Q&A loop
    while True:
        print("-" * 50)
        user_question = input("Please ask a question about digestive health: ")

        if user_question.lower() == 'quit':
            print("Exiting assistant. Goodbye!")
            break

        if not user_question.strip():
            continue

        # a. Generate an embedding for the user's question
        print("\n-> Finding relevant information in the documents...")
        try:
            question_embedding = embedding_model.get_embeddings([user_question])[0].values
        except Exception as e:
            print(f"   Error generating embedding for the question: {e}")
            continue

        # b. Query ChromaDB for the most relevant document chunks
        try:
            relevant_docs = collection.query(
                query_embeddings=[question_embedding],
                n_results=TOP_N_RESULTS
            )
        except Exception as e:
            print(f"   Error querying ChromaDB: {e}")
            continue

        # c. Construct the augmented prompt for the generation model
        context_chunks = relevant_docs['documents'][0]

        print(str(relevant_docs))

        context_for_prompt = "\n\n".join(context_chunks)

        prompt = f"""
        You are a helpful assistant who provides clear and concise answers about digestive health.
        Your answers must be based *only* on the information contained in the context provided below, which comes from Canadian Society of Intestinal Research documents. Do not use any outside knowledge.

        Context from the documents:
        ---
        {context_for_prompt}
        ---

        Based *primarily* on the context above, please answer the following question.

        Question: {user_question}
        """

        print("\n-> Constructed prompt for the model:" + prompt)

        # d. Generate the answer using the modern SDK method
        print("-> Generating a response...")
        try:
            generation_config = GenerationConfig(
                max_output_tokens=8192,
                temperature=0.5,
            )
            response = generation_model.generate_content(
                prompt,
                generation_config=generation_config
            )

            print("\n--- Answer ---\n")
            print(response.text)
            print("\n--------------\n")

        except Exception as e:
            print(f"   Error generating response from the model: {e}")


if __name__ == '__main__':
    main()
