import os
import json
import time

from dotenv import load_dotenv

# Load environment variables before importing other libraries
load_dotenv()

# Disable Hugging Face symlinks warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

# --- Configuration ---

DATA_DIR = "data"
PINECONE_INDEX_NAME = "silly-parse"
GEMINI_EMBEDDING_MODEL_NAME = "models/gemini-embedding-001"
EMBEDDING_DIMENSIONALITY = 768
CHUNK_SIZE = 5
OVERLAP_SIZE = 4
EMBEDDING_BATCH_SIZE = 100

# API Credentials
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Filter words
FILTER_WORDS_STR = os.getenv("FILTER_WORDS", "")
FILTER_WORDS = [word.strip().lower() for word in FILTER_WORDS_STR.split(',') if word.strip()]

# Global variables
embedding_model = None
gemini_client = None
embedding_service_choice = None

# --- Initialize APIs ---
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY must be set in .env")

try:
    # [FIX] Pinecone V3 no longer requires 'environment', it infers it securely from the API key
    pinecone = Pinecone(api_key=PINECONE_API_KEY)
    print("Pinecone initialized successfully.")
    print(f"Using Pinecone API Key: {PINECONE_API_KEY[:4]}...{PINECONE_API_KEY[-4:]}")
except Exception as e:
    print(f"Error initializing Pinecone: {e}")
    exit(1)

# --- Helper Functions ---

def clean_message_content(content: str) -> str:
    """Replaces filtered words in the message content with '[REDACTED]'."""
    if not content:
        return ""
    cleaned_content = content
    for word in FILTER_WORDS:
        cleaned_content = cleaned_content.replace(word, "[REDACTED]", -1)
        cleaned_content = cleaned_content.replace(word.capitalize(), "[REDACTED]", -1)
        cleaned_content = cleaned_content.replace(word.upper(), "[REDACTED]", -1)
    return cleaned_content

def get_embedding(text: str):
    """Generates an embedding for the given text using the selected embedding model."""
    global embedding_model, embedding_service_choice, gemini_client
    try:
        if embedding_service_choice == "huggingface":
            embedding = embedding_model.encode(text).tolist()
        elif embedding_service_choice == "gemini":
            response = gemini_client.models.embed_content(
                model=GEMINI_EMBEDDING_MODEL_NAME,
                contents=text,
                config=types.EmbedContentConfig(
                    output_dimensionality=EMBEDDING_DIMENSIONALITY
                )
            )
            # [FIX] The new SDK returns an object. We use dot notation to access values.
            embedding = response.embeddings[0].values
        else:
            raise ValueError("Embedding service not selected or invalid.")
        return embedding
    except Exception as e:
        print(f"Error generating embedding for text: '{text[:50]}...': {e}")
        return None

def process_discord_data():
    """
    Processes Discord JSON files, extracts messages, groups them into turns,
    creates sliding window chunks, generates embeddings, and upserts to Pinecone.
    """
    print(f"Starting data ingestion from '{DATA_DIR}'...")

    try:
        existing_indexes = pinecone.list_indexes()
        print(f"Pinecone reports existing indexes: {existing_indexes}")
    except Exception as e:
        print(f"Error listing Pinecone indexes: {e}")
        exit(1)

    # [FIX] Access the index names using object dot notation (.name instead of ['name'])
    index_names = [idx.name for idx in existing_indexes]

    if PINECONE_INDEX_NAME not in index_names:
        print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}'...")
        pinecone.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSIONALITY,
            metric='cosine',
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Index '{PINECONE_INDEX_NAME}' created.")
    else:
        print(f"Index '{PINECONE_INDEX_NAME}' already exists.")

    index = pinecone.Index(PINECONE_INDEX_NAME)

    all_turns = []
    file_count = 0

    print(f"\nStarting recursive scan of '{DATA_DIR}'...")
    for root, dirs, files in os.walk(DATA_DIR):
        for file_name in files:
            if file_name.endswith(".json"):
                file_path = os.path.join(root, file_name)
                file_count += 1

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    if isinstance(data, list):
                        messages = data
                    elif isinstance(data, dict):
                        messages = data.get('messages', [])
                    else:
                        continue

                    human_messages = []
                    for msg in messages:
                        if not msg.get('author', {}).get('bot', False):
                            cleaned_content = clean_message_content(msg.get('content', ''))
                            if cleaned_content:
                                human_messages.append({
                                    'author_name': msg.get('author', {}).get('global_name') or msg.get('author', {}).get('username', 'Unknown'),
                                    'content': cleaned_content,
                                    'timestamp': msg.get('timestamp')
                                })

                    if not human_messages:
                        continue

                    current_turns_for_file = []
                    if human_messages:
                        current_turn_author = human_messages[0]['author_name']
                        current_turn_messages = [human_messages[0]['content']]

                        for i in range(1, len(human_messages)):
                            msg = human_messages[i]
                            if msg['author_name'] == current_turn_author:
                                current_turn_messages.append(msg['content'])
                            else:
                                all_turns.append(f"{current_turn_author}: {' '.join(current_turn_messages)}")
                                current_turn_author = msg['author_name']
                                current_turn_messages = [msg['content']]

                        all_turns.append(f"{current_turn_author}: {' '.join(current_turn_messages)}")

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    if not all_turns:
        print("No turns were generated from any files. Exiting.")
        return

    print(f"\nFinished processing {file_count} files. Total turns generated: {len(all_turns)}")

    all_text_chunks = []
    chunk_metadata = []
    chunk_id_counter = 0

    i = 0
    while i < len(all_turns):
        chunk_end = min(i + CHUNK_SIZE, len(all_turns))
        current_chunk_turns = all_turns[i:chunk_end]

        if not current_chunk_turns:
            break

        text_chunk = "\n".join(current_chunk_turns)

        all_text_chunks.append(text_chunk)
        chunk_metadata.append({"id": f"chunk-{chunk_id_counter}", "text": text_chunk})
        chunk_id_counter += 1

        if chunk_end == len(all_turns):
            break
        else:
            i += (CHUNK_SIZE - OVERLAP_SIZE)
            if i >= len(all_turns):
                break

    print(f"Generated {len(all_text_chunks)} text chunks for embedding.")

    total_vectors_upserted = 0

    for i in range(0, len(all_text_chunks), EMBEDDING_BATCH_SIZE):
        batch_texts = all_text_chunks[i : i + EMBEDDING_BATCH_SIZE]
        batch_metadata = chunk_metadata[i : i + EMBEDDING_BATCH_SIZE]

        print(f"Generating embeddings for batch {i // EMBEDDING_BATCH_SIZE + 1}...")

        vectors_to_upsert_current_batch = []
        try:
            if embedding_service_choice == "huggingface":
                batch_embeddings = embedding_model.encode(batch_texts).tolist()
            elif embedding_service_choice == "gemini":
                # [FIX] Instead of looping, pass the entire list of `batch_texts` to the API at once!
                response = gemini_client.models.embed_content(
                    model=GEMINI_EMBEDDING_MODEL_NAME,
                    contents=batch_texts,
                    config=types.EmbedContentConfig(
                        output_dimensionality=EMBEDDING_DIMENSIONALITY
                    )
                )
                # [FIX] Use a list comprehension to grab the 'values' array from every embedding object returned
                batch_embeddings = [emb.values for emb in response.embeddings]
            else:
                raise ValueError("Embedding service not selected or invalid.")

            for j, embedding in enumerate(batch_embeddings):
                original_chunk_info = batch_metadata[j]
                vectors_to_upsert_current_batch.append({
                    "id": original_chunk_info["id"],
                    "values": embedding,
                    "metadata": {"text": original_chunk_info["text"]}
                })

            if vectors_to_upsert_current_batch:
                index.upsert(vectors=vectors_to_upsert_current_batch)
                total_vectors_upserted += len(vectors_to_upsert_current_batch)

        except Exception as e:
            print(f"Error during processing batch {i // EMBEDDING_BATCH_SIZE + 1}: {e}")

    print(f"\nIngestion process completed. Total vectors upserted: {total_vectors_upserted}.")

# --- Main Execution ---
if __name__ == "__main__":
    while True:
        choice = input("Choose embedding service (huggingface/gemini): ").lower().strip()
        if choice in ["huggingface", "gemini"]:
            embedding_service_choice = choice
            break
        else:
            print("Invalid choice. Please enter 'huggingface' or 'gemini'.")

    if embedding_service_choice == "huggingface":
        try:
            embedding_model = SentenceTransformer('thenlper/gte-base')
            print("Hugging Face embedding model loaded.")
        except Exception as e:
            print(f"Error loading HF model: {e}")
            exit(1)
    elif embedding_service_choice == "gemini":
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY must be set in .env for Gemini embeddings.")
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("Gemini API configured for embeddings.")

    if not os.path.exists(DATA_DIR):
        print(f"The data directory '{DATA_DIR}' does not exist.")
    else:
        process_discord_data()