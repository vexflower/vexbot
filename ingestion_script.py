import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone, PodSpec, ServerlessSpec
import google.generativeai as genai # Changed import from google.generativeai to google.genai
import time

# --- Configuration ---
load_dotenv()

DATA_DIR = "data"
PINECONE_INDEX_NAME = "silly-parse"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONALITY = 768
CHUNK_SIZE = 5  # Number of turns per chunk
OVERLAP_SIZE = 4 # Number of turns to overlap between chunks (e.g., if CHUNK_SIZE=5, OVERLAP_SIZE=4 means 1 new turn per chunk)
EMBEDDING_BATCH_SIZE = 45 # Number of text chunks to embed in a single batch call

# Pinecone API credentials
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT") # e.g., "gcp-starter"

# Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Filter words
FILTER_WORDS_STR = os.getenv("FILTER_WORDS", "")
FILTER_WORDS = [word.strip().lower() for word in FILTER_WORDS_STR.split(',') if word.strip()]

# --- Initialize APIs ---
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY must be set in .env")
if not PINECONE_ENVIRONMENT:
    raise ValueError("PINECONE_ENVIRONMENT must be set in .env")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY must be set in .env")

try:
    pinecone = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    print("Pinecone initialized successfully.")
    print(f"Using Pinecone API Key: {PINECONE_API_KEY[:4]}...{PINECONE_API_KEY[-4:]}") # Masked for security
    print(f"Using Pinecone Environment: {PINECONE_ENVIRONMENT}")
except Exception as e:
    print(f"Error initializing Pinecone: {e}")
    exit(1)

try:
    # Attempt to configure using genai.configure (modern approach)
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print("Gemini API configured successfully.")
    except AttributeError:
        # Fallback for older versions or specific environments where configure might be missing
        print("Warning: `genai.configure` not found. This might indicate an outdated `google-generativeai` library.")
        print("Attempting to set API key via environment variable `GOOGLE_API_KEY` as a fallback.")
        os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
        # No explicit success message here, as get_model will confirm if it worked.

    gemini_model = genai.get_model(EMBEDDING_MODEL)
    print("Gemini model loaded successfully.")
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    print("Please ensure you have the latest `google-generativeai` library installed (`pip install --upgrade google-generativeai`).")
    exit(1)

# --- Helper Functions ---

def clean_message_content(content: str) -> str:
    """Replaces filtered words in the message content with '[REDACTED]'."""
    if not content:
        return ""
    cleaned_content = content
    for word in FILTER_WORDS:
        # Using a simple replace for demonstration. For more robust filtering,
        # consider regex with word boundaries.
        cleaned_content = cleaned_content.replace(word, "[REDACTED]", -1)
        cleaned_content = cleaned_content.replace(word.capitalize(), "[REDACTED]", -1)
        cleaned_content = cleaned_content.replace(word.upper(), "[REDACTED]", -1)
    return cleaned_content

def get_embedding(text: str):
    """Generates an embedding for the given text using Gemini."""
    try:
        response = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=EMBEDDING_DIMENSIONALITY
        )
        return response['embedding']
    except Exception as e:
        print(f"Error generating embedding for text: '{text[:50]}...': {e}")
        return None

def process_discord_data():
    """
    Processes Discord JSON files, extracts messages, groups them into turns,
    creates sliding window chunks, generates embeddings, and upserts to Pinecone.
    """
    print(f"Starting data ingestion from '{DATA_DIR}'...")

    # Debugging Pinecone index existence
    try:
        existing_indexes = pinecone.list_indexes()
        print(f"Pinecone reports existing indexes: {existing_indexes}")
    except Exception as e:
        print(f"Error listing Pinecone indexes: {e}")
        print("This might indicate an issue with your API key or environment preventing listing.")
        exit(1)

    # Extract just the names for comparison
    index_names = [idx['name'] for idx in existing_indexes]

    print(f"Target PINECONE_INDEX_NAME: '{PINECONE_INDEX_NAME}'")
    print(f"List of existing index names: {index_names}")
    print(f"Is '{PINECONE_INDEX_NAME}' not in existing index names? {PINECONE_INDEX_NAME not in index_names}")

    # Ensure Pinecone index exists
    if PINECONE_INDEX_NAME not in index_names:
        print(f"DEBUG: Entering 'create_index' block because index was not found in list.")
        print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}'...")
        # Note: Your existing index is serverless. If you were to create a new one,
        # you should use ServerlessSpec instead of PodSpec.
        # For example:
        # pinecone.create_index(
        #     name=PINECONE_INDEX_NAME,
        #     dimension=EMBEDDING_DIMENSIONALITY,
        #     metric='cosine',
        #     spec=ServerlessSpec(cloud="aws", region="us-east-1") # Adjust cloud/region as needed
        # )
        # However, since the index already exists, this block should now be skipped.
        pinecone.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSIONALITY,
            metric='cosine',
            spec=PodSpec(environment=PINECONE_ENVIRONMENT) # This will likely fail if trying to create a serverless index with PodSpec
        )
        print(f"Index '{PINECONE_INDEX_NAME}' created.")
    else:
        print(f"Index '{PINECONE_INDEX_NAME}' already exists.")

    index = pinecone.Index(PINECONE_INDEX_NAME)

    all_turns = []
    file_count = 0

    # Recursively walk through the data directory
    print(f"\nStarting recursive scan of '{DATA_DIR}'...")
    for root, dirs, files in os.walk(DATA_DIR):
        print(f"  Current directory: {root}")
        print(f"  Subdirectories found: {dirs}")
        print(f"  Files found in current directory: {files}")
        for file_name in files:
            if file_name.endswith(".json"):
                file_path = os.path.join(root, file_name)
                file_count += 1
                print(f"Processing file: {file_path}")

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # Handle different JSON structures: list of messages or dict with 'messages' key
                    if isinstance(data, list):
                        messages = data
                    elif isinstance(data, dict):
                        messages = data.get('messages', [])
                    else:
                        print(f"Warning: Unexpected JSON structure in {file_path}. Skipping.")
                        continue
                    
                    # Filter out bot messages and clean content
                    human_messages = []
                    for msg in messages:
                        if not msg.get('author', {}).get('bot', False):
                            cleaned_content = clean_message_content(msg.get('content', ''))
                            if cleaned_content: # Only include messages with content after cleaning
                                human_messages.append({
                                    'author_name': msg.get('author', {}).get('global_name') or msg.get('author', {}).get('username', 'Unknown'),
                                    'content': cleaned_content,
                                    'timestamp': msg.get('timestamp')
                                })
                    
                    if not human_messages:
                        print(f"No human messages found or remaining after filtering in {file_name}. Skipping.")
                        continue

                    # Turn-based grouping
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
                        
                        # Add the last turn
                        all_turns.append(f"{current_turn_author}: {' '.join(current_turn_messages)}")

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from {file_path}: {e}")
                except Exception as e:
                    print(f"An unexpected error occurred while processing {file_path}: {e}")

    if not all_turns:
        print("No turns were generated from any files. Exiting.")
        return

    print(f"\nFinished processing {file_count} JSON files. Total turns generated: {len(all_turns)}")
    print("Creating sliding window chunks and generating embeddings...")

    all_text_chunks = []
    chunk_metadata = [] # To store original text_chunk and its ID
    chunk_id_counter = 0

    # Sliding window vectorization
    i = 0
    while i < len(all_turns):
        chunk_end = min(i + CHUNK_SIZE, len(all_turns))
        current_chunk_turns = all_turns[i:chunk_end]

        if not current_chunk_turns:
            break

        text_chunk = "\n".join(current_chunk_turns)
        
        # Store the text chunk and its future ID for batch processing
        all_text_chunks.append(text_chunk)
        chunk_metadata.append({"id": f"chunk-{chunk_id_counter}", "text": text_chunk})
        chunk_id_counter += 1
        
        # Move the window
        if chunk_end == len(all_turns): # Reached the end
            break
        else:
            i += (CHUNK_SIZE - OVERLAP_SIZE)
            if i >= len(all_turns): # Ensure we don't go past the end if overlap is too large
                break
            
    print(f"Generated {len(all_text_chunks)} text chunks for embedding.")

    vectors_to_upsert = []
    
    # Process embeddings in batches
    for i in range(0, len(all_text_chunks), EMBEDDING_BATCH_SIZE):
        batch_texts = all_text_chunks[i : i + EMBEDDING_BATCH_SIZE]
        batch_metadata = chunk_metadata[i : i + EMBEDDING_BATCH_SIZE]
        
        print(f"Generating embeddings for batch {i // EMBEDDING_BATCH_SIZE + 1}/{(len(all_text_chunks) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE} ({len(batch_texts)} chunks)...")
        
        try:
            batch_response = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=batch_texts,
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=EMBEDDING_DIMENSIONALITY
            )
            
            for j, embedding in enumerate(batch_response['embedding']):
                original_chunk_info = batch_metadata[j]
                vectors_to_upsert.append({
                    "id": original_chunk_info["id"],
                    "values": embedding,
                    "metadata": {"text": original_chunk_info["text"]}
                })
            print("creating a batch of embeddings...")
            time.sleep(30) # Add a 30-second delay after each batch
        except Exception as e:
            print(f"Error generating embeddings for a batch: {e}")
            # Depending on desired error handling, you might want to skip this batch
            # or retry. For now, we'll just log and continue.
            
    print(f"Generated {len(vectors_to_upsert)} vectors for upsert.")

    # Upsert to Pinecone
    if vectors_to_upsert:
        try:
            # Pinecone upsert can take a list of dictionaries
            index.upsert(vectors=vectors_to_upsert)
            print(f"Successfully upserted {len(vectors_to_upsert)} vectors to Pinecone index '{PINECONE_INDEX_NAME}'.")
        except Exception as e:
            print(f"Error during Pinecone upsert: {e}")
    else:
        print("No vectors to upsert.")

    print("\nIngestion process completed.")

# --- Main Execution ---
if __name__ == "__main__":
    # Create the @data directory if it doesn't exist for testing purposes
    if not os.path.exists(DATA_DIR):
        print(f"'{DATA_DIR}' directory not found. Creating it. Please place your Discord JSON files inside.")
        os.makedirs(DATA_DIR)
        # Example dummy JSON file for testing if the directory was just created
        dummy_data = {
            "messages": [
                {"author": {"name": "User1", "bot": False}, "content": "Hello everyone!", "timestamp": "2023-01-01T10:00:00.000Z"},
                {"author": {"name": "User1", "bot": False}, "content": "How are you all doing today?", "timestamp": "2023-01-01T10:01:00:000Z"},
                {"author": {"name": "Bot", "bot": True}, "content": "I am a bot.", "timestamp": "2023-01-01T10:02:00:000Z"},
                {"author": {"name": "User2", "bot": False}, "content": "I'm doing great, thanks!", "timestamp": "2023-01-01T10:03:00:000Z"},
                {"author": {"name": "User1", "bot": False}, "content": "That's good to hear. This is a bad word.", "timestamp": "2023-01-01T10:04:00:000Z"},
                {"author": {"name": "User2", "bot": False}, "content": "Indeed, it is a bad word.", "timestamp": "2023-01-01T10:05:00:000Z"},
                {"author": {"name": "User3", "bot": False}, "content": "New user here!", "timestamp": "2023-01-01T10:06:00:000Z"},
                {"author": {"name": "User3", "bot": False}, "content": "Glad to join.", "timestamp": "2023-01-01T10:07:00:000Z"},
                {"author": {"name": "User1", "bot": False}, "content": "Welcome!", "timestamp": "2023-01-01T10:08:00:000Z"},
                {"author": {"name": "User2", "bot": False}, "content": "Hi there!", "timestamp": "2023-01-01T10:09:00:000Z"},
                {"author": {"name": "User2", "bot": False}, "content": "Another message from user2.", "timestamp": "2023-01-01T10:10:00:000Z"},
                {"author": {"name": "User1", "bot": False}, "content": "Final message from user1.", "timestamp": "2023-01-01T10:11:00:000Z"},
            ]
        }
        with open(os.path.join(DATA_DIR, "dummy_discord_export.json"), 'w', encoding='utf-8') as f:
            json.dump(dummy_data, f, indent=2)
        print("Created a dummy JSON file for demonstration. Please configure your .env and run again.")
        print("Example .env content:")
        print("PINECONE_API_KEY=YOUR_PINECONE_API_KEY")
        print("PINECONE_ENVIRONMENT=YOUR_PINECONE_ENVIRONMENT")
        print("GEMINI_API_KEY=YOUR_GEMINI_API_KEY")
        print("FILTER_WORDS=bad word,another bad word")
    else:
        process_discord_data()