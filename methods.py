import os
import random
import uuid
import discord

from pinecone import Pinecone
from google import genai
from google.genai import types

# --- API Initialization ---
# We load these from the environment variables set in run_bot.py
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "silly-parse"

# NEW: Load a custom persona from the .env file!
# If one isn't set, it defaults to this chill fallback persona.
BOT_PERSONA = os.getenv(
    "BOT_PERSONA",
    "You are a chill, helpful Discord bot acting like a peer in the server. Keep your responses natural, conversational, and match the tone of the users. Don't sound like a robot."
)

# Initialize our clients
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
pinecone = Pinecone(api_key=PINECONE_API_KEY) if PINECONE_API_KEY else None
index = pinecone.Index(PINECONE_INDEX_NAME) if pinecone else None

async def execute_roll(max_number: int) -> str:
    """Handles the logic for rolling a dice."""
    if max_number <= 0:
        return "Please provide a positive number."
    roll_result = random.randint(1, max_number)
    return f"🎲 You rolled a **{roll_result}** (1-{max_number})!"

def execute_ping(latency: float) -> str:
    """Handles the ping logic."""
    latency_ms = round(latency * 1000)
    return f"Pong! {latency_ms}ms"

async def execute_ask(prompt: str, images: list = None, short_term_history: str = "") -> str:
    """Queries Pinecone for context, then asks Gemini the question using that context (RAG). Handles images and short-term history."""
    if not gemini_client or not index:
        return "APIs are not fully configured. Please check your Gemini and Pinecone keys."

    try:
        # 1. Convert the user's question into an embedding vector
        emb_res = await gemini_client.aio.models.embed_content(
            model="models/gemini-embedding-001",
            contents=prompt or "image context", # Fallback if prompt is empty but image exists
            config=types.EmbedContentConfig(output_dimensionality=768)
        )
        query_vector = emb_res.embeddings[0].values

        # 2. Search Pinecone for the top 5 most similar message chunks
        search_results = index.query(
            vector=query_vector,
            top_k=5,
            include_metadata=True
        )

        # 3. Extract the text from the Pinecone results
        context_texts = []
        for match in search_results.matches:
            if 'text' in match.metadata:
                context_texts.append(match.metadata['text'])

        # Join them together with a separator
        context_str = "\n---\n".join(context_texts)

        # 4. Create the master prompt for Gemini using the custom persona and history
        system_prompt = f"""{BOT_PERSONA}
        
If the context provided below doesn't have the answer, just respond naturally to the best of your ability. Do NOT use prefixes like "Answer:" or "Bot:". Just speak.

[RECENT CHAT HISTORY (Immediate Context - Read this to understand the current conversation flow)]
{short_term_history}

[PINECONE MEMORY (Past Context)]
{context_str}"""

        # Combine text and images for the new SDK
        request_contents = [system_prompt, f"User's Message: {prompt}"]

        if images:
            for img in images:
                request_contents.append(
                    types.Part.from_bytes(data=img['data'], mime_type=img['mime_type'])
                )

        # 5. Ask Gemini using the async client
        response = await gemini_client.aio.models.generate_content(
            model='gemini-2.5-flash',
            contents=request_contents
        )

        text = response.text

        # Removed the "**Question:** \n\n **Answer:**" formatting
        # Just return the raw conversational text!
        if len(text) > 1980:
            return text[:1980] + "..."
        return text

    except Exception as e:
        return f"Uh oh, something broke in my brain: {e}"

async def ingest_live_message(author: str, content: str):
    """Listens to live chat and pushes it immediately into Pinecone."""
    if not gemini_client or not index:
        return

    try:
        # Format it exactly like your batch ingestor does
        text_chunk = f"{author}: {content}"

        # Generate embedding
        emb_res = await gemini_client.aio.models.embed_content(
            model="models/gemini-embedding-001",
            contents=text_chunk,
            config=types.EmbedContentConfig(output_dimensionality=768)
        )
        vector = emb_res.embeddings[0].values

        # Upsert single message into Pinecone with a random unique ID
        index.upsert([{
            "id": f"live-{uuid.uuid4()}",
            "values": vector,
            "metadata": {"text": text_chunk}
        }])
    except Exception as e:
        print(f"Failed to ingest live message: {e}")