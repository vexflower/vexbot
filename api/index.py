from fastapi import FastAPI, Request, HTTPException
import os
from dotenv import load_dotenv
import google.generativeai as genai
import pinecone
import discord

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
ADMIN_SECRET = os.getenv("ADMIN_SECRET")

genai.configure(api_key=GEMINI_API_KEY)
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_ENV)

intents = discord.Intents.default()
intents.messages = True
intents.guilds = True
intents.message_content = True
client = discord.Client(intents=intents)

app = FastAPI()

@app.post("/api")
async def handle_request(request: Request):
    data = await request.json()
    action = data.get("action")
    secret = data.get("secret")

    if secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")

    if action == "admin_command":
        command = data.get("command")
        params = data.get("params")
        # Implement admin commands here
        return {"status": "admin command received"}
    elif action == "rag_query":
        # Implement RAG query here
        return {"status": "rag query received"}
    else:
        raise HTTPException(status_code=400, detail="Invalid action")

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith('!ping'):
        await message.channel.send('Pong!')
    elif message.content.startswith('!sync'):
        await message.channel.send('Syncing...')
    elif message.content.startswith('!purge'):
        # Implement purge logic
        await message.channel.send('Purging...')
    elif message.content.startswith('!analyze'):
        # Implement analyze logic
        await message.channel.send('Analyzing...')
    elif message.content.startswith('!ask'):
        # Implement ask logic
        await message.channel.send('Thinking...')

# It's not recommended to run the bot and the API in the same process for Vercel.
# This part is for local testing.
# if __name__ == "__main__":
#     import uvicorn
#     # To run the bot and the API together, you'd need to use asyncio.
#     # For simplicity, we're not doing that here.
#     # You would typically run the bot as a separate process.
#     # client.run(DISCORD_TOKEN)
#     uvicorn.run(app, host="0.0.0.0", port=8000)
