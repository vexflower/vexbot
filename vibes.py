import os
import json
import random
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found in .env file.")
    exit(1)

client = genai.Client(api_key=GEMINI_API_KEY)
DATA_DIR = "data"

def analyze_vibe():
    print(f"Scanning '{DATA_DIR}' for chat history...")
    all_messages = []

    if not os.path.exists(DATA_DIR):
        print(f"Error: Could not find the '{DATA_DIR}' folder.")
        return

    # 1. Read all JSON files just like the ingestor did
    for root, dirs, files in os.walk(DATA_DIR):
        for file_name in files:
            if file_name.endswith(".json"):
                file_path = os.path.join(root, file_name)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    messages = data if isinstance(data, list) else data.get('messages', [])
                    for msg in messages:
                        # Ignore bot messages, grab human content
                        if not msg.get('author', {}).get('bot', False) and msg.get('content'):
                            author = msg.get('author', {}).get('global_name') or msg.get('author', {}).get('username', 'Unknown')
                            all_messages.append(f"{author}: {msg['content']}")
                except Exception as e:
                    print(f"Skipping {file_path} due to error: {e}")

    if not all_messages:
        print("No messages found to analyze!")
        return

    print(f"Found {len(all_messages)} total messages. Sampling up to 1500 for the vibe check...")

    # 2. Grab a random sample of messages (so we get a good mix of different days/conversations)
    sample_size = min(1500, len(all_messages))
    sampled_messages = random.sample(all_messages, sample_size)
    chat_transcript = "\n".join(sampled_messages)

    # 3. Ask Gemini to analyze the social dynamics and write the prompt
    prompt = f"""You are an expert at analyzing social dynamics and writing AI personas.
Read the following Discord chat transcript. Analyze the vibe, tone, inside jokes, slang, capitalization habits (e.g., do they use all lowercase?), and humor style of this specific friend group.

Then, write a customized system prompt for an AI that will join this server.
The AI must act like a natural peer, seamlessly adapting to their specific style. It should NEVER sound like an AI assistant, customer service rep, or a generic "chill bot."

Output ONLY the raw text that should go into the persona prompt. Do not include real names of people. Do not include introductory text. Make it detailed (1-2 paragraphs), specifying exactly how the bot should type and behave based on the transcript.

CHAT TRANSCRIPT:
{chat_transcript}
"""

    print("Analyzing your server's vibe with Gemini... (This might take a few seconds)")

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )

        # 4. Output the result for you to copy/paste!
        persona = response.text.strip().replace('\n', ' ')

        print("\n" + "="*50)
        print("✨ HERE IS YOUR CUSTOM SERVER PERSONA ✨")
        print("="*50 + "\n")
        print(f'BOT_PERSONA="{persona}"')
        print("\n" + "="*50)
        print("Copy the line above and paste it into your .env file!")

    except Exception as e:
        print(f"Error communicating with Gemini: {e}")

if __name__ == "__main__":
    analyze_vibe()