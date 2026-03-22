import random
import discord
import google.generativeai as genai

async def execute_roll(max_number: int) -> str:
    """Handles the logic for rolling a dice."""
    if max_number <= 0:
        return "Please provide a positive number."
    roll_result = random.randint(1, max_number)
    return f"🎲 You rolled a **{roll_result}** (1-{max_number})!"

async def execute_ask(prompt: str) -> str:
    """Handles asking a question to the Gemini model."""
    if not genai:
         return "Gemini API is not configured."
    try:
        model = genai.GenerativeModel('gemini-3.1-flash-lite-preview')
        response = await model.generate_content_async(prompt)
        text = response.text
        if len(text) > 1980:
             return f"**Question:** {prompt}\n\n**Answer:** {text[:1980]}..."
        return f"**Question:** {prompt}\n\n**Answer:** {text}"
    except Exception as e:
        return f"An error occurred: {e}"

def execute_ping(latency: float) -> str:
    """Handles the ping logic."""
    latency_ms = round(latency * 1000)
    return f"Pong! {latency_ms}ms"
