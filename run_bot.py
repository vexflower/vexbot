import os
from datetime import timedelta

from dotenv import load_dotenv
import discord
from discord import app_commands
from discord.utils import get
import google.generativeai as genai
import methods

# --- Environment and API Setup ---
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


# --- Bot Definition ---
class VexBot(discord.Client):
    def __init__(self, *, bot_intents: discord.Intents):
        super().__init__(intents=bot_intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        try:
            # Global sync (can take up to ~1 hour to propagate)
            synced = await self.tree.sync()
            print(f"Synced {len(synced)} command(s) globally.")
        except Exception as e:
            print(f"Failed to sync commands: {e}")

# --- Bot Initialization ---
intents = discord.Intents.default()
intents.message_content = True
intents.members = True # Required for member management

# Add a configurable text command prefix (can be overridden in .env)
COMMAND_PREFIX = os.getenv("COMMAND_PREFIX", "!")

client = VexBot(bot_intents=intents)

# --- Events ---
@client.event
async def on_ready():
    print(f'Logged in as {client.user} (ID: {client.user.id})')
    print('------')

# --- Slash Commands ---

@client.tree.command(description="Roll a dice.")
@app_commands.describe(number="The maximum number to roll (defaults to 100).")
async def roll(interaction: discord.Interaction, number: int = 100):
    result = await methods.execute_roll(number)
    if "Please provide a positive number." in result:
        await interaction.response.send_message(result, ephemeral=True)
    else:
        await interaction.response.send_message(result)

@client.tree.command(description="Ask a question to the Gemini model.")
@app_commands.describe(prompt="The question you want to ask.")
async def ask(interaction: discord.Interaction, prompt: str):
    await interaction.response.defer(thinking=True)
    result = await methods.execute_ask(prompt)
    if "An error occurred" in result or "not configured" in result:
         await interaction.followup.send(result, ephemeral=True)
    else:
         await interaction.followup.send(result)


# --- Moderation Commands ---

@client.tree.command(description="Kicks a user from the server.")
@app_commands.describe(member="The user to kick.", reason="The reason for kicking.")
@app_commands.checks.has_permissions(kick_members=True)
async def kick(interaction: discord.Interaction, member: discord.Member, reason: str = "No reason provided."):
    if member == interaction.user:
        await interaction.response.send_message("You cannot kick yourself.", ephemeral=True)
        return
    await member.kick(reason=reason)
    await interaction.response.send_message(f"👢 Kicked {member.mention} for: {reason}.")

@client.tree.command(description="Bans a user from the server.")
@app_commands.describe(member="The user to ban.", reason="The reason for banning.")
@app_commands.checks.has_permissions(ban_members=True)
async def ban(interaction: discord.Interaction, member: discord.Member, reason: str = "No reason provided."):
    if member == interaction.user:
        await interaction.response.send_message("You cannot ban yourself.", ephemeral=True)
        return
    await member.ban(reason=reason)
    await interaction.response.send_message(f"🔨 Banned {member.mention} for: {reason}.")

@client.tree.command(description="Mutes a user for a specified duration.")
@app_commands.describe(member="The user to mute.", duration_minutes="Mute duration in minutes.", reason="The reason for muting.")
@app_commands.checks.has_permissions(moderate_members=True)
async def mute(interaction: discord.Interaction, member: discord.Member, duration_minutes: int, reason: str = "No reason provided."):
    if member == interaction.user:
        await interaction.response.send_message("You cannot mute yourself.", ephemeral=True)
        return
    duration = discord.utils.utcnow() + timedelta(minutes=duration_minutes)
    await member.timeout(duration, reason=reason)
    await interaction.response.send_message(f"🔇 Muted {member.mention} for {duration_minutes} minutes. Reason: {reason}.")

@client.tree.command(description="Places a user in quarantine.")
@app_commands.describe(member="The user to quarantine.", reason="The reason for quarantining.")
@app_commands.checks.has_permissions(manage_roles=True)
async def quarantine(interaction: discord.Interaction, member: discord.Member, reason: str = "No reason provided."):
    if member == interaction.user:
        await interaction.response.send_message("You cannot quarantine yourself.", ephemeral=True)
        return
        
    guild = interaction.guild
    quarantine_role = get(guild.roles, name="Quarantine")

    if not quarantine_role:
        # Create the role if it doesn't exist
        quarantine_role = await guild.create_role(name="Quarantine", reason="Creating quarantine role for bot.")
        # Deny permission to see channels for the new role
        for channel in guild.channels:
            await channel.set_permissions(quarantine_role, view_channel=False)

    await member.add_roles(quarantine_role, reason=reason)
    await interaction.response.send_message(f"☣️ Quarantined {member.mention}. Reason: {reason}.")


# --- Error Handling for Commands ---
@client.tree.error
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    if isinstance(error, app_commands.MissingPermissions):
        await interaction.response.send_message("You don't have the required permissions to use this command.", ephemeral=True)
    else:
        await interaction.response.send_message(f"An unexpected error occurred: {error}", ephemeral=True)
        raise error

# --- Message Commands (Prefix and Suffix Style) ---
# Add message-based command handling to support prefix (e.g. !ping) and simple suffix styles (e.g. ping!)
@client.event
async def on_message(message: discord.Message):
    # Ignore messages from bots (including ourselves)
    if message.author.bot:
        return

    content = message.content.strip()
    prefix = COMMAND_PREFIX

    # Helper: send truncated content safely
    async def safe_send(channel, text):
        if len(text) > 1900:
            await channel.send(text[:1900] + "...")
        else:
            await channel.send(text)

    # Process prefix-style commands: e.g. !ping, !roll 20, !ask What is AI?
    if content.startswith(prefix):
        parts = content[len(prefix):].split()
        if not parts:
            return
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd == "ping":
            await message.channel.send(methods.execute_ping(client.latency))
            return

        if cmd == "roll":
            max_n = 100
            if args:
                try:
                    max_n = max(1, int(args[0]))
                except ValueError:
                    await message.channel.send("Please provide a valid positive integer for roll, e.g. `!roll 20`.")
                    return
            result = await methods.execute_roll(max_n)
            await message.channel.send(result)
            return

        if cmd == "ask":
            if not args:
                await message.channel.send("Usage: `!ask <your question>`")
                return
            prompt = " ".join(args)
            async with message.channel.typing():
                result = await methods.execute_ask(prompt)
                await safe_send(message.channel, result)
            return

        if cmd in ("help", "commands"):
            help_text = (
                f"Available commands (prefix: `{prefix}`):\n"
                "`!ping` — check latency.\n"
                "`!roll [max]` — roll a random number between 1 and max (default 100).\n"
                "`!ask <question>` — ask the configured Gemini model a question.\n"
                "Also available as slash commands (use / in Discord)."
            )
            await message.channel.send(help_text)
            return

        # Unknown prefix command: ignore or give a hint
        # (Avoid being noisy — only respond when user asks for help or uses a known command.)
        return

    # Process simple suffix-style commands like 'ping!' or 'roll 20!'
    # We'll treat trailing punctuation like '!' or '?' as triggers when the first token matches a command.
    # Example: "ping!" -> ping command, "roll 20!" -> roll command.
    if content.endswith("!") or content.endswith("?"):
        stripped = content.rstrip('!?.').strip()
        if not stripped:
            return
        parts = stripped.split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd == "ping":
            await message.channel.send(methods.execute_ping(client.latency))
            return
        if cmd == "roll":
            max_n = 100
            if args:
                try:
                    max_n = max(1, int(args[0]))
                except ValueError:
                    await message.channel.send("Please provide a valid positive integer for roll, e.g. `roll 20!`.")
                    return
            result = await methods.execute_roll(max_n)
            await message.channel.send(result)
            return
        if cmd == "ask":
            if not args:
                await message.channel.send("Usage: `ask <your question>!`")
                return
            prompt = " ".join(args)
            async with message.channel.typing():
                result = await methods.execute_ask(prompt)
                await safe_send(message.channel, result)
            return

    # If nothing matched, do nothing; keep slash commands and other events working.
 
# --- Running the Bot ---
if __name__ == "__main__":
    if not DISCORD_TOKEN:
        print("Error: DISCORD_TOKEN is not set in the .env file.")
    else:
        client.run(DISCORD_TOKEN)