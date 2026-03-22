import os
import discord
from discord.ext import commands
from dotenv import load_dotenv

# --- Environment Setup ---
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

# --- Bot Initialization ---
# We use commands.Bot here for a slightly different, simpler setup 
# compared to the discord.Client + app_commands tree in run_bot.py
intents = discord.Intents.default()
intents.message_content = True
intents.members = True 

bot = commands.Bot(command_prefix="?", intents=intents, help_command=commands.DefaultHelpCommand())

# --- Events ---
@bot.event
async def on_ready():
    print(f'Purge Bot logged in as {bot.user} (ID: {bot.user.id})')
    print('Ready to purge user messages across the server.')
    print('Type ?purgeuser <user_id_or_mention_or_username> to begin.')
    print('------')

# --- Commands ---
@bot.command(name="purgeuser", help="Purges all messages from a specific user across all text channels.")
@commands.has_permissions(administrator=True) # Ensure only admins can run this
async def purge_user(ctx, target: str, limit: int = 1000):
    """
    Purges messages from a specific user across the entire server.
    :param target: User ID, mention, or exact username.
    :param limit: Maximum number of messages to search per channel (default 1000).
    """
    await ctx.send(f"🔍 Searching for user: `{target}` and starting purge process (Searching up to {limit} messages per channel)...")
    
    target_user = None

    # 1. Try to parse as Mention or ID
    try:
        if target.startswith('<@') and target.endswith('>'):
            target_id = int(target.replace('<@', '').replace('!', '').replace('>', ''))
        else:
            target_id = int(target)
        
        target_user = ctx.guild.get_member(target_id)
        if not target_user:
            target_user = await bot.fetch_user(target_id)
    except ValueError:
        # 2. If it's not a number/mention, try to find by exact username
        target_user = discord.utils.get(ctx.guild.members, name=target)
        if not target_user:
            # Also check global_name or display_name as a fallback
            target_user = discord.utils.find(lambda m: m.display_name == target or m.global_name == target, ctx.guild.members)

    if not target_user:
        await ctx.send("❌ Could not find that user. Please provide a valid User ID, Mention, or Exact Username.")
        return

    # Check function for the purge
    def is_target(message):
        return message.author.id == target_user.id

    total_deleted = 0
    failed_channels = []

    # Iterate through all text channels in the guild
    for channel in ctx.guild.text_channels:
        try:
            # Purge messages matching the check function
            deleted = await channel.purge(limit=limit, check=is_target)
            if deleted:
                total_deleted += len(deleted)
                print(f"Deleted {len(deleted)} messages in #{channel.name}")
        except discord.Forbidden:
            failed_channels.append(channel.name)
        except discord.HTTPException as e:
            print(f"HTTP Exception in #{channel.name}: {e}")
        except Exception as e:
             print(f"Unexpected error in #{channel.name}: {e}")

    # Final report
    report = f"✅ Finished purging. Deleted a total of **{total_deleted}** messages from **{target_user.name}**."
    if failed_channels:
        report += f"\n⚠️ Note: Missing permissions to read/manage messages in the following channels: {', '.join(failed_channels)}"
    
    await ctx.send(report)

@purge_user.error
async def purge_user_error(ctx, error):
    if isinstance(error, commands.MissingPermissions):
        await ctx.send("❌ You don't have the required permissions (Administrator) to use this command.")
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send("❌ Please specify a user! Usage: `?purgeuser <user_id_or_mention_or_username> [limit]`")
    else:
        await ctx.send(f"❌ An error occurred: {error}")

# --- Running the Bot ---
if __name__ == "__main__":
    if not DISCORD_TOKEN:
        print("Error: DISCORD_TOKEN is not set in the .env file.")
    else:
        bot.run(DISCORD_TOKEN)
