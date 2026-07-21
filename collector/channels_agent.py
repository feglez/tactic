import os
import json
import re
import asyncio
from telethon import TelegramClient
from dotenv import load_dotenv
from telethon.tl.functions.channels import JoinChannelRequest, GetParticipantRequest

load_dotenv(override=True)

# Client configuration
api_id = os.environ.get("API_ID")   # Replace with your API ID
api_hash = os.environ.get("API_HASH")  # Replace with your API hash
session_name = 'monitor_bot'

# Folder where the JSON message logs are stored
logs_folder = 'logs'

joined_channels_file = 'joined_channels.txt'

# Create the Telegram client
client = TelegramClient(session_name, api_id, api_hash)

# Regular expression to detect Telegram channel links
telegram_link_pattern = re.compile(r"(?:https?://t\.me/|t\.me/|@)([\w\d_]+)")

# Read the list of channels the bot has already joined
def load_joined_channels():
    if not os.path.exists(joined_channels_file):
        return set()
    with open(joined_channels_file, 'r', encoding='utf-8') as file:
        return set(line.strip() for line in file)

# Record a newly joined channel in the joined-channels file
def save_joined_channel(channel):
    with open(joined_channels_file, 'a', encoding='utf-8') as file:
        file.write(f"{channel}\n")

# Check whether the bot is already a member of a channel
async def is_in_channel(channel):
    try:
        await client(GetParticipantRequest(channel, 'me'))
        return True  # The bot is already in the channel
    except:
        return False  # Not in the channel, or no access

async def find_and_join_channels():
    joined_channels = load_joined_channels()  # Load the channels already joined
    new_channels = set()

    # Walk the JSON files in the logs folder
    for filename in os.listdir(logs_folder):
        if filename.endswith('.json'):
            file_path = os.path.join(logs_folder, filename)

            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    messages = json.load(file)

                for message in messages:
                    text = message.get("mensaje", "")
                    matches = telegram_link_pattern.findall(text)

                    for channel in matches:
                        if channel in joined_channels or channel in new_channels:
                            continue

                        if await is_in_channel(channel):
                            print(f"✅ The bot is already in channel: {channel}")
                            joined_channels.add(channel)
                            save_joined_channel(channel)
                            continue

                        print(f"🔍 New channel detected: {channel}")
                        try:
                            await client(JoinChannelRequest(channel))
                            print(f"✅ Successfully joined channel: {channel}")
                            new_channels.add(channel)
                            save_joined_channel(channel)
                        except Exception as e:
                            print(f"❌ Error joining {channel}: {e}")

            except json.JSONDecodeError as e:
                print(f"⚠️ Corrupt file or invalid format: {file_path} | {e}")

# Run the script
async def main():
    print("🔎 Scanning messages for channel links...")
    await client.start()
    await find_and_join_channels()
    print("✅ Finished.")

if __name__ == '__main__':
    asyncio.run(main())
