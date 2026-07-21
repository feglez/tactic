import asyncio
from telethon import TelegramClient
from telethon.tl.functions.channels import JoinChannelRequest
from telethon.errors import UserAlreadyParticipantError
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# Configuration
api_id = os.environ.get("API_ID")
api_hash = os.environ.get("API_HASH")
session_name = 'monitor_bot'

client = TelegramClient(session_name, api_id, api_hash)

async def join_channels_from_file():
    await client.start()

    # Read the channels.txt file
    try:
        with open("channels.txt", "r", encoding="utf-8") as f:
            channels = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("❌ channels.txt file not found")
        return

    for channel in channels:
        channel_input = channel

        # Extract just the identifier when a full link was given
        if "t.me/" in channel_input:
            channel_input = channel_input.split("t.me/")[-1]

        try:
            await client(JoinChannelRequest(channel_input))
            print(f"✅ Successfully joined: {channel_input}")
            with open("joined_channels.txt", "a", encoding="utf-8") as f:
                f.write(channel_input + "\n")
        except UserAlreadyParticipantError:
            print(f"🔁 Already a member of: {channel_input}")
        except Exception as e:
            print(f"❌ Error joining {channel_input}: {e}")

    await client.disconnect()

if __name__ == "__main__":
    asyncio.run(join_channels_from_file())
