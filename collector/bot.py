from telethon import TelegramClient, events
from telethon.tl.functions.channels import JoinChannelRequest
from telethon.errors import UserAlreadyParticipantError
import json
from datetime import datetime
import asyncio
import os
from dotenv import load_dotenv
import re
import hashlib

load_dotenv(override=True)

# Configuration
api_id = os.environ.get("API_ID")
api_hash = os.environ.get("API_HASH")
session_name = 'monitor_bot'

client = TelegramClient(session_name, api_id, api_hash)

# Robust connection check
async def ensure_connection():
    if not client.is_connected():
        print("Connecting...")
        await client.connect()
    if not await client.is_user_authorized():
        await client.start()

# NOTE: the Spanish keys and sentinel values in the JSON logs written below
# ("canal", "mensajes", "mensaje_id", "fecha", "hora", "emisor", "texto",
# "desconocido", ...) are the on-disk log format of the collected TLiveUGR
# dataset and are parsed by the pipeline's ingestion stage. Do not rename them.

# Save a message (flat log format)
def save_message(channel_name, sender_name, message):
    os.makedirs('logs', exist_ok=True)
    message_dict = {
        "fecha": datetime.now().strftime("%Y-%m-%d"),
        "hora": datetime.now().strftime("%H:%M:%S"),
        "emisor": sender_name,
        "mensaje": message
    }
    log_file = os.path.join('logs', f"{channel_name}.json")

    try:
        with open(log_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = []

    data.append(message_dict)
    with open(log_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# List channels
async def list_channels():
    await ensure_connection()
    channels = []
    async for dialog in client.iter_dialogs():
        if dialog.is_channel and not dialog.is_user:
            channels.append(dialog.name)
    return channels

# Download a channel's messages
async def download_messages(channel, limit=100):
    from telethon.tl.functions.channels import JoinChannelRequest
    from telethon.errors import UserAlreadyParticipantError
    import json, os

    await ensure_connection()
    os.makedirs('logs', exist_ok=True)

    messages = []
    output_file = os.path.join('logs', f"{channel}_messages.json")

    # Load previously downloaded messages if any
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as file:
            try:
                messages = json.load(file)
            except json.JSONDecodeError:
                messages = []

    # Join the channel if not already a member
    try:
        await client(JoinChannelRequest(channel))
        print(f"✅ Joined channel {channel}")
    except UserAlreadyParticipantError:
        print(f"🔁 Already a member of channel {channel}")
    except Exception as e:
        return f"❌ Could not access or join channel '{channel}': {e}"

    # Resolve the channel entity
    try:
        entity = await client.get_entity(channel)
    except Exception as e:
        return f"❌ Could not resolve channel '{channel}': {e}"

    # Download messages
    try:
        async for message in client.iter_messages(entity, limit=limit):
            sender = await message.get_sender()
            sender_name = sender.username if sender and hasattr(sender, 'username') else "Desconocido"

            messages.append({
                "fecha": message.date.strftime("%Y-%m-%d"),
                "hora": message.date.strftime("%H:%M:%S"),
                "emisor": sender_name,
                "mensaje": message.message or "[Mensaje vacío]"
            })

        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(messages, file, ensure_ascii=False, indent=4)

        return f"{len(messages)} messages saved to {output_file}"

    except Exception as e:
        return f"❌ Error downloading messages from channel '{channel}': {e}"

async def join_channel(link):
    try:
        await client(JoinChannelRequest(link))
        print(f"✅ Successfully joined channel: {link}")
        with open("joined_channels.txt", "a", encoding="utf-8") as f:
            f.write(link + "\n")
    except UserAlreadyParticipantError:
        print(f"🔁 Already a member of channel: {link}")
    except Exception as e:
        print(f"❌ Error joining channel {link}: {e}")


async def get_channel_metadata(chat):
    return {
        "id": chat.id,
        "nombre": chat.title,
        "url": f"https://t.me/{chat.username}" if chat.username else "",
        "tipo": "privado" if chat.username is None else "público",
        "miembros": chat.participants_count if hasattr(chat, 'participants_count') else "desconocido"
    }

async def get_sender_metadata(sender, event):
    return {
        "id": sender.id if sender else "desconocido",
        "username": sender.username if sender and hasattr(sender, 'username') else "desconocido",
    }

def hash_file(file_bytes):
    sha256 = hashlib.sha256()
    sha256.update(file_bytes)
    return sha256.hexdigest()

async def save_full_message(event, channel_metadata, sender_metadata):
    os.makedirs('logs', exist_ok=True)
    log_file = os.path.join('logs', f"{channel_metadata['nombre']}.json")

    # Load the existing structure if the log already exists
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
                messages = data.get("mensajes", [])
            except json.JSONDecodeError:
                data = {"canal": channel_metadata, "mensajes": []}
                messages = data["mensajes"]
    else:
        data = {
            "canal": channel_metadata,
            "mensajes": []
        }
        messages = data["mensajes"]

    # Handle attachments if present
    attachments = []
    if event.message.media:
        file_bytes = await event.message.download_media(bytes)
        if file_bytes:
            attachments.append({
                "tipo": event.message.media.__class__.__name__,
                "nombre": event.message.file.name if event.message.file else "desconocido",
                "hash_sha256": hash_file(file_bytes)
            })

    new_message = {
        "mensaje_id": event.message.id,
        "fecha": datetime.now().strftime("%Y-%m-%d"),
        "hora": datetime.now().strftime("%H:%M:%S"),
        "emisor": sender_metadata,
        "mensaje": {
            "texto": event.message.message,
            "reenviado": event.message.fwd_from is not None,
            "editado": event.message.edit_date is not None,
            "adjuntos": attachments,
            "urls_externas": re.findall(r'(https?://\S+)', event.message.message or "")
        }
    }

    messages.append(new_message)

    # Write the structure back
    with open(log_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


# Watch for new messages
@client.on(events.NewMessage)
async def handler(event):
    if event.is_channel and not event.is_private:
        sender = await event.get_sender()
        chat = await event.get_chat()

        channel_metadata = await get_channel_metadata(chat)
        sender_metadata = await get_sender_metadata(sender, event)

        print(f"📨 Message received from channel: {channel_metadata['nombre']}")

        await save_full_message(event, channel_metadata, sender_metadata)

        links = re.findall(r'(?:https?://)?t\.me/(\w+)', event.message.message or "")
        for link in links:
            await join_channel(link)

# Start monitoring
async def start_monitoring():
    await ensure_connection()
    print("Monitoring messages... Ctrl+C to stop.")
    await client.run_until_disconnected()

# Direct-execution mode for testing
if __name__ == "__main__":
    with client:
        client.loop.run_until_complete(start_monitoring())
