from flask import Flask, request, jsonify
from bot import list_channels, download_messages, start_monitoring
from themes import apply_bertopic
from flask_cors import CORS
import pandas as pd
import asyncio
import threading
import logging
from channel_classifier import classify_channels
from channel_classifier import classify_single_channel
from bot import download_messages
from themes import apply_bertopic_to_channel
from async_runner import run_async_task

app = Flask(__name__)
CORS(app)

# Load the category CSV at startup. data/ is generated at runtime, so it does
# not exist yet on a fresh deployment.
try:
    categories_df = pd.read_csv("data/results.csv")
except FileNotFoundError:
    categories_df = pd.DataFrame(columns=["channel"])

@app.route("/monitor", methods=["POST"])
def monitor():
    def run_monitor():
        asyncio.run(start_monitoring())
    thread = threading.Thread(target=run_monitor, daemon=True)
    thread.start()
    return jsonify({"status": "Monitoring started in the background"})

@app.route("/download", methods=["POST"])
def download():
    data = request.get_json()
    channel = data.get("channel")
    limit = data.get("limit", 100)
    if not channel:
        return jsonify({"error": "Missing channel name"}), 400
    result = run_async_task(download_messages(channel, limit))
    return jsonify({"result": result})

@app.route("/channels", methods=["GET"])
def get_channels():
    channels = asyncio.run(list_channels())
    return jsonify(channels)

@app.route("/classify", methods=["POST"])
def classify():
    try:
        result_path = classify_channels()
        return jsonify({"status": "Classification finished", "result_path": result_path})
    except Exception as e:
        logging.exception("Error while classifying channels")
        return jsonify({"error": str(e)}), 500

@app.route("/topics", methods=["POST"])
def apply_topics():
    try:
        result = apply_bertopic()
        return jsonify({"status": result})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/category/<channel>", methods=["GET"])
def get_category(channel):
    channel = channel.strip().lower()

    from bot import download_messages
    from themes import apply_bertopic_to_channel
    from channel_classifier import classify_single_channel

    try:
        # Force a CSV reload in case it changed
        global categories_df
        categories_df = pd.read_csv("data/results.csv", on_bad_lines="skip")
        channel_row = categories_df[categories_df["channel"].str.lower() == channel]

        # Already classified: return it
        if not channel_row.empty:
            row = channel_row.iloc[0]
            return jsonify({
                "channel": row["channel"],
                "main_category": row["main_category"],
                "secondary_categories": row["secondary_categories"],
                "explanation": row["explanation"]
            })

        # 🧱 FULL FLOW: not classified yet, so run the whole process
        print(f"[🛠] Classifying new channel: {channel}")

        # 1. Download messages (awaited properly)
        download_result = run_async_task(download_messages(channel, 100))
        print(f"[✔] Download finished: {download_result}")

        # 2. Apply BERTopic
        apply_bertopic_to_channel(channel)
        print(f"[✔] BERTopic applied to {channel}")

        # 3. Classify the channel
        result_row = classify_single_channel(channel)
        print(f"[✔] Channel classified: {result_row}")

        if not result_row:
            return jsonify({"error": f"Could not classify channel '{channel}'"}), 404

        # 4. Reload and return the new result
        categories_df = pd.read_csv("data/results.csv")
        row = categories_df[categories_df["channel"].str.lower() == channel].iloc[0]

        return jsonify({
            "channel": row["channel"],
            "main_category": row["main_category"],
            "secondary_categories": row["secondary_categories"],
            "explanation": row["explanation"]
        })
    except ValueError as ve:
            # Specific case: not enough messages for BERTopic
            if "At least 2 message batches" in str(ve):
                logging.warning(f"[⚠️] Channel '{channel}' has too few messages to classify")
                df_result = pd.DataFrame([{
                    "channel": channel,
                    "main_category": "No data",
                    "secondary_categories": "[]",
                    "darkgram_category": ""
                }])
                try:
                    existing = pd.read_csv("data/results.csv")
                    existing = existing[existing["channel"].str.lower() != channel]
                    df_result = pd.concat([existing, df_result], ignore_index=True)
                except FileNotFoundError:
                    pass
                df_result.to_csv("data/results.csv", index=False)

                return jsonify({
                    "channel": channel,
                    "main_category": "No data",
                    "secondary_categories": []
                })

            else:
                import traceback
                traceback.print_exc()
                return jsonify({"error": str(ve)}), 500

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500



def run_async(coroutine):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(coroutine)

if __name__ == '__main__':
    app.run(debug=True)
