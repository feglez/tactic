# channel_classifier.py
import pandas as pd
import requests
import json
import re
import random
from tqdm import tqdm
import ast
import os
import logging
from colorlog import ColoredFormatter

# Colored logging setup
log_format = '\n\n%(asctime)s - %(log_color)s%(levelname)s%(reset)s\n%(message)s\n\n'
formatter = ColoredFormatter(
    log_format,
    datefmt='%Y-%m-%d %H:%M:%S',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
)

handler = logging.StreamHandler()
handler.setFormatter(formatter)
logging.basicConfig(level=logging.INFO, handlers=[handler])

# Ollama configuration
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434/api/generate")
MODEL_NAME = os.environ.get("OLLAMA_MODEL")
random.seed(42)

def get_next_test_number(path):
    if not os.path.exists(path):
        return 1
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    test_numbers = [int(m.group(1)) for f in folders if (m := re.match(r'Jtest(\d+)_categories', f))]
    return max(test_numbers) + 1 if test_numbers else 1

def parse_list_column(col):
    def convert(x):
        if isinstance(x, list):
            return x
        try:
            return ast.literal_eval(x)
        except Exception:
            logging.warning(f"❌ Could not convert to list: {x}")
            return []
    return col.apply(convert)


def classify_channels():
    # Load data
    try:
        channel_data_df = pd.read_csv("consolidated_channel_data.csv", encoding='utf-8')
        messages_df = pd.read_csv("merged_telegram_messages_preprocessed.csv", encoding='utf-8', low_memory=False)
    except FileNotFoundError as e:
        logging.error(f"File not found: {e.filename}")
        raise

    # Parse list columns
    for col in ['topics', 'topic_names', 'representations', 'topic_percentages']:
        channel_data_df[col] = parse_list_column(channel_data_df[col])

    # Base prompt
    prompt_template = """You are a cybersecurity expert tasked with classifying Telegram channels based on their publications to identify cybercriminal activities. For each channel, you will be provided with the following information:

1. **BERTopic analysis results**: {bertopic_results}
   - Includes one main topic and 0 or more secondary topics, each with a generated name (e.g., "-1_info_bank_cc_bin") and representative keywords (e.g., ['info', 'bank', 'cc', 'bin']).
2. **Sample of 40 random publications**: {sampled_messages_str}
   - A sample of up to 40 randomly selected messages from the channel.
3. **Channel name**: {channel_id}
4. **Existing categories**: {existing_categories_str}
   - A list of categories already assigned to other channels.

### Instructions:
- First, analyze the data and think through your reasoning step by step **inside <think> tags**.
- After reasoning, provide **the final output** outside the <think> tags.
- **Classification Task**:
  - Assign **one main category** based on the main BERTopic topic.
  - Assign **exactly one secondary category** for each secondary BERTopic topic provided.
  - If there are no secondary topics, assign only the main category.
- **Category Assignment**:
  - **Prioritize BERTopic results**: use topic names and keywords as the primary basis for classification.
  - **Confirm with sample publications**: review the sample publications to confirm or adjust your choices. Override the BERTopic-based selection only if the sample shows a clear mismatch.
  - **Minimize channel name weight**: use the channel name only if it clearly supports the classification, as it may be misleading.
  - **Reuse existing categories**: prioritize reusing existing categories like "DDoS and Botnets" if they accurately reflect the channel’s cybercriminal focus. If similar but distinct, create a new category that is broad and crime-related.
  - **Create new categories if needed**: if no existing category fits, create a new, concise category that explicitly ties to cybercriminal activities (e.g., "Ransomware Operations", "Exploit Kits").
  - **Ensure cybercrime relevance**: all categories must describe cybercriminal activities. Do not use categories that reflect legitimate or non-criminal activities (e.g., "Cloud Storage Services") unless explicitly linked to illicit use, such as "Cloud Storage for Malware Hosting."
  - **Prefer general categories**: use broad, cybercrime-related categories that capture the overall theme (e.g., "Financial Fraud" instead of "Credit Card BIN Sales").
- **Output Format**:
  - If secondary topics are present: `("Main Category", "Secondary Category1", "Secondary Category2", ...)`
  - If no secondary topics: `("Main Category")`
  - The number of secondary categories must match the number of secondary topics. Do not add extra categories or leave them blank.
  - Even if evidence is weak, assign categories that reflect potential cybercriminal intent based on available data.
- **Guidelines**:
  - Always assign a main category; do not use placeholders like "None".
  - Ensure category names are plain text, with no special characters or formatting.
  - For non-English messages, translate to English and classify based on the translation.
  - Avoid categories describing legitimate tools, services, or business activities (e.g., "Email Marketing Tools") unless they are clearly exploited for cybercrime, such as "Email Marketing for Phishing."

Classify this channel now."""  # Replace with the full text if desired

    # Bookkeeping
    classified_channels = []
    channel_messages = {}
    full_answers = {}
    result_path = os.path.join(".", "results", "free_classification")
    existing_categories = set([
        "Compromised user credentials",
        "Pirated software",
        "Pirated media",
        "Social media manipulation tools",
        "Blackhat hacking resources"
    ])

    for _, row in tqdm(channel_data_df.iterrows(), total=len(channel_data_df), desc="Classifying channels", unit="channel"):
        channel_id = row['Channel ID']
        topic_names = row['topic_names']
        representations = row['representations']
        original_category = row['Category']

        bertopic_results = f"Main topic: {topic_names[0]}, Keywords: {representations[0]}"
        if len(topic_names) > 1:
            bertopic_results += "\n" + "\n".join(
                [f"Secondary topic: {t}, Keywords: {r}" for t, r in zip(topic_names[1:], representations[1:])]
            )

        # Samples
        channel_msgs = messages_df[messages_df['Channel ID'] == channel_id]['message'].dropna().tolist()
        sampled_messages = random.sample(channel_msgs, min(40, len(channel_msgs))) if channel_msgs else ["No messages available"]
        sampled_messages_str = "\n\n".join(sampled_messages)

        formatted_messages = f"***********{channel_id}**************\n"
        for msg in sampled_messages:
            formatted_messages += f"******* START OF MESSAGE *******\n{msg}\n******* END OF MESSAGE *******\n\n\n\n\n\n"
        channel_messages[channel_id] = formatted_messages

        # Send to the API
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt_template.format(
                bertopic_results=bertopic_results,
                sampled_messages_str=sampled_messages_str,
                channel_id=channel_id,
                existing_categories_str=str(list(existing_categories))
            ),
            "stream": False
        }

        try:
            response = requests.post(OLLAMA_API_URL, json=payload)
            response.raise_for_status()
            result = response.json()
            classification = re.sub(r'<.*?>', '', result.get("response", "").strip())
            classification = classification.split("\n")[-1].strip()

            category_match = re.findall(r'"(.+?)"', classification)
            categories = [cat.strip() for cat in category_match]
            main_category = categories[0] if categories else "Error in classification format"
            secondary_categories = categories[1:] if len(categories) > 1 else []

            existing_categories.update(categories)

            logging.info(f"Channel: {channel_id}\n  Main category: {main_category}\n  Secondary: {secondary_categories}\n  Original: {original_category}")

        except Exception as e:
            logging.error(f"{channel_id}: Classification failed: {e}")
            classification = "API error"
            main_category = "API error"
            secondary_categories = []

        full_answers[channel_id] = classification
        classified_channels.append({
            "channel": channel_id,
            "main_category": main_category,
            "secondary_categories": secondary_categories,
            "darkgram_category": original_category
        })

    # Save results
    classified_df = pd.DataFrame(classified_channels)
    if classified_df.empty:
        logging.info("No channels were classified.")
        return None

    N = get_next_test_number(result_path)
    folder_name = f"Jtest{N}_categories"
    full_path = os.path.join(result_path, folder_name)
    os.makedirs(full_path, exist_ok=True)
    channel_messages_folder = os.path.join(full_path, "channel_messages_evaluated")
    os.makedirs(channel_messages_folder, exist_ok=True)

    for channel_id in channel_messages:
        channel_folder = os.path.join(channel_messages_folder, re.sub(r'[<>:\"/\\\\|?*]', '_', channel_id))
        os.makedirs(channel_folder, exist_ok=True)
        with open(os.path.join(channel_folder, "messages.txt"), "w", encoding="utf-8") as f:
            f.write(channel_messages[channel_id])
        with open(os.path.join(channel_folder, "full_answer.txt"), "w", encoding="utf-8") as f:
            f.write(full_answers.get(channel_id, "No answer available"))

    with open(os.path.join(full_path, "prompt.txt"), "w", encoding="utf-8") as f:
        f.write(prompt_template)

    classified_df.to_csv(os.path.join(full_path, "results.csv"), index=False)

    summary_data = {
        "Total channels": len(classified_df),
        "Error in classification format": (classified_df['main_category'] == "Error in classification format").sum(),
        "Error parsing response": (classified_df['main_category'] == "Error parsing response").sum(),
        "API error": (classified_df['main_category'] == "API error").sum()
    }

    summary_df = pd.DataFrame([summary_data])
    summary_df.to_csv(os.path.join(full_path, "summary.csv"), index=False)

    logging.info(f"Classification finished. Results at: {full_path}/results.csv")

    return full_path  # <- returns the path so the API can use it

def classify_single_channel(channel_id):
    import json
    import re
    import logging
    import random
    import pandas as pd
    import os
    import requests
    from channel_classifier import parse_list_column

    prompt_template = """
You are a cybersecurity expert tasked with classifying Telegram channels based on their publications to identify cybercriminal activities. For each channel, you will be provided with the following information:

1. **BERTopic analysis results**: {bertopic_results}
   - Includes one main topic and 0 or more secondary topics, each with a generated name (e.g., "-1_info_bank_cc_bin") and representative keywords (e.g., ['info', 'bank', 'cc', 'bin']).
2. **Sample of 40 random publications**: {sampled_messages_str}
   - A sample of up to 40 randomly selected messages from the channel.
3. **Channel name**: {channel_id}
4. **Existing categories**: {existing_categories_str}
   - A list of categories already assigned to other channels.

### Instructions:
- First, analyze the data and think through your reasoning step by step **inside <think> tags**.
- After reasoning, provide:
  1. A brief explanation of your main category choice (1–3 sentences), make a professional explanation about that category, dont tell anything about what you think. Avoid things like "Okay, so I'm trying to figure out how to categorize this channel called "examplechannel" based on the data provided. The user has given me a sample of messages from the channel and some existing categories that other channels have been assigned. My task is to analyze the content of these messages and determine the appropriate category or categories for this channel. First, I'll read through the messages carefully"
  2. Then the final output in this format:
     - If secondary topics are present: ("Main Category", "Secondary Category1", ...)
     - If no secondary topics: ("Main Category")

Only include the explanation first, then the output on a **separate line at the end**.
"""

    try:
        channel_data_df = pd.read_csv("consolidated_channel_data.csv", encoding='utf-8')

        log_path = f"logs/{channel_id}_messages.json"
        if not os.path.exists(log_path):
            logging.error(f"Message log file not found for channel: {channel_id}")
            return None

        with open(log_path, "r", encoding="utf-8") as f:
            messages_log = json.load(f)

        messages_df = pd.DataFrame(messages_log)
        messages_df["Channel ID"] = channel_id

    except FileNotFoundError as e:
        logging.error(f"File not found: {e.filename}")
        raise

    for col in ['topics', 'topic_names', 'representations', 'topic_percentages']:
        channel_data_df[col] = parse_list_column(channel_data_df[col])

    row = channel_data_df[channel_data_df['Channel ID'].str.lower() == channel_id.lower()]
    if row.empty:
        logging.warning(f"Channel '{channel_id}' not found in consolidated_channel_data.csv, assigning an empty category.")
        topic_names = ["No topic"]
        representations = ["[]"]
        original_category = ""
    else:
        row = row.iloc[0]
        topic_names = row['topic_names']
        representations = row['representations']
        original_category = row['Category']

    bertopic_results = f"Main topic: {topic_names[0]}, Keywords: {representations[0]}"
    if len(topic_names) > 1:
        bertopic_results += "\n" + "\n".join(
            [f"Secondary topic: {t}, Keywords: {r}" for t, r in zip(topic_names[1:], representations[1:])]
        )

    channel_msgs = messages_df[messages_df['Channel ID'].str.lower() == channel_id.lower()]['mensaje'].dropna().tolist()
    sampled_messages = random.sample(channel_msgs, min(40, len(channel_msgs))) if channel_msgs else ["No messages available"]
    sampled_messages_str = "\n\n".join(sampled_messages)

    existing_categories = {
        "Compromised user credentials", "Pirated software", "Pirated media",
        "Social media manipulation tools", "Blackhat hacking resources"
    }

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt_template.format(
            bertopic_results=bertopic_results,
            sampled_messages_str=sampled_messages_str,
            channel_id=channel_id,
            existing_categories_str=str(list(existing_categories))
        ),
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        result = response.json()

        full_response = re.sub(r'<.*?>', '', result.get("response", "").strip())
        lines = full_response.splitlines()
        explanation = "\n".join(lines[:-1]).strip()
        classification_line = lines[-1].strip()

        category_match = re.findall(r'"(.+?)"', classification_line)
        categories = [cat.strip() for cat in category_match]
        main_category = categories[0] if categories else "Error in classification format"
        secondary_categories = categories[1:] if len(categories) > 1 else []

        prompt_preview = payload['prompt'][:700] + "..." if len(payload['prompt']) > 700 else payload['prompt']
        logging.info(f"[🧠 PROMPT SENT TO OLLAMA]\n{prompt_preview}")
        logging.info(f"[📩 RAW OLLAMA RESPONSE]\n{full_response}")
        logging.info(f"[📝 EXPLANATION]: {explanation}")

    except requests.exceptions.HTTPError as http_err:
        logging.error(f"[❌ HTTP ERROR] {http_err.response.status_code} - {http_err.response.text}")
        main_category = "API error"
        secondary_categories = []
        explanation = "No explanation due to API error."
    except Exception as e:
        logging.exception(f"[❌ UNHANDLED EXCEPTION] Failed to classify channel '{channel_id}': {e}")
        main_category = "API error"
        secondary_categories = []
        explanation = "No explanation due to general exception."

    result_row = {
        "channel": channel_id,
        "main_category": main_category,
        "secondary_categories": str(secondary_categories),
        "darkgram_category": original_category,
        "explanation": explanation
    }

    results_path = "data/results.csv"
    try:
        df_result = pd.read_csv(results_path)

        # Drop badly encoded columns if present
        columns_to_remove = [
            'main+AF8-category', 'secondary+AF8-categories',
            'darkgram+AF8-category'
        ]
        df_result = df_result.drop(columns=[col for col in columns_to_remove if col in df_result.columns], errors='ignore')

        # Replace the row if it already exists
        df_result = df_result[df_result["channel"].str.lower() != channel_id.lower()]
        df_result = pd.concat([df_result, pd.DataFrame([result_row])], ignore_index=True)

    except FileNotFoundError:
        df_result = pd.DataFrame([result_row])

    df_result.to_csv(results_path, index=False)
    logging.info(f"Channel '{channel_id}' classified as '{main_category}'. Saved to results.csv")
    return result_row
