# themes.py
import pandas as pd
from bertopic import BERTopic
import os
import json


def apply_bertopic():
    # Read the CSV
    df = pd.read_csv("merged_telegram_messages_preprocessed.csv", dtype={
        "from_scheduled": str, "edit_hide": str, "pinned": str, "noforwards": str,
        "fwd_from": str, "reply_to_scheduled": str, "forum_topic": str,
        "reply_markup": str, "replies": str, "post_author": str
    })
    df = df.dropna(subset=["message"])

    # Build batches
    df["batch"] = (df.groupby("URL").cumcount() // 10) + 1
    df_grouped = df.groupby(["URL", "batch"])["message"].apply(lambda x: " ".join(x)).reset_index()
    df_grouped.to_csv("batched_messages.csv", index=False)

    # Apply BERTopic
    messages = df_grouped["message"].tolist()
    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(messages)

    # Save intermediate results
    df_grouped["topic"] = topics
    df_grouped["topic_probability"] = probs
    df_grouped.to_csv("batched_messages_with_topics.csv", index=False)

    # Topic info
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv("extracted_topics.csv", index=False)

    # Per-channel distribution
    distrib = df_grouped.groupby(["URL", "topic"]).size().reset_index(name="count")
    total_batches = df_grouped.groupby("URL")["batch"].nunique().reset_index(name="total_batches")
    distrib = distrib.merge(total_batches, on="URL")
    distrib["topic_percentage"] = (distrib["count"] / distrib["total_batches"]) * 100
    distrib.to_csv("channel_topic_assignments_unfiltered.csv", index=False)

    # Filtering
    filtered_data = []
    for channel, group in distrib.groupby("URL"):
        temp_threshold = 25
        filtered_group = group[group["topic_percentage"] >= temp_threshold]
        if len(filtered_group) == 0:
            while temp_threshold > 0:
                temp_threshold -= 1
                filtered_group = group[group["topic_percentage"] >= temp_threshold]
                if len(filtered_group) >= 2 or temp_threshold == 0:
                    break
        filtered_data.append(filtered_group)

    distrib_filtered = pd.concat(filtered_data, ignore_index=True)
    distrib_filtered.to_csv("channel_topic_assignments_filtered.csv", index=False)

    # Keep only the filtered topics
    filtered_topics = distrib_filtered["topic"].unique()
    extracted_topics = pd.read_csv("extracted_topics.csv")
    filtered_extracted = extracted_topics[extracted_topics["Topic"].isin(filtered_topics)]
    filtered_extracted.to_csv("filtered_extracted_topics.csv", index=False)

    return "BERTopic finished and results saved."

def apply_bertopic_to_channel(channel_id):
    import pandas as pd
    import os
    import json
    from bertopic import BERTopic

    json_path = os.path.join("logs", f"{channel_id}_messages.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        messages_log = json.load(f)

    if not messages_log:
        raise ValueError(f"No messages in the log file for channel '{channel_id}'")

    df = pd.DataFrame(messages_log)
    df.insert(0, "URL", channel_id)
    df = df.rename(columns={"mensaje": "message"})
    df["message"] = df["message"].astype(str).str.strip()
    df = df.dropna(subset=["message"])
    df = df[df["message"] != ""]

    if df.empty:
        raise ValueError(f"No valid messages for channel '{channel_id}'")

    df["batch"] = (df.groupby("URL").cumcount() // 10) + 1
    df_grouped = df.groupby(["URL", "batch"])["message"].apply(lambda x: " ".join(x)).reset_index()
    messages = df_grouped["message"].tolist()

    # Drop batches that are empty after grouping
    messages = [msg for msg in messages if msg.strip()]

    if len(messages) < 2:
        raise ValueError(f"At least 2 message batches are required to apply BERTopic. Only {len(messages)} available")

    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(messages)

    df_grouped["topic"] = topics
    df_grouped["topic_probability"] = probs

    topic_info = topic_model.get_topic_info()
    channel_topic_distribution = df_grouped.groupby(["URL", "topic"]).size().reset_index(name="count")
    total_batches_per_channel = df_grouped.groupby("URL")["batch"].nunique().reset_index()
    total_batches_per_channel = total_batches_per_channel.rename(columns={"batch": "total_batches"})

    channel_topic_distribution = channel_topic_distribution.merge(total_batches_per_channel, on="URL")
    channel_topic_distribution["topic_percentage"] = (
        channel_topic_distribution["count"] / channel_topic_distribution["total_batches"] * 100
    )

    threshold = 25
    filtered_topic_group = channel_topic_distribution[channel_topic_distribution["topic_percentage"] >= threshold]
    if filtered_topic_group.empty:
        threshold = max(channel_topic_distribution["topic_percentage"].max(), 0)
        filtered_topic_group = channel_topic_distribution[channel_topic_distribution["topic_percentage"] >= threshold]

    filtered_topics = filtered_topic_group["topic"].unique()
    filtered_topic_info = topic_info[topic_info["Topic"].isin(filtered_topics)]

    topic_names = filtered_topic_info["Name"].tolist()
    representations = filtered_topic_info["Representation"].tolist()
    topic_percentages = filtered_topic_group.sort_values("topic")["topic_percentage"].tolist()

    consolidated = pd.DataFrame([{
        "Channel ID": channel_id,
        "topics": list(map(int, filtered_topics)),
        "topic_names": topic_names,
        "representations": representations,
        "topic_percentages": topic_percentages,
        "Category": ""
    }])

    output_path = "consolidated_channel_data.csv"
    try:
        existing = pd.read_csv(output_path)
        existing = existing[existing["Channel ID"].str.lower() != channel_id.lower()]
        combined = pd.concat([existing, consolidated], ignore_index=True)
    except FileNotFoundError:
        combined = consolidated

    combined.to_csv(output_path, index=False)
    return True
