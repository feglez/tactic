# Collector: live Telegram channel monitoring and classification client

This directory contains the monitoring client used to collect the **TLiveUGR**
dataset snapshot. It is a Telethon-based tool that joins public Telegram
channels, logs their messages as JSON, and exposes a small Flask API that can
download a channel's recent history, run BERTopic over it, and classify the
channel with a locally hosted LLM served through Ollama.

## Components

| File | Purpose |
|---|---|
| `bot.py` | Telethon client: connects, joins channels, logs new channel messages (metadata, text, attachment hashes, external URLs) to `logs/<channel>.json`, and follows `t.me` links found in messages. Also provides on-demand history download (`download_messages`) and channel listing. |
| `join_channel.py` | Batch-joins every channel listed in a `channels.txt` file (one username or `t.me/...` link per line); appends successes to `joined_channels.txt`. |
| `channels_agent.py` | Scans the JSON logs for `t.me` links and joins any newly discovered channels, tracking them in `joined_channels.txt`. |
| `async_runner.py` | Background asyncio event loop so coroutines can be run safely from Flask request handlers. |
| `themes.py` | BERTopic stage: builds 10-message batches per channel, fits a topic model, and writes per-channel topic distributions (full-corpus and single-channel variants). |
| `channel_classifier.py` | LLM classification stage: builds a prompt from BERTopic results plus up to 40 sampled messages and queries a local Ollama server; parses the returned categories and stores results (`data/results.csv`, per-run result folders). |
| `app.py` | Flask API tying it all together: `/monitor`, `/download`, `/channels`, `/topics`, `/classify`, and `/category/<channel>` (download, then BERTopic, then LLM classification for a single channel). |

Note on the log format: the JSON message logs are written with the original
Spanish field names (`canal`, `mensajes`, `mensaje_id`, `fecha`, `hora`,
`emisor`, `texto`, ...). This is the on-disk format of the collected TLiveUGR
dataset and is what the pipeline's ingestion stage parses, so these keys are
kept as-is.

## Configuration

Credentials and endpoints are read from the environment (a `.env` file is
loaded via `python-dotenv`):

| Variable | Required | Meaning |
|---|---|---|
| `API_ID` | yes | Telegram API ID (from https://my.telegram.org). |
| `API_HASH` | yes | Telegram API hash. |
| `OLLAMA_MODEL` | yes (for classification) | Name of the model served by the local Ollama instance. |
| `OLLAMA_API_URL` | no | Ollama generate endpoint (default `http://localhost:11434/api/generate`). |

The Telethon session is stored locally as `monitor_bot.session`; never
distribute session files.

## Channel list

The set of channels to monitor is supplied by the user as a plain-text
`channels.txt` file (one channel username or `t.me` link per line), consumed by
`join_channel.py`. **The channel list studied in the paper is not distributed
in this repository.**

Runtime outputs (`logs/`, `data/`, `joined_channels.txt`,
`consolidated_channel_data.csv`, result folders) are generated locally and are
not distributed either.

## Running

```bash
pip install telethon python-dotenv flask flask-cors pandas bertopic requests tqdm colorlog
python join_channel.py        # join the channels in channels.txt
python bot.py                 # start live monitoring
python app.py                 # or: start the Flask API (port 5000)
```

## Responsible use

Collecting Telegram data with this tool must comply with the Telegram Terms
of Service and API rules, and with applicable data-protection law (e.g. GDPR)
and your institution's ethics requirements. Only monitor channels you are
permitted to study, minimize the personal data you store, and secure any
collected material.
