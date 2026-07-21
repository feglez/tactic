"""Custom stopword list for the BERTopic CountVectorizer: standard English
stopwords plus Telegram-specific boilerplate terms."""

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

TELEGRAM_ARTIFACTS = {
    "channel", "group", "join", "admin", "user", "bot", "http", "https",
    "t.me", "telegram", "subscribe", "click", "link", "share", "message",
    "messages", "invite", "view", "joinchat", "posted", "username",
    "bio", "support", "follow", "reply", "dm", "pm", "forwarded", "available",
    "contact", "chat", "update", "shared", "report"
}

# `list()` over a set yields a process-dependent order (string hash
# randomization). This is harmless: CountVectorizer treats stop_words as a
# set, and the trainers pop stop_words from the params dict before
# hashing/logging, so neither results nor model names depend on this order.
STOPWORDS = list(ENGLISH_STOP_WORDS.union(TELEGRAM_ARTIFACTS))
