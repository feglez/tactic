"""The analyst-defined cybercrime taxonomy used by the LLM classification stage.

It comprises 55 categories grouped into 9 thematic classes, delimited by the
section comments below.
"""

from __future__ import annotations

CYBERCRIME_CATEGORIES = [
    # Access, malware, infra
    "Hacking Tools & Services",
    "Exploit Kits",
    "Initial Access Brokers (IABs)",
    "Compromised RDP/VPN/SSH Access",
    "Botnet Panels",
    "DDoS-for-Hire",
    "Malware-as-a-Service (MaaS)",
    "Ransomware-as-a-Service (RaaS)",
    "Keyloggers & Stealers",
    "Infostealer Logs",
    "Bulletproof Hosting & Anonymous Infrastructure",

    # Credential abuse & phishing
    "Credential Stuffing Tools & Checkers",
    "Combo Lists & Wordlists",
    "Phishing Kits & Scampages",
    "OTP Bots & 2FA Bypass",

    # Accounts, payments, financial fraud
    "Account Shops (Stolen/Cracked)",
    "Carding",
    "CVV/Dumps/Fullz Shops",
    "Bank Drops & Mules",
    "Cashout Tutorials",
    "Refund/Chargeback Methods",
    "PayPal & Crypto Wallet Access",
    "SIM Swapping",

    # Data trading & privacy abuse
    "Leaked Databases",
    "PII/Fullz Packs",
    "Doxing Services",
    "Spyware & Stalkerware",

    # Social engineering & scams
    "Social Engineering Playbooks & Scripts",
    "eWhoring & Romance Scams",
    "Traffic/Install Brokerage & Malvertising",
    "SMTP/Bulk Mailing & Spam Services",

    # Sexual exploitation & adult-content abuse
    "Adult Content Leaks & Premium Porn Piracy",
    "Non-consensual Intimate Imagery (NCII/Revenge Porn)",
    "Deepfake Porn Tools & Trading",
    "Sextortion & Sexual Blackmail",
    "Voyeur/Hidden-Cam Leaks",
    "Cam Site Account Cracks & Trading",
    "Prostitution/Escort Brokerage",
    "Child Sexual Abuse Material (CSAM)",
    "Grooming/Luring of Minors",

    # Piracy & counterfeit (non-adult)
    "Software Piracy",
    "Streaming & IPTV Cracks",
    "eBook & Course Warez",
    "Counterfeit Goods",
    "Fake ID & Document Forgery",

    # Markets & illicit goods
    "Darknet Marketplaces",
    "Weapons & Explosives",
    "Drugs & Chemicals",

    # Influence / conflict
    "Propaganda Channels",
    "Terrorist Resources",
    "Cyber Warfare Playbooks",
    "Disinformation-as-a-Service",
    "SMM Panels & Fake Engagement",
    "Clickfarms & Device Farms",
    "Review & Rating Fraud",
]


def sorted_categories() -> list:
    """Alphabetically sorted taxonomy, exactly as inserted into the system prompt."""
    return sorted(CYBERCRIME_CATEGORIES)


def category_block() -> str:
    """The "- <category>" block substituted into the system prompt."""
    return "\n".join(f"- {c}" for c in sorted_categories())


def is_known_category(name: str) -> bool:
    """Return True if ``name`` is one of the taxonomy categories."""
    return name in CYBERCRIME_CATEGORIES
