"""Message token-budgeting for the fixed-context LLM prompt.

A dry-run request (max 1 output token) measures the base prompt's token count.
The remaining context window, minus a variability margin, is converted to an
approximate character budget, and the caller's shuffled messages are greedily
packed whole until that budget (plus a small slack) is reached.
"""

from __future__ import annotations


def compute_message_budget(
    num_ctx: int,
    base_prompt_eval_count: int,
    variability_threshold: int,
    avg_chars_per_token: int,
):
    """Return (max_tokens_for_messages, approximate_char_budget)."""
    max_tokens_for_messages = num_ctx - base_prompt_eval_count - variability_threshold
    approximate_messages_length = max_tokens_for_messages * avg_chars_per_token
    return max_tokens_for_messages, approximate_messages_length


def select_messages(messages_list: list, approximate_messages_length: int, last_chars_budget: int):
    """Greedily pack whole messages up to the char budget (+slack).

    Iterates the already-shuffled message list and stops once the running
    length reaches the budget. A message is added only if it still fits within
    budget + last_chars_budget.

    Returns (selected_messages, longest_message_length, total_msg_length).
    """
    selected_messages = []
    total_msg_length = 0
    longest_message_length = 0
    for msg in messages_list:
        current_msg_length = len(msg)

        if total_msg_length >= approximate_messages_length:
            break

        if total_msg_length + current_msg_length <= approximate_messages_length + last_chars_budget:
            selected_messages.append(msg)
            if len(msg) > longest_message_length:
                longest_message_length = len(msg)
            total_msg_length += len(msg)

    return selected_messages, longest_message_length, total_msg_length
