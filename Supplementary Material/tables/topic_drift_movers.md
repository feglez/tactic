# Ten most-drifted topics between the 2022 and 2025 corpora

This table lists the ten topics of the frozen BERTopic model whose share moves
most between the two corpora, ordered by |Δ| (paper Section 5.5). Shares are
percentages of each corpus's micro-batches, not of its channels, and Δ is the
2025 share minus the 2022 one; the two corpora were collected by different
instruments, so Δ reflects population and collection method as well as elapsed
time (see the paper). Entries marked <0.1 are non-zero but round to zero at one
decimal, topic 7 is absent from TLiveUGR outright, and the analysis is
reproducible with `analysis/topic_drift.py`.

| Topic | Descriptors | 2022 | 2025 | Δ |
|---:|---|---:|---:|---:|
| 38 | `withdraw_instant_payment_earn` | 1.6% | 22.7% | +21.1 |
| 30 | `giftcards_paypall_paypal_cashapp` | 1.3% | 8.1% | +6.8 |
| 5 | `crying_sadness_depression_tears` | 3.0% | 9.4% | +6.4 |
| 15 | `maxx_maxed_m2_unlock` | 2.3% | 8.5% | +6.2 |
| 6 | `vaccines_vaccine_vaccinated` | 5.1% | <0.1% | −5.1 |
| 7 | `holocaust_reich_nazis_jews` | 4.7% | 0.0% | −4.7 |
| 13 | `elections_ballots_election_voters` | 4.4% | <0.1% | −4.4 |
| 16 | `subscription_hack_cracked_paypal` | 2.5% | 6.8% | +4.4 |
| 1 | `hindi_pune_marathi_maharashtra` | 4.6% | 0.9% | −3.8 |
| 8 | `pubgm_pubg_apk_bgmi` | 4.0% | 0.9% | −3.1 |
