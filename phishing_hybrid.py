import csv
import re
from pathlib import Path

from transformers import pipeline


MODEL_ID = "dima806/phishing-email-detection"
REVISION = "310a5602280486e01dd60e37230980cfcc327b41"


def check_rules(text):
    """Return matching rule names. These are limited text heuristics."""
    text = text.lower()
    reasons = []

    credential_request = re.search(
        r"\b(send|reply|share|provide|email)\b"
        r".{0,100}\b(password|verification code|one-time code)\b",
        text,
    )

    gift_card_request = (
        re.search(r"\bgift cards?\b", text)
        and re.search(
            r"\b(send|email|text)\b.{0,100}\b(codes?|photos?|pictures?)\b",
            text,
        )
    )

    if credential_request:
        reasons.append("Credential-sharing request")

    if gift_card_request:
        reasons.append("Gift-card code request")

    return reasons


def main():
    print("Loading the same model revision as our baseline...")

    classifier = pipeline(
        "text-classification",
        model=MODEL_ID,
        revision=REVISION,
        device=-1,
        trust_remote_code=False,
        model_kwargs={"use_safetensors": True},
    )

    # New fictional examples: four suspicious and four legitimate.
    tests = [
        (
            "Credential request",
            "Suspicious",
            "Hello, this is the help desk. Please send your password "
            "and one-time code so we can restore your mailbox.",
        ),
        (
            "Gift card request",
            "Suspicious",
            "I'm visiting a client. Purchase two gift cards and text "
            "me the codes. I will pay you back tomorrow.",
        ),
        (
            "Payment diversion",
            "Suspicious",
            "Please pay our outstanding invoice to this replacement "
            "bank account today. Do not contact the accounting team.",
        ),
        (
            "Document lure",
            "Suspicious",
            "A confidential salary document is waiting. Open "
            "https://payroll-review.example and sign in with your "
            "work credentials to view your raise.",
        ),
        (
            "Security training",
            "Legitimate",
            "Never send your password or verification code to anyone. "
            "Our help desk will never ask for them.",
        ),
        (
            "Gift card safety notice",
            "Legitimate",
            "Scammers may ask you to buy gift cards and send the codes. "
            "Report these requests to the security team.",
        ),
        (
            "Urgent schedule change",
            "Legitimate",
            "Urgent schedule change: tomorrow's training starts at "
            "8 AM instead of 9 AM. No action is required.",
        ),
        (
            "Normal support instructions",
            "Legitimate",
            "Your support ticket is resolved. Open your usual company "
            "app to confirm access. Do not reply with credentials.",
        ),
    ]

    label_map = {
        "PHISHING EMAIL": "Suspicious",
        "SAVE EMAIL": "Legitimate",
    }

    rows = []

    for name, expected, text in tests:
        result = classifier(text, truncation=True)[0]
        ai_prediction = label_map[result["label"]]
        reasons = check_rules(text)

        # Rules can flag a message; they cannot clear an AI warning.
        hybrid_prediction = (
            "Suspicious" if reasons else ai_prediction
        )

        rows.append({
            "test": name,
            "text": text,
            "expected": expected,
            "ai_prediction": ai_prediction,
            "ai_score": float(result["score"]),
            "rule_matches": "; ".join(reasons),
            "hybrid_prediction": hybrid_prediction,
            "model": MODEL_ID,
            "revision": REVISION,
        })

        print(f"\n--- {name} ---")
        print(f"Expected: {expected}")
        print(f"AI: {ai_prediction}")
        print(f"Rules: {', '.join(reasons) if reasons else 'None'}")
        print(f"Hybrid: {hybrid_prediction}")

    print("\n--- COMPARISON ON THE SAME NEW EXAMPLES ---")

    for field, title in [
        ("ai_prediction", "AI alone"),
        ("hybrid_prediction", "AI plus rules"),
    ]:
        correct = sum(row[field] == row["expected"] for row in rows)
        false_positives = sum(
            row[field] == "Suspicious"
            and row["expected"] == "Legitimate"
            for row in rows
        )
        false_negatives = sum(
            row[field] == "Legitimate"
            and row["expected"] == "Suspicious"
            for row in rows
        )

        print(f"\n{title}")
        print(f"Correct: {correct}/{len(rows)}")
        print(f"False positives: {false_positives}")
        print(f"False negatives: {false_negatives}")

    output = Path(__file__).resolve().parent / "phishing_hybrid_results.csv"

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to: {output}")
    print("Small fictional test set; not real-world accuracy.")


if __name__ == "__main__":
    main()
