import csv
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from huggingface_hub import model_info
from transformers import pipeline


def main():
    model_id = "dima806/phishing-email-detection"
    run_time = datetime.now(timezone.utc).isoformat()

    print("Checking model revision...")
    revision = model_info(model_id).sha

    print("Loading phishing classifier...")
    classifier = pipeline(
        "text-classification",
        model=model_id,
        revision=revision,
        device=-1,
        trust_remote_code=False,
        model_kwargs={"use_safetensors": True},
    )

    # Fictional messages with labels based on the intended scenarios.
    tests = [
        {
            "name": "Legitimate urgency",
            "expected": "Legitimate",
            "text": (
                "URGENT: A water leak has closed our office. "
                "Please work from home today. Call your usual manager "
                "if you have questions."
            ),
        },
        {
            "name": "Legitimate password guidance",
            "expected": "Legitimate",
            "text": (
                "Security reminder: Never share your password or "
                "verification codes. To review your account, open "
                "the company portal using your saved bookmark."
            ),
        },
        {
            "name": "Subtle credential request",
            "expected": "Suspicious",
            "text": (
                "Hi Edward, we are updating the staff directory. "
                "Please reply with your work username, password, "
                "and current verification code so I can finish your entry."
            ),
        },
        {
            "name": "Gift card impersonation",
            "expected": "Suspicious",
            "text": (
                "Hi Edward, I'm in a meeting and need a quick favor. "
                "Buy three gift cards and email me photos of the codes. "
                "Keep this between us; I'll reimburse you later."
            ),
        },
        {
            "name": "Legitimate payment discussion",
            "expected": "Legitimate",
            "text": (
                "Hi Edward, the invoice total is incorrect. "
                "Please review it in our usual accounting system. "
                "We have not changed our bank details or payment process."
            ),
        },
        {
            "name": "Payment diversion attempt",
            "expected": "Suspicious",
            "text": (
                "Thanks for your business. Our bank account has changed. "
                "Send today's invoice payment to the new account below. "
                "Our phone lines are down, so please do not call to verify."
            ),
        },
    ]

    label_map = {
        "PHISHING EMAIL": "Suspicious",
        "SAVE EMAIL": "Legitimate",
    }

    rows = []
    correct = 0
    false_positives = 0
    false_negatives = 0
    unknown_labels = 0

    transformers_version = version("transformers")
    torch_version = version("torch")

    for number, test in enumerate(tests, start=1):
        result = classifier(test["text"], truncation=True)[0]

        raw_label = result["label"]
        prediction = label_map.get(raw_label, "Unknown")
        score = float(result["score"])
        matched = prediction == test["expected"]

        if matched:
            correct += 1
        elif prediction == "Unknown":
            unknown_labels += 1
        elif prediction == "Suspicious":
            false_positives += 1
        elif prediction == "Legitimate":
            false_negatives += 1

        rows.append({
            "test_number": number,
            "test_name": test["name"],
            "text": test["text"],
            "expected": test["expected"],
            "prediction": prediction,
            "raw_label": raw_label,
            "score": score,
            "correct": matched,
            "model": model_id,
            "revision": revision,
            "run_time_utc": run_time,
            "transformers_version": transformers_version,
            "torch_version": torch_version,
        })

        print(f"\n--- Test {number}: {test['name']} ---")
        print(f"Text: {test['text']}")
        print(f"Expected: {test['expected']}")
        print(f"Prediction: {prediction} ({raw_label})")
        print(f"Model score: {score:.3f}")
        print(f"Result: {'CORRECT' if matched else 'MISMATCH'}")

    # Save alongside this script, regardless of the Terminal directory.
    output_path = Path(__file__).resolve().parent / "phishing_results.csv"

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\n--- SUMMARY ---")
    print(f"Correct: {correct}/{len(tests)}")
    print(f"False positives (legitimate flagged): {false_positives}")
    print(f"False negatives (suspicious missed): {false_negatives}")
    print(f"Unknown model labels: {unknown_labels}")
    print("These six examples do not measure real-world accuracy.")
    print("Model scores are not guarantees of correctness.")
    print(f"\nModel revision: {revision}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
