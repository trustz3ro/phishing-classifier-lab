# Phishing Email Classification Lab

A Python learning project evaluating a pretrained phishing-email
classifier and comparing its predictions with a simple rule-assisted
approach.

## Purpose

Explore how AI can support phishing detection, identify classification
errors, and test whether basic detection rules improve results.

## Tools

- Python
- Hugging Face Transformers and Hub
- PyTorch
- Regular expressions
- CSV results logging

## Model

Model: dima806/phishing-email-detection

Recorded revision:
310a5602280486e01dd60e37230980cfcc327b41

The model's labels are mapped as follows:

- PHISHING EMAIL → Suspicious
- SAVE EMAIL → Legitimate

## Method

1. Ran an initial smoke test with four straightforward fictional messages.
2. Evaluated six harder fictional messages and saved the predictions.
3. Added rules for credential-sharing and gift-card-code requests.
4. Compared AI alone with AI plus rules on eight new fictional messages.

Expected labels were manually assigned based on each intended scenario.
These small, hand-written sets are development examples, not a
representative benchmark.

## Results

| Experiment | Correct | False positives | False negatives |
|---|---:|---:|---:|
| Initial smoke test | 4/4 | 0 | 0 |
| Harder baseline | 3/6 | 1 | 2 |
| New examples: AI alone | 4/8 | 3 | 1 |
| New examples: AI plus rules | 4/8 | 3 | 1 |

The baseline and hybrid experiments used different messages.
Their totals should not be used to claim an improvement or decline.
The valid direct comparison is AI alone versus AI plus rules on
the same eight new examples.

## Findings

- The baseline model missed credential theft and gift-card impersonation.
- Incorrect baseline predictions received scores above 0.99.
- On the new examples, both approaches missed a payroll document lure.
- Both approaches flagged legitimate security and support messages.
- The rules matched scam-related phrases inside safety advice.
- The rules changed no predictions on the eight-example comparison set.

## Limitations

- Model confidence does not guarantee correctness.
- Text alone omits sender identity, headers, and destination verification.
- Simple phrase rules do not reliably understand negation or context.
- These results do not establish real-world detection accuracy.
- This is an educational prototype, not a production email filter.

## Files

- phishing_test.py: baseline evaluation with CSV output
- phishing_test_baseline.py: preserved baseline script
- phishing_results_baseline.csv: preserved baseline results
- phishing_hybrid.py: AI and rule-assisted comparison
- phishing_hybrid_results.csv: comparison results

## Run

With the project's virtual environment activated:

    python phishing_test.py
    python phishing_hybrid.py

The first model load requires downloading model files.
The scripts also require their installed Python dependencies.

## Next Steps

- Preserve current examples as development tests.
- Prepare a separate, manually labeled evaluation set.
- Evaluate future changes on the same examples for fair comparisons.
- Track false positives and false negatives separately.
- Record dependency versions and model revisions for reproducibility.
