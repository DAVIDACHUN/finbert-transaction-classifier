### Headline metrics (test splits)

| Model | Split | Accuracy | Macro-F1 | ECE raw | ECE temp-scaled |
|---|---|---|---|---|---|
| TF-IDF + LogReg | test_seen | 0.953 | 0.955 | 0.025 | 0.013 |
| TF-IDF + LogReg | test_unseen | 0.779 | 0.791 | 0.150 | 0.069 |
| FinBERT (fine-tuned) | test_seen | 0.953 | 0.958 | 0.019 | 0.015 |
| FinBERT (fine-tuned) | test_unseen | 0.794 | 0.780 | 0.054 | 0.082 |

### Auto-routing (threshold picked on validation, applied to test)

| Model | Split | Target acc (val) | Threshold | Coverage (auto) | Auto-routed acc | Human-review rate |
|---|---|---|---|---|---|---|
| TF-IDF + LogReg | test_seen | 98.0% | 0.380 | 96.9% | 97.80% | 3.1% |
| TF-IDF + LogReg | test_seen | 99.0% | 0.472 | 96.2% | 98.34% | 3.8% |
| TF-IDF + LogReg | test_seen | 99.5% | 0.595 | 94.9% | 99.09% | 5.1% |
| TF-IDF + LogReg | test_unseen | 98.0% | 0.380 | 88.5% | 83.65% | 11.5% |
| TF-IDF + LogReg | test_unseen | 99.0% | 0.472 | 82.7% | 85.81% | 17.3% |
| TF-IDF + LogReg | test_unseen | 99.5% | 0.595 | 72.2% | 89.01% | 27.8% |
| FinBERT (fine-tuned) | test_seen | 98.0% | 0.405 | 97.7% | 97.06% | 2.3% |
| FinBERT (fine-tuned) | test_seen | 99.0% | 0.627 | 95.0% | 99.02% | 5.0% |
| FinBERT (fine-tuned) | test_seen | 99.5% | 0.771 | 93.6% | 99.64% | 6.4% |
| FinBERT (fine-tuned) | test_unseen | 98.0% | 0.405 | 94.7% | 82.46% | 5.3% |
| FinBERT (fine-tuned) | test_unseen | 99.0% | 0.627 | 84.3% | 87.67% | 15.7% |
| FinBERT (fine-tuned) | test_unseen | 99.5% | 0.771 | 78.9% | 90.36% | 21.1% |

### FinBERT seed variance (3 fine-tuning runs)

| Split | Mean accuracy | Std dev |
|---|---|---|
| val | 0.966 | 0.003 |
| test_seen | 0.955 | 0.002 |
| test_unseen | 0.806 | 0.011 |
