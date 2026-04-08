# Word2Vec NumPy Implementation

This script is a standalone implementation of Word2Vec built using NumPy. The logic and optimizations are based on the original research by Mikolov et al.

## Included Improvements
The implementation goes beyond basic skip-grams by incorporating several features from the Mikolov papers:

* **Negative Sampling (SGNS):** Replaces the standard softmax with a more efficient binary objective during training
* **Sub-sampling:** Programmatically reduces the frequency of common words (like "the" or "is") to prioritize more informative tokens
* **Phrase Detection:** Features a multi-pass system to detect and merge common phrases like "united_states" or "new_york" into single tokens
* **Optimized Noise Distribution:** Uses the unigram distribution raised to the $3/4$ power for selecting negative samples

## Requirements
* Python 3.10+
* NumPy
* `datasets` library (to fetch the training corpus)

## Getting Started
To run the default training pipeline on a subset of WikiText-103:

```bash
python word2vec.py
```

The script will automatically tokenize the text, detect phrases, sub-sample frequent words, and begin the training process. Once finished, it prints the nearest neighbors for a few test words to verify the embedding quality.