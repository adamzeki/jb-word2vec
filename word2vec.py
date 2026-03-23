import re
import collections
import numpy as np

RAW_TEXT = """
anarchism originated as a term of abuse first used against early working class
radicals including the diggers of the english revolution and the sans culottes of
the french revolution whilst the term is still used in a pejorative way to describe
any act that used violent means to destroy the organization of society it has also
been taken up as a positive label by self-defined anarchists the word anarchism is
derived from the greek without archons ruler chief king anarchism as a political
philosophy is the belief that rulers are unnecessary and should be abolished although
there are differing interpretations of what this means anarchism also refers to
related social movements that advocate for the elimination of authoritarian
institutions particularly the state the word anarchy is often used by non-anarchists
as a pejorative term implying chaos but anarchists usually use the term to mean a
society based on voluntary cooperation without domination or hierarchy several topics
in this article may require some understanding of the history of anarchism
"""

def tokenise(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    return text.split()


def build_vocab(tokens: list[str], min_count: int = 1) -> tuple[dict, dict, np.ndarray]:
    freq = collections.Counter(tokens)

    vocab = sorted(
        [w for w, c in freq.items() if c >= min_count],
        key=lambda w: -freq[w]
    )

    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}

    counts = np.array([freq[w] for w in vocab], dtype=np.float64)

    return word2idx, idx2word, counts


def subsample_tokens(token_ids: list[int], counts: np.ndarray, rng: np.random.Generator, threshold: float = 1e-3) -> list[int]:
    N = len(token_ids)
    f = counts / N

    p_keep = np.minimum(1.0, np.sqrt(threshold / f)) # Based on the formula within the paper on negative sampling and sub-sampling

    ids = np.array(token_ids, dtype=np.int32)

    u = rng.random(size=N)

    mask = u < p_keep[ids]

    filtered = ids[mask].tolist()

    n_removed = N - len(filtered)
    pct = 100.0 * n_removed / N
    print(f"Sub-sampling: {N} -> {len(filtered)} tokens "
          f"(removed {n_removed}, {pct:.1f}%)")

    return filtered


def make_noise_distribution(counts: np.ndarray, power: float = 0.75) -> np.ndarray: # Based on the finding in the paper, that unigram distribution raised to the power odf 3/4 outperforms unigram and uniform distribution 
    powered = counts ** power
    noise_dist = powered / powered.sum()
    return noise_dist


def generate_skipgram_pairs(token_ids: list[int], window_size: int = 2) -> list[tuple[int, int]]:
    pairs = []
    n = len(token_ids)

    for t in range(n):
        lo = max(0, t - window_size)
        hi = min(n - 1, t + window_size)

        for j in range(lo, hi + 1):
            if j == t:
                continue
            pairs.append((token_ids[t], token_ids[j]))

    return pairs


def init_weights(vocab_size: int, embed_dim: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    scale = 0.5 / embed_dim
    W = rng.uniform(-scale, scale, (vocab_size, embed_dim))
    C = rng.uniform(-scale, scale, (vocab_size, embed_dim))
    return W, C


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-x))


def sgns_step(centre_id: int, context_id: int, neg_ids: np.ndarray, W: np.ndarray, C: np.ndarray, lr: float) -> float:
    w_vec = W[centre_id]
    c_pos = C[context_id]
    c_neg = C[neg_ids]

    score_pos = np.dot(w_vec, c_pos)

    scores_neg = c_neg @ w_vec

    sig_pos = sigmoid(score_pos)
    sig_neg = sigmoid(scores_neg)

    eps = 1e-10
    loss_pos = -np.log(sig_pos + eps)
    loss_neg = -np.log(1.0 - sig_neg + eps)
    loss = loss_pos + loss_neg.sum()

    err_pos = sig_pos - 1.0

    err_neg = sig_neg

    grad_w = err_pos * c_pos + err_neg @ c_neg

    grad_c_pos = err_pos * w_vec

    grad_c_neg = np.outer(err_neg, w_vec) 
    
    W[centre_id] -= lr * grad_w
    C[context_id] -= lr * grad_c_pos
    C[neg_ids] -= lr * grad_c_neg
    return float(loss)


def train(tokens: list[str],
          embed_dim: int = 50,
          window_size: int = 2,
          num_negatives: int = 5,
          learning_rate: float = 0.025,
          epochs: int = 5,
          min_count: int = 1,
          subsample_t: float = 1e-3,
          seed: int = 42
          ) -> tuple[np.ndarray, dict, dict]:

    word2idx, idx2word, counts = build_vocab(tokens, min_count)
    vocab_size = len(word2idx)
    print(f"Vocabulary size: {vocab_size}")

    token_ids = [word2idx[t] for t in tokens if t in word2idx]
    print(f"Corpus length: {len(token_ids)} tokens")

    noise_dist = make_noise_distribution(counts, power=0.75)

    rng = np.random.default_rng(seed)

    if subsample_t > 0.0:
        token_ids = subsample_tokens(token_ids, counts, rng, threshold=subsample_t)
    else:
        print("Sub-sampling disabled")

    pairs = generate_skipgram_pairs(token_ids, window_size)
    print(f"Training pairs count: {len(pairs)}")

    W, C = init_weights(vocab_size, embed_dim, rng)

    for epoch in range(1, epochs + 1):

        order = rng.permutation(len(pairs))

        total_loss = 0.0

        for step, pair_idx in enumerate(order):     # Using enumerate with step in case we want to print loss by step
            centre_id, context_id = pairs[pair_idx]

            neg_ids = rng.choice(
                vocab_size,         # Uses np.arange(vocab_size) internally
                size=num_negatives,
                replace=False,      # We want to select unique samples
                p=noise_dist
            )

            loss = sgns_step(centre_id, context_id, neg_ids, W, C, learning_rate)
            total_loss += loss

        avg_loss = total_loss / len(pairs)
        print(f"Epoch {epoch}/{epochs}  —  avg loss: {avg_loss:.4f}")

    return W, word2idx, idx2word


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def most_similar(word: str, W: np.ndarray, word2idx: dict, idx2word: dict, top_k: int = 5) -> list[tuple[str, float]]:
    if word not in word2idx:
        print(f"'{word}' not in vocabulary.")
        return []

    idx = word2idx[word]
    query_vec = W[idx]

    dots = W @ query_vec

    norms = np.linalg.norm(W, axis=1)

    sims = dots / (norms * np.linalg.norm(query_vec) + 1e-10)

    sims[idx] = -np.inf

    top_indices = np.argsort(sims)[-top_k:][::-1]

    return [(idx2word[i], float(sims[i])) for i in top_indices]


if __name__ == "__main__":
    print("Word2Vec with Skip-Gram, Negative Sampling and Sub-Sampling, implemented in NumPy")

    tokens = tokenise(RAW_TEXT)

    W, word2idx, idx2word = train(
        tokens,
        embed_dim = 50,
        window_size = 2,
        num_negatives = 5,
        learning_rate = 0.025,
        epochs = 30,
        min_count = 1,
        subsample_t = 1e-2,
        seed = 42,
    )

    print("\n-------------- Nearest neighbours -------------- ")
    for seed_word in ["anarchism", "political", "state", "society",]:
        neighbours = most_similar(seed_word, W, word2idx, idx2word, top_k=5)
        if neighbours:
            pairs_str = ", ".join(f"{w} ({s:.3f})" for w, s in neighbours)
            print(f"  {seed_word:15s} → {pairs_str}")

    print("\n-------------- Pair cosine similarities --------------")
    probe_pairs = [
        ("anarchism", "state"),
        ("political", "society"),
    ]
    for w1, w2 in probe_pairs:
        if w1 in word2idx and w2 in word2idx:
            sim = cosine_similarity(W[word2idx[w1]], W[word2idx[w2]])
            print(f"sim({w1}, {w2}) = {sim:.4f}")

    print("\nDone.")
