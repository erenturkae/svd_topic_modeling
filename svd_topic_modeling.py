# Abdullah Eren Erenturk
# 150210327

import os
import re
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt


# TASK 1 – TF-IDF Matrix Construction

# English stop words
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can't",
    "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from",
    "further", "get", "got", "had", "hadn't", "has", "hasn't", "have",
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't",
    "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't",
    "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only",
    "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should",
    "shouldn't", "so", "some", "such", "than", "that", "that's", "the",
    "their", "theirs", "them", "themselves", "then", "there", "there's",
    "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't",
    "what", "what's", "when", "when's", "where", "where's", "which", "while",
    "who", "who's", "whom", "why", "why's", "will", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves",
    # domain specific extras
    "said", "mr", "mrs", "ms", "one", "two", "also", "us", "uk", "new",
    "year", "years", "last", "first", "time", "people", "told", "says",
    "say", "make", "made", "way", "use", "used", "now", "three", "may",
    "like", "just", "back", "well", "much", "many", "set", "go", "come",
    "going", "however", "although", "within", "without", "around", "among",
    "another", "still", "since", "became", "become", "even", "added",
}


def tokenize(text):
    """
    Lowercase, strip non-alpha, split on whitespace, remove stop words,
    and keep tokens with length >= 3.
    """
    text = text.lower()
    # keep only letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if len(t) >= 3 and t not in STOP_WORDS]

    return tokens


def build_tfidf_matrix(documents):
    """
    Build a TF-IDF term-document matrix (m x n), L2-normalized per column.

    Returns
    A: np.ndarray (m x n)   — TF-IDF matrix
    vocab: list[str]        — vocabulary (row labels)
    """

    n = len(documents)

    # build vocabulary
    vocab_set = set()
    for doc in documents:
        vocab_set.update(doc)
    vocab = sorted(vocab_set)
    term2idx = {t: i for i, t in enumerate(vocab)}
    m = len(vocab)

    # TF: raw term counts per document
    TF = np.zeros((m, n), dtype=np.float64)
    for j, doc in enumerate(documents):
        for token in doc:
            if token in term2idx:
                TF[term2idx[token], j] += 1
    # divide each column by total tokens in that doc (normalized TF)
    col_sums = TF.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    TF = TF / col_sums

    # IDF: log((1 + n) / (1 + df)) + 1  (smooth IDF)
    df = (TF > 0).sum(axis=1)
    idf = np.log((1.0 + n) / (1.0 + df)) + 1.0

    # TF-IDF
    A = TF * idf[:, np.newaxis] # broadcast over columns

    # L2-normalize each document column
    col_norms = np.linalg.norm(A, axis=0, keepdims=True)
    col_norms[col_norms == 0] = 1
    A = A / col_norms

    return A, vocab


# TASK 2 – Eigenvalue Decomposition from Scratch

def power_iteration(B, max_iter=2000, tol=1e-9, seed=42):
    """
    Power Method to find the dominant eigenpair of symmetric PSD matrix B.

    Returns
    eigenvalue: float
    eigenvector (v): np.ndarray
    """

    rng = np.random.default_rng(seed)
    n = B.shape[0]
    v = rng.standard_normal(n)
    v = v / np.linalg.norm(v)

    lambda_old = 0.0
    for _ in range(max_iter):
        w = B @ v                               # dominant direction
        lambda_new = float(v @ w)               # Rayleigh quotient
        norm_w = np.linalg.norm(w)
        if norm_w < 1e-15:
            break
        v = w / norm_w
        if abs(lambda_new - lambda_old) < tol:
            break
        lambda_old = lambda_new

    # final Rayleigh quotient for accuracy
    eigenvalue = float(v @ (B @ v))

    return eigenvalue, v


def eigen_decomposition_top_k(B, k):
    """
    Compute the top-k eigenpairs of B using Power Method + Deflation.

    Returns
    eigenvalues: np.ndarray (k,)
    eigenvectors: np.ndarray (n, k)  — columns are eigenvectors
    """

    n = B.shape[0]
    eigenvalues = np.zeros(k)
    eigenvectors = np.zeros((n, k))

    B_deflated = B.copy()

    for i in range(k):
        seed = i * 17 + 3   # different seed per step
        lam, v = power_iteration(B_deflated, seed=seed)

        # clamp small negatives caused by floating-point to zero
        if lam < 0:
            lam = 0.0

        eigenvalues[i] = lam
        eigenvectors[:, i] = v

        # deflation: remove this component
        B_deflated = B_deflated - lam * np.outer(v, v)

    return eigenvalues, eigenvectors


# TASK 3 – SVD Construction & Singular Value Analysis

def svd_from_scratch(A, k):
    """
    Compute truncated SVD A ≈ U Σ Vᵀ using eigendecomposition of AᵀA.

    Returns
    U: np.ndarray (m, k)
    sigma: np.ndarray (k,)
    VT: np.ndarray (k, n)
    """

    B = A.T @ A     # (n x n) symmetric PSD
    eigenvalues, V = eigen_decomposition_top_k(B, k)    # V: (n, k)
    # singular values = sqrt of eigenvalues  (clamp negatives)
    sigma = np.sqrt(np.maximum(eigenvalues, 0.0))

    # U = A V Σ⁻¹  — compute column by column to avoid division by zero
    m = A.shape[0]
    U = np.zeros((m, k))
    for i in range(k):
        if sigma[i] > 1e-12:
            U[:, i] = (A @ V[:, i]) / sigma[i]
        else:
            # zero singular value (assign zero column)
            U[:, i] = 0.0
    VT = V.T    # (k, n)

    return U, sigma, VT


# TASK 4 – Truncated SVD & Reconstruction Error

def reconstruction_error(A, U, sigma, VT, k):
    """
    Compute Frobenius reconstruction error ||A - A_k||_F.
    Uses only the top-k components of the provided decomposition.
    """

    A_k = (U[:, :k] * sigma[:k]) @ VT[:k, :]       # (m x n)
    diff = A - A_k
    error = float(np.sqrt(np.sum(diff ** 2)))

    return error


# Helper – Dataset Loading
CATEGORIES = ["business", "entertainment", "politics", "sport", "tech"]


def load_bbc_dataset(data_path, max_docs=100):
    """
    Load up to max_docs documents per category from the BBC folder structure.

    Returns
    documents : list[list[str]]   — tokenised docs
    labels    : list[str]          — category name per doc
    """

    documents = []
    labels = []

    for cat in CATEGORIES:
        cat_dir = os.path.join(data_path, cat)
        if not os.path.isdir(cat_dir):
            print(f"[WARNING] Category folder not found: {cat_dir}")
            continue

        files = sorted(f for f in os.listdir(cat_dir) if f.endswith(".txt"))
        files = files[:max_docs]

        for fname in files:
            fpath = os.path.join(cat_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
            except Exception as e:
                print(f"[WARNING] Could not read {fpath}: {e}")
                continue

            tokens = tokenize(text)
            if len(tokens) < 5:
                continue

            documents.append(tokens)
            labels.append(cat)

    print(f"Loaded {len(documents)} documents across {len(set(labels))} categories.")
    return documents, labels


# Plotting helpers
CAT_COLORS = {
    "business":      "#2196F3",
    "entertainment": "#FF5722",
    "politics":      "#4CAF50",
    "sport":         "#9C27B0",
    "tech":          "#FF9800",
}


def plot_singular_value_decay(sigma, output_path):
    total_energy = np.sum(sigma ** 2)
    cumvar = np.cumsum(sigma ** 2) / total_energy * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(range(1, len(sigma) + 1), sigma, "o-", color="#1565C0", markersize=4)
    ax1.set_xlabel("Component index")
    ax1.set_ylabel("Singular value σ_i")
    ax1.set_title("Singular Value Decay")
    ax1.grid(True, alpha=0.3)

    ax2.plot(range(1, len(sigma) + 1), cumvar, "s-", color="#E65100", markersize=4)
    ax2.axhline(90, color="gray", linestyle="--", label="90% threshold")
    ax2.set_xlabel("Number of components k")
    ax2.set_ylabel("Cumulative explained variance (%)")
    ax2.set_title("Cumulative Explained Variance")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fpath = os.path.join(output_path, "singular_value_decay.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"Saved: {fpath}")


def plot_reconstruction_errors(ranks, errors, output_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ranks, errors, "D-", color="#6A1B9A", markersize=7)
    for r, e in zip(ranks, errors):
        ax.annotate(f"{e:.4f}", (r, e), textcoords="offset points",
                    xytext=(4, 4), fontsize=8)
    ax.set_xlabel("Rank k")
    ax.set_ylabel("||A - A_k||_F")
    ax.set_title("Frobenius Reconstruction Error vs. Rank")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fpath = os.path.join(output_path, "reconstruction_errors.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"Saved: {fpath}")


def plot_document_embeddings(VT, sigma, labels, output_path):
    """
    Project documents to 2D concept space: D = Σ_k V_kᵀ  → use first 2 dims.
    """
    D = (sigma[:, np.newaxis] * VT)   # (k, n)
    x = D[0, :]
    y = D[1, :]

    fig, ax = plt.subplots(figsize=(9, 6))
    for cat in CATEGORIES:
        idx = [i for i, l in enumerate(labels) if l == cat]
        ax.scatter(x[idx], y[idx], label=cat, alpha=0.65, s=30,
                   color=CAT_COLORS.get(cat, "gray"))
    ax.set_xlabel("Concept 1")
    ax.set_ylabel("Concept 2")
    ax.set_title("Document Embeddings in 2D SVD Concept Space")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fpath = os.path.join(output_path, "document_embeddings_2D.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"Saved: {fpath}")


# Main pipeline

def main():
    parser = argparse.ArgumentParser(description="SVD Topic Modeling")
    parser.add_argument("--data_path", required=True, help="Path to BBC dataset root folder")
    parser.add_argument("--output_path", required=True, help="Directory for output files")
    parser.add_argument("--max_docs", type=int, default=100, help="Max documents per category")
    parser.add_argument("--k_svd", type=int, default=50, help="Number of singular values/vectors")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    topic_terms_dir = os.path.join(args.output_path, "topic_terms")
    os.makedirs(topic_terms_dir, exist_ok=True)

    # 1. Load & tokenise
    print("\n[TASK 1] Building TF-IDF matrix …")
    documents, labels = load_bbc_dataset(args.data_path, args.max_docs)
    if len(documents) == 0:
        raise RuntimeError("No documents loaded. Check --data_path.")

    A, vocab = build_tfidf_matrix(documents)
    print(f"TF-IDF matrix shape: {A.shape}  (terms × documents)")

    np.save(os.path.join(args.output_path, "tfidf_matrix.npy"), A)
    print("Saved: tfidf_matrix.npy")

    # 2 & 3. Truncated SVD
    k = args.k_svd
    print(f"\n[TASK 2 & 3] Computing truncated SVD  k = {k} …")
    U, sigma, VT = svd_from_scratch(A, k)
    print(f"U: {U.shape}  sigma: {sigma.shape}  VT: {VT.shape}")

    # Validation against numpy (not used in analysis)
    print("\n[Validation] Comparing against numpy.linalg.svd …")
    _, sigma_np, _ = np.linalg.svd(A, full_matrices=False)
    max_diff = np.max(np.abs(sigma - sigma_np[:k]))
    print(f"  Max |σ_scratch - σ_numpy| over top-{k}: {max_diff:.2e}")
    if max_diff < 1e-3:
        print("  ✓ Within acceptable tolerance (< 1e-3)")
    else:
        print("  ⚠ Difference exceeds tolerance — check deflation steps")

    plot_singular_value_decay(sigma, args.output_path)

    # Energy analysis
    total_energy = np.sum(sigma_np ** 2)   # full spectrum
    cumvar_full = np.cumsum(sigma ** 2) / total_energy * 100
    k90 = int(np.searchsorted(cumvar_full, 90.0)) + 1
    print(f"\nMinimum k for 90% energy retention: {k90}")

    # 4. Reconstruction errors
    print("\n[TASK 4] Computing reconstruction errors …")
    rank_list = [r for r in [2, 5, 10, 20, 50] if r <= k]
    errors = []
    lines = ["rank\tFrobenius_error"]
    for r in rank_list:
        err = reconstruction_error(A, U, sigma, VT, r)
        errors.append(err)
        lines.append(f"{r}\t{err:.6f}")
        print(f"  k={r:3d}  ||A - A_k||_F = {err:.6f}")

    with open(os.path.join(args.output_path, "reconstruction_errors.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("Saved: reconstruction_errors.txt")

    plot_reconstruction_errors(rank_list, errors, args.output_path)

    # 5. Topic analysis & visualisation
    print("\n[TASK 5] Topic analysis …")

    # Document embeddings in concept space
    plot_document_embeddings(VT, sigma, labels, args.output_path)

    # Top-10 terms per singular vector (topic)
    num_topics = min(5, k)
    for t in range(num_topics):
        weights = np.abs(U[:, t])
        top_idx = np.argsort(weights)[::-1][:10]
        top_terms = [(vocab[i], float(weights[i])) for i in top_idx]

        fname = os.path.join(topic_terms_dir, f"topic_{t+1}_terms.txt")
        with open(fname, "w") as f:
            f.write(f"Topic {t+1}  (σ_{t+1} = {sigma[t]:.4f})\n")
            f.write("-" * 35 + "\n")
            for term, w in top_terms:
                f.write(f"  {term:<20s}  {w:.6f}\n")
        print(f"  Topic {t+1}: {[x[0] for x in top_terms[:5]]} …")

    print("\n[DONE] All outputs written to:", args.output_path)


if __name__ == "__main__":
    main()
