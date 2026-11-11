"""Movie recommendations on MovieLens 1M: four models, one honest evaluation.

Models
  - Popularity: recommend the most-watched movies (the baseline to beat)
  - ItemKNN:   "people who watched X also watched Y" (cosine item similarity)
  - PureSVD:   matrix factorization with a truncated SVD
  - EASE:      a linear item-item model with a closed-form solution (Steck, 2019)

Evaluation
  Leave-last-out by time: each user's most recent movie is hidden, every model
  ranks all movies the user hasn't seen, and we check whether the hidden movie
  lands in the top 10. Metrics: HitRate@10, NDCG@10, catalog coverage, and a
  bootstrap confidence interval for the difference between the two best models.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds

K = 10


# ---------- data ----------
def load_movielens(path="ml-1m"):
    ratings = pd.read_csv(f"{path}/ratings.dat", sep="::", engine="python",
                          names=["user", "item", "rating", "ts"])
    movies = pd.read_csv(f"{path}/movies.dat", sep="::", engine="python",
                         names=["item", "title", "genres"], encoding="latin-1")
    return ratings, movies


def split_last_out(ratings):
    """Hide each user's most recent interaction as the test item."""
    ratings = ratings.sort_values(["user", "ts"])
    last = ratings.groupby("user").tail(1)
    train = ratings.drop(last.index)
    return train, last


def to_matrix(train, test):
    users = np.sort(train["user"].unique())
    items = np.sort(train["item"].unique())
    u_idx = {u: i for i, u in enumerate(users)}
    i_idx = {m: j for j, m in enumerate(items)}
    rows = train["user"].map(u_idx).to_numpy()
    cols = train["item"].map(i_idx).to_numpy()
    X = sp.csr_matrix((np.ones(len(train), dtype=np.float32), (rows, cols)),
                      shape=(len(users), len(items)))
    test = test[test["user"].isin(u_idx) & test["item"].isin(i_idx)]
    test_u = test["user"].map(u_idx).to_numpy()
    test_i = test["item"].map(i_idx).to_numpy()
    return X, test_u, test_i, items


# ---------- models: each returns a users x items score matrix ----------
def popularity(X, users):
    pop = np.asarray(X.sum(axis=0)).ravel()
    return np.tile(pop, (len(users), 1))


def item_knn(X, users):
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=0)).ravel()) + 1e-9
    Xn = X.multiply(1 / norms).tocsc()
    S = (Xn.T @ Xn).toarray()
    np.fill_diagonal(S, 0)
    return X[users] @ S


def pure_svd(X, users, factors=64):
    _, _, vt = svds(X.astype(np.float64), k=factors)
    return (X[users] @ vt.T) @ vt


def ease(X, users, lam=500.0):
    G = (X.T @ X).toarray().astype(np.float64)
    G[np.diag_indices_from(G)] += lam
    P = np.linalg.inv(G)
    B = -P / np.diag(P)
    np.fill_diagonal(B, 0)
    return X[users] @ B


MODELS = {"Popularity": popularity, "ItemKNN": item_knn, "PureSVD": pure_svd, "EASE": ease}


# ---------- evaluation ----------
def evaluate(scores, X, users, test_items, k=K):
    """Per-user hit and NDCG, plus the set of items ever recommended."""
    scores = np.asarray(scores, dtype=np.float64)
    seen = X[users].toarray() > 0
    scores[seen] = -np.inf                      # never recommend what they've watched
    target = scores[np.arange(len(users)), test_items]
    rank = (scores > target[:, None]).sum(axis=1)          # 0 = best
    hit = (rank < k).astype(float)
    ndcg = np.where(rank < k, 1 / np.log2(rank + 2), 0.0)
    top = np.argpartition(-scores, k, axis=1)[:, :k]
    return hit, ndcg, np.unique(top)


def bootstrap_diff(a, b, n=2000, seed=0):
    """95% CI for mean(a - b), resampling users."""
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return d.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def run(X, test_u, test_i, batch=1000):
    n_items = X.shape[1]
    results, per_user = [], {}
    for name, model in MODELS.items():
        hits, ndcgs, covered = [], [], set()
        for start in range(0, len(test_u), batch):
            users = test_u[start:start + batch]
            h, n, top = evaluate(model(X, users), X, users, test_i[start:start + batch])
            hits.append(h); ndcgs.append(n); covered.update(top.tolist())
        h, n = np.concatenate(hits), np.concatenate(ndcgs)
        per_user[name] = n
        results.append({"model": name, f"HitRate@{K}": h.mean(), f"NDCG@{K}": n.mean(),
                        "coverage": len(covered) / n_items})
    return pd.DataFrame(results).set_index("model"), per_user


if __name__ == "__main__":
    ratings, movies = load_movielens()
    train, test = split_last_out(ratings)
    X, test_u, test_i, items = to_matrix(train, test)
    print(f"{X.shape[0]:,} users, {X.shape[1]:,} movies, {X.nnz:,} training interactions")
    table, per_user = run(X, test_u, test_i)
    print(table.round(4))
    best, second = table[f"NDCG@{K}"].sort_values(ascending=False).index[:2]
    m, lo, hi = bootstrap_diff(per_user[best], per_user[second])
    print(f"\n{best} vs {second}: NDCG@{K} difference {m:+.4f} (95% CI {lo:+.4f} to {hi:+.4f})")
