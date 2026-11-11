# Movie recommendations on MovieLens 1M

Which recommender should a streaming product ship? I compared four models on 1M ratings from 6,040 users
and evaluated them the way production teams do: hide each user's most recent movie and check whether the
model ranks it in the top 10.

## Results (leave-last-out, top-10)
| Model | HitRate@10 | NDCG@10 | Catalog coverage |
|---|---|---|---|
| Popularity (baseline) | 3.6% | 0.018 | 5% |
| ItemKNN (item-item cosine) | 7.0% | 0.037 | 13% |
| PureSVD (matrix factorization) | 9.7% | 0.049 | 33% |
| **EASE** (Steck, 2019) | **11.0%** | **0.055** | **35%** |

- **EASE wins, and the gap is real:** +0.0054 NDCG@10 over PureSVD (bootstrap 95% CI +0.0029 to +0.0080).
- **3× the hit rate of popularity** while recommending 7× more of the catalog, so it isn't just pushing blockbusters.
- Example: a user who recently watched *Mulan*, *Hercules* and *A Bug's Life* gets *The Lion King*, *The Little Mermaid* and *Sleeping Beauty*; their actual next movie was *Pocahontas*.

## How it works
- **Split:** each user's last interaction (by timestamp) is the test item; all earlier ones are training data.
- **Ranking:** every model scores all movies the user hasn't seen; already-watched movies are never recommended.
- **Significance:** per-user NDCG differences are bootstrapped (2,000 resamples) to get a confidence interval.
- Pure NumPy/SciPy, no special libraries; the full run takes about 2 minutes on a CPU.

## Run it
Open `movielens_recommender.ipynb` in Google Colab and choose **Runtime → Run all**.
The notebook downloads MovieLens 1M from GroupLens and runs `recsys.py`.

## Data and credit
MovieLens 1M (F. Maxwell Harper and Joseph A. Konstan, 2015). EASE: Harald Steck, *Embarrassingly Shallow Autoencoders for Sparse Data*, WWW 2019.

Built by **Shirisha Gajendra** · MS Data Science, Stony Brook University
