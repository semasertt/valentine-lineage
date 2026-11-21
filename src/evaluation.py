"""From Valentine's column scores to evaluated lineage predictions
(notebooks/04_experiments.ipynb).

1. ``table_scores``: for every ordered candidate pair ``source -> target``,
   take the best score of each *target* column (over all source columns) and
   aggregate these per-column scores into one table-pair score.
2. ``ranking_metrics``: evaluate the scores as a ranking (PR-AUC, AUROC over
   all pairs; Hits@k and MRR over the candidate sources of each derived table).
3. ``threshold_metrics``: precision / recall / F1 at a threshold chosen on
   the train split only and applied unchanged to test and multi.
4. Baselines that do not use Valentine: random ranking, table-name
   similarity and raw value containment.
"""

from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from eda import containment
from preprocessing import base_name

COVERAGE_MIN = 0.5      # a target column is "covered" from this column score on

AGGREGATIONS = {
    "max": "best single target column (catches a source that gave one column to a join)",
    "top3": "mean of the three best target columns",
    "mean": "mean over all target columns (rewards full copies)",
    "coverage": f"share of target columns with score >= {COVERAGE_MIN}",
}


# --- 1. Column scores -> table-pair scores ---------------------------------------

def best_per_target_column(column_scores, data):
    """One row per (split, matcher, source, target, target column) with the
    best score that any source column got for it. Target columns that
    Valentine did not return get 0."""
    best = (column_scores.groupby(["split", "matcher", "source", "target", "target_column"])
            .score.max().reset_index())
    frames = []
    for (split, matcher, source, target), group in best.groupby(["split", "matcher", "source", "target"]):
        full = pd.Series(0.0, index=data[split][target].columns)
        full.update(group.set_index("target_column").score)
        frames.append(pd.DataFrame({"split": split, "matcher": matcher, "source": source, "target": target,
                                    "target_column": full.index, "score": full.values}))
    return pd.concat(frames, ignore_index=True)


def aggregate(scores):
    """The aggregations of AGGREGATIONS for one vector of per-column scores."""
    ordered = np.sort(np.asarray(scores, dtype=float))[::-1]
    return {
        "max": ordered[0],
        "top3": ordered[:3].mean(),
        "mean": ordered.mean(),
        "coverage": (ordered >= COVERAGE_MIN).mean(),
    }


def table_scores(column_scores, pairs, data):
    """Long table: one row per (matcher, aggregation, candidate pair) with
    the pair's score and ground-truth columns. Pairs for which a matcher
    returned no column pair at all get score 0."""
    best = best_per_target_column(column_scores, data)
    rows = [{"split": split, "matcher": matcher, "source": source, "target": target, **aggregate(group.score)}
            for (split, matcher, source, target), group in best.groupby(["split", "matcher", "source", "target"])]
    wide = pd.DataFrame(rows)
    keys = ["split", "source", "target"]
    out = []
    for matcher in column_scores.matcher.unique():
        scored = pairs.merge(wide[wide.matcher == matcher], on=keys, how="left")
        scored["matcher"] = matcher
        out.append(scored.fillna({a: 0.0 for a in AGGREGATIONS}))
    wide = pd.concat(out, ignore_index=True)
    return wide.melt(id_vars=[c for c in wide.columns if c not in AGGREGATIONS],
                     value_vars=list(AGGREGATIONS), var_name="aggregation", value_name="score")


# --- 2. Ranking metrics ---------------------------------------------------------

def hits_and_rr(scored, ks=(1, 3, 10)):
    """Per derived target with at least one true source: rank all candidate
    sources by score; Hits@k = a true source is among the first k, RR = 1 /
    rank of the first true source. Ties are broken pessimistically (a true
    source tied with negatives is ranked after them), so a matcher that gives
    0 to everything does not get credit."""
    rows = []
    for (split, target), group in scored.groupby(["split", "target"]):
        if not group.is_lineage.any():
            continue
        first_true = group.loc[group.is_lineage, "score"].max()
        rank = int((group.score >= first_true).sum() - (group.is_lineage & (group.score == first_true)).sum() + 1)
        rows.append({"split": split, "target": target, "rank": rank, "candidates": len(group),
                     **{f"Hits@{k}": float(rank <= k) for k in ks}, "RR": 1.0 / rank})
    return pd.DataFrame(rows)


def ranking_metrics(scored):
    """PR-AUC and AUROC over all candidate pairs; Hits@k and MRR per target."""
    y, s = scored.is_lineage.astype(int), scored.score
    per_target = hits_and_rr(scored)
    return {
        "PR-AUC": average_precision_score(y, s) if y.any() else np.nan,
        "AUROC": roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan,
        **{c: per_target[c].mean() for c in ["Hits@1", "Hits@3", "Hits@10"]},
        "MRR": per_target["RR"].mean(),
    }


def evaluate_all(scores, splits=("train", "test", "multi"), group=("matcher", "aggregation")):
    """ranking_metrics for every configuration, per split and on test+multi
    (the held-out splits) and all splits together."""
    rows = []
    views = {split: [split] for split in splits}
    views["held-out (test+multi)"] = ["test", "multi"]
    views["all"] = list(splits)
    for keys, part in scores.groupby(list(group)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for view, members in views.items():
            subset = part[part.split.isin(members)]
            rows.append({**dict(zip(group, keys)), "evaluated on": view,
                         "positives": int(subset.is_lineage.sum()), "pairs": len(subset),
                         **ranking_metrics(subset)})
    return pd.DataFrame(rows)


# --- 3. Threshold metrics ---------------------------------------------------------

def best_threshold(scored):
    """Threshold that maximises F1 on the given pairs (used on train only)."""
    y = scored.is_lineage.astype(int)
    best = (0.0, np.inf)
    for t in np.unique(scored.score):
        f1 = f1_score(y, scored.score >= t, zero_division=0)
        if f1 > best[0]:
            best = (f1, t)
    return best[1]


def threshold_metrics(scored, threshold):
    y, pred = scored.is_lineage.astype(int), scored.score >= threshold
    return {"threshold": threshold, "predicted links": int(pred.sum()),
            "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0),
            "F1": f1_score(y, pred, zero_division=0)}


def threshold_table(scores, group=("matcher", "aggregation")):
    """Choose the F1-optimal threshold on train, report it on train (seen)
    and on test+multi (unseen)."""
    rows = []
    for keys, part in scores.groupby(list(group)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        t = best_threshold(part[part.split == "train"])
        for view, members in [("train", ["train"]), ("held-out (test+multi)", ["test", "multi"])]:
            rows.append({**dict(zip(group, keys)), "evaluated on": view,
                         **threshold_metrics(part[part.split.isin(members)], t)})
    return pd.DataFrame(rows)


# --- 4. Paper-style Hits@10 ------------------------------------------------------

def hits_vs_sampled_negatives(scored, negatives=50, k=10, repeats=200, seed=0):
    """Closer to the baseline paper: each positive pair is ranked against a
    random sample of ``negatives`` negative pairs from the same split
    (pessimistic ties); Hits@k averaged over positives and ``repeats`` samples."""
    rng = np.random.default_rng(seed)
    hits = []
    for split, part in scored.groupby("split"):
        neg = part.loc[~part.is_lineage, "score"].to_numpy()
        for s in part.loc[part.is_lineage, "score"]:
            for _ in range(repeats):
                sample = rng.choice(neg, size=min(negatives, len(neg)), replace=False)
                hits.append((sample >= s).sum() + 1 <= k)
    return float(np.mean(hits))


# --- 5. Baselines without Valentine -----------------------------------------------

def random_baseline(pairs, repeats=500, seed=0):
    """Expected metrics of a random ranking (scores ~ U(0,1))."""
    rng = np.random.default_rng(seed)
    runs = []
    for _ in range(repeats):
        scored = pairs.assign(score=rng.random(len(pairs)))
        runs.append(scored)
    return runs


def name_baseline(pairs, tables):
    """Similarity of the table names without type prefix (vwOrderFulfillment ->
    OrderFulfillment vs Orders): what a person would guess from names alone."""
    names = {(r.split, r.table): base_name(r.name).lower() for r in tables.itertuples()}
    score = [SequenceMatcher(None, names[(p.split, p.source)], names[(p.split, p.target)]).ratio()
             for p in pairs.itertuples()]
    return pairs.assign(score=score)


def containment_baseline(pairs, data):
    """Raw value overlap without Valentine (the EDA value score): for each
    target column the best share of its distinct values found in one source
    column, averaged over the target's columns."""
    scores = []
    for p in pairs.itertuples():
        source, target = data[p.split][p.source], data[p.split][p.target]
        best = [max(containment(target[t], source[s]) for s in source.columns) for t in target.columns]
        scores.append(float(np.mean(best)))
    return pairs.assign(score=scores)


# --- 6. Error analysis ------------------------------------------------------------

def direction_check(scored):
    """For each true link A -> B: is score(A -> B) higher than score(B -> A)?"""
    lookup = scored.set_index(["split", "source", "target"]).score
    rows = []
    for p in scored[scored.is_lineage].itertuples():
        forward, backward = p.score, lookup.get((p.split, p.target, p.source), np.nan)
        rows.append({"split": p.split, "source": p.source_name, "target": p.target_name,
                     "forward": forward, "backward": backward,
                     "result": "correct" if forward > backward else "tie" if forward == backward else "wrong"})
    return pd.DataFrame(rows)


def top_false_positives(scored, threshold):
    """Pairs above the threshold that are not lineage links, with their relation."""
    fp = scored[(scored.score >= threshold) & ~scored.is_lineage]
    return fp.sort_values("score", ascending=False)
