"""Second iteration (notebooks/05_second_iteration.ipynb): combining signals
and deciding the direction of a link.

The first iteration (``evaluation.py``) scores every ordered pair
``source -> target`` with one matcher and one aggregation. Its error analysis
showed three weaknesses, each addressed here:

1. ``mean`` misses join sources that gave a single column, ``max`` is fooled by
   shared keys      -> ``hybrid_aggregation`` mixes the two;
2. names and values carry different evidence -> ``combine_matchers`` averages
   matchers of both families, ``stacking`` learns weights on ``train``;
3. the score does not say which table came first -> ``direction_signals`` and
   ``apply_direction`` add a second signal (object type, row counts).

Every weight or choice is fitted on ``train`` only and reported on the
held-out splits (``test`` + ``multi``), as in the first iteration.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from eda import usable

KEYS = ["split", "source", "target"]
HELD_OUT = ["test", "multi"]

# Object types in creation order. In the database they are visible in the
# catalog (base table, view, table, temporary table); the benchmark's naming
# convention (vw_, Table_, ##Temp, Final_) marks the scenario step.
TYPE_ORDER = {"original": 0, "view": 1, "derived table": 2, "temporary": 3, "final": 4}
# Weaker prior that needs only the catalog: base tables of the original schema
# come before every object created later, derived objects are not ordered.
CATALOG_ORDER = {"original": 0, "view": 1, "derived table": 1, "temporary": 1, "final": 1}


# --- Helpers ------------------------------------------------------------------

def config(scored, matcher, aggregation):
    """The pairs scored by one configuration, one row per candidate pair."""
    part = scored[(scored.matcher == matcher) & (scored.aggregation == aggregation)]
    return part.drop(columns=["matcher", "aggregation"]).reset_index(drop=True)


def pr_auc(part):
    return average_precision_score(part.is_lineage.astype(int), part.score)


def wide_scores(scored):
    """One row per candidate pair, one column per ``matcher | aggregation``."""
    wide = scored.pivot_table(index=KEYS, columns=["matcher", "aggregation"], values="score")
    wide.columns = [f"{m} | {a}" for m, a in wide.columns]
    info = scored.drop_duplicates(KEYS).drop(columns=["matcher", "aggregation", "score"])
    return info.merge(wide.reset_index(), on=KEYS)


# --- 1. Hybrid aggregation ----------------------------------------------------

def hybrid_aggregation(scored, matcher, weight):
    """score = weight * mean + (1 - weight) * max for one matcher.
    weight = 1 is the first-iteration ``mean``, weight = 0 is ``max``."""
    mean, top = config(scored, matcher, "mean"), config(scored, matcher, "max")
    out = mean.copy()
    out["score"] = weight * mean.score.to_numpy() + (1 - weight) * top.set_index(KEYS).loc[
        list(mean[KEYS].itertuples(index=False, name=None)), "score"].to_numpy()
    return out


def fit_weight(scored, matcher, grid=np.linspace(0, 1, 21)):
    """The hybrid weight with the highest PR-AUC on train, and the full curve."""
    curve = []
    for w in grid:
        h = hybrid_aggregation(scored, matcher, w)
        curve.append({"weight": round(float(w), 2),
                      "train": pr_auc(h[h.split == "train"]),
                      "held-out": pr_auc(h[h.split.isin(HELD_OUT)])})
    curve = pd.DataFrame(curve)
    return float(curve.loc[curve.train.idxmax(), "weight"]), curve


# --- 2. Combining matchers ----------------------------------------------------

def combine_matchers(wide, columns, weights=None):
    """Weighted mean of several ``matcher | aggregation`` scores (all in [0, 1])."""
    weights = np.ones(len(columns)) if weights is None else np.asarray(weights, dtype=float)
    out = wide.drop(columns=[c for c in wide.columns if " | " in c]).copy()
    out["score"] = wide[columns].to_numpy() @ weights / weights.sum()
    return out


def stacking(wide, columns, C=0.1, seed=0):
    """Logistic regression on the ``columns`` scores, fitted on train only
    (11 positives, so strongly regularised and class-balanced). Returns the
    scored pairs and the learned coefficients."""
    train = wide[wide.split == "train"]
    model = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed)
    model.fit(train[columns], train.is_lineage.astype(int))
    out = wide.drop(columns=[c for c in wide.columns if " | " in c]).copy()
    out["score"] = model.predict_proba(wide[columns])[:, 1]
    coefs = pd.Series(model.coef_[0], index=columns).sort_values(ascending=False)
    return out, coefs


def choose_C(wide, columns, grid=(0.01, 0.1, 1.0, 10.0), seed=0):
    """Regularisation strength of ``stacking`` by cross-validation on train
    only: the derived tables of train are split into folds (all candidate
    sources of one target stay in the same fold) and the mean PR-AUC over the
    held-in folds is compared."""
    from sklearn.model_selection import GroupKFold
    train = wide[wide.split == "train"].reset_index(drop=True)
    folds = GroupKFold(n_splits=4)
    rows = []
    for C in grid:
        scores = []
        for fit, check in folds.split(train, groups=train.target):
            part = train.iloc[check]
            if not part.is_lineage.any():
                continue
            model = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed)
            model.fit(train.iloc[fit][columns], train.iloc[fit].is_lineage.astype(int))
            scores.append(average_precision_score(part.is_lineage.astype(int),
                                                  model.predict_proba(part[columns])[:, 1]))
        rows.append({"C": C, "CV PR-AUC (train)": float(np.mean(scores))})
    rows = pd.DataFrame(rows)
    return float(rows.loc[rows["CV PR-AUC (train)"].idxmax(), "C"]), rows


# --- 3. Direction ---------------------------------------------------------------

def object_types(splits):
    """{(split, table): object type} from ``Split.kind`` (unlogged objects are
    typed by their name prefix, like logged ones)."""
    types = {}
    for split in splits.values():
        for name in usable(split):
            kind = split.kind(name)
            if kind == "unlogged":
                kind = next((k for p, k in (("vw", "view"), ("temp", "temporary"), ("final", "final"),
                                            ("table", "derived table")) if name.startswith(p)), "derived table")
            types[(split.name, name)] = kind
    return types


def direction_signals(pairs, scored_config, splits, type_order=TYPE_ORDER):
    """For every candidate pair, three independent direction votes in
    {+1 = source -> target is the plausible direction, -1 = the reverse, 0 = no
    opinion}:

    * ``score``: score(A -> B) vs score(B -> A) of the configuration (how much
      of B is found in A, compared with the reverse);
    * ``type``: object type order (a base table is not derived from a view, a
      view not from a temporary table, ...); ``type_order=CATALOG_ORDER`` only
      uses "original base table before anything created later";
    * ``rows``: a derived table has at most as many rows as its largest source
      (selection, join on a key, aggregation), so the larger table comes first.
    """
    types = object_types(splits)
    rows = {(s.name, n): len(df) for s in splits.values() for n, df in usable(s).items()}
    lookup = scored_config.set_index(KEYS).score
    out = pairs[KEYS + ["source_name", "target_name", "label", "is_lineage", "link"]].copy()
    forward = lookup.loc[list(out[KEYS].itertuples(index=False, name=None))].to_numpy()
    backward = lookup.reindex(list(out[["split", "target", "source"]].itertuples(index=False, name=None))).to_numpy()
    out["vote_score"] = np.sign(np.round(forward - np.nan_to_num(backward), 6))
    out["source_type"] = [types[(p.split, p.source)] for p in out.itertuples()]
    out["target_type"] = [types[(p.split, p.target)] for p in out.itertuples()]
    out["vote_type"] = np.sign([type_order[t] - type_order[s] for s, t in zip(out.source_type, out.target_type)])
    out["vote_rows"] = np.sign([rows[(p.split, p.source)] - rows[(p.split, p.target)] for p in out.itertuples()])
    return out


def decide(votes, order=("vote_type", "vote_score", "vote_rows")):
    """First non-zero vote in ``order`` (0 if all signals abstain)."""
    decision = np.zeros(len(votes))
    for column in reversed(order):
        v = votes[column].to_numpy()
        decision = np.where(v != 0, v, decision)
    return decision


def direction_accuracy(votes, order):
    """On the true links: share decided correctly / wrongly / undecided."""
    links = votes[votes.is_lineage]
    d = decide(links, order)
    return {"correct": int((d > 0).sum()), "undecided": int((d == 0).sum()), "wrong": int((d < 0).sum())}


def apply_direction(scored_config, votes, order, penalty=0.0):
    """Multiply the score of pairs whose decided direction is the reverse by
    ``penalty`` (0 removes them from the top of the ranking)."""
    d = pd.Series(decide(votes, order), index=pd.MultiIndex.from_frame(votes[KEYS]))
    out = scored_config.copy()
    factor = np.where(d.loc[list(out[KEYS].itertuples(index=False, name=None))].to_numpy() < 0, penalty, 1.0)
    out["score"] = out.score * factor
    return out
