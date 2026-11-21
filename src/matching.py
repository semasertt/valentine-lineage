"""Run Valentine on the candidate table pairs (notebooks/04_experiments.ipynb).

Every ordered candidate pair ``source -> target`` of ``data/processed/pairs.csv``
is matched with every matcher in ``MATCHERS``. The source table is passed to
Valentine first, so asymmetric matchers see the pair in the direction that is
being scored. The output is one row per column pair returned by Valentine::

    split, matcher, source, target, source_column, target_column, score

Valentine's column scores are turned into table-pair scores in ``evaluation.py``.
Matching all 842 pairs with all matchers takes about 20-30 minutes, so the result
is cached in ``results/column_scores_<names>.csv.gz`` and re-used unless
``overwrite=True``.
"""

import time
import warnings
from pathlib import Path

import pandas as pd
from valentine import valentine_match
from valentine.algorithms import Coma, Cupid, DistributionBased, JaccardDistanceMatcher, SimilarityFlooding

from preprocessing import base_name, load_processed

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# name -> (signal the matcher uses, factory). Factories, because some matchers
# keep state between calls.
MATCHERS = {
    "COMA (schema)": ("schema", lambda: Coma()),
    "COMA (schema + instances)": ("schema + instances", lambda: Coma(use_instances=True)),
    "Cupid": ("schema", lambda: Cupid()),
    "Similarity Flooding": ("schema", lambda: SimilarityFlooding()),
    "Jaccard distance": ("instances", lambda: JaccardDistanceMatcher()),
    "Distribution-based": ("instances", lambda: DistributionBased()),
}

FAMILY = {name: family for name, (family, _) in MATCHERS.items()}


# --- Compatibility fix for DistributionBased ----------------------------------------

def patch_distribution_based():
    """DistributionBased solves a small integer program with CBC (via PuLP).

    Valentine 1.1.1 passes the options ``["RandomS", "42"]``, which PuLP turns
    into ``-RandomS -42`` (an invalid, negative seed), and the CBC build shipped
    with PuLP 4.x (cbcbox) crashes in its pre-processing step on some of these
    problems (``PulpSolverError: Error while trying to execute .../cbc``). We pass
    a valid seed and switch CBC pre-processing off. Pre-processing only speeds
    up the search, so the optimal clustering, and thus the matches, stay the same.
    If CBC still fails (9 of the 842 pairs, all original -> original), the same
    problem is solved with HiGHS (``pip install highspy``), another exact solver.
    """
    from pulp import COIN_CMD, HiGHS, PulpSolverError

    import valentine.algorithms.distribution_based.discovery as discovery

    class CbcWithFallback(COIN_CMD):
        fallbacks = 0

        def actualSolve(self, lp, **kwargs):
            try:
                return super().actualSolve(lp, **kwargs)
            except PulpSolverError:
                if not HiGHS(msg=False).available():
                    raise
                CbcWithFallback.fallbacks += 1
                return HiGHS(msg=False).actualSolve(lp, **kwargs)

    def coin_cmd(msg=False, options=None):
        return CbcWithFallback(msg=msg, options=["randomSeed 42", "preprocess off"])

    discovery.COIN_CMD = coin_cmd
    return CbcWithFallback


SOLVER = patch_distribution_based()


# --- Table names -----------------------------------------------------------------

def name_lookup(tables, mode="raw"):
    """Names passed to Valentine (see ``preprocessing.table_names``):
    raw = object name in the database, base = without its type prefix,
    neutral = no names (Valentine's defaults)."""
    if mode == "neutral":
        return None
    names = {(row.split, row.table): row.name for row in tables.itertuples()}
    if mode == "base":
        names = {key: base_name(name) for key, name in names.items()}
    return names


# --- Matching ----------------------------------------------------------------------

def match_pair(source, target, matcher, names=None):
    """Valentine's column matches for one ordered table pair, as rows
    (source_column, target_column, score). The first DataFrame passed to
    Valentine is the source; if both get the same name (possible with
    ``base`` names) the target gets a suffix."""
    if names is not None and names[0] == names[1]:
        names = [names[0], names[1] + "_2"]
    result = valentine_match([source, target], matcher, df_names=names, instance_sample_size=None)
    first = names[0] if names else None
    rows = []
    for pair, score in result.items():
        if first is None or pair.source_table == first:
            rows.append((pair.source_column, pair.target_column, float(score)))
        else:
            rows.append((pair.target_column, pair.source_column, float(score)))
    return rows


def run_matchers(data, pairs, tables, matchers=None, names="raw", progress=True):
    """Match every candidate pair with every matcher.

    Returns (column scores, timings): one row per column pair returned by
    Valentine, and one row per (matcher, table pair) with the run time."""
    matchers = matchers or list(MATCHERS)
    lookup = name_lookup(tables, names)
    score_rows, time_rows = [], []
    for matcher_name in matchers:
        make = MATCHERS[matcher_name][1]
        start_all = time.perf_counter()
        for i, pair in enumerate(pairs.itertuples(), start=1):
            source, target = data[pair.split][pair.source], data[pair.split][pair.target]
            df_names = None if lookup is None else [lookup[(pair.split, pair.source)],
                                                    lookup[(pair.split, pair.target)]]
            start = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rows = match_pair(source, target, make(), df_names)
            time_rows.append({"matcher": matcher_name, "split": pair.split, "source": pair.source,
                              "target": pair.target, "seconds": time.perf_counter() - start})
            score_rows += [{"split": pair.split, "matcher": matcher_name, "source": pair.source,
                            "target": pair.target, "source_column": s, "target_column": t, "score": score}
                           for s, t, score in rows]
            if progress and i % 100 == 0:
                print(f"  {matcher_name}: {i}/{len(pairs)} pairs", flush=True)
        if progress:
            print(f"{matcher_name}: {len(pairs)} pairs in {time.perf_counter() - start_all:.0f} s", flush=True)
    return pd.DataFrame(score_rows), pd.DataFrame(time_rows)


def column_scores(names="raw", matchers=None, overwrite=False, folder=RESULTS_DIR, progress=True):
    """Column scores of all candidate pairs, computed once and cached.

    Matchers that are already in the cache are not run again, so the cache
    can be extended with new matchers or re-built one matcher at a time."""
    folder.mkdir(parents=True, exist_ok=True)
    scores_path = folder / f"column_scores_{names}.csv.gz"
    times_path = folder / f"timings_{names}.csv"
    matchers = matchers or list(MATCHERS)

    if scores_path.exists() and not overwrite:
        scores, times = pd.read_csv(scores_path), pd.read_csv(times_path)
        todo = [m for m in matchers if m not in set(times.matcher)]
    else:
        scores, times, todo = pd.DataFrame(), pd.DataFrame(), matchers

    if todo:
        data, pairs, tables = load_processed()
        for matcher in todo:   # save after every matcher, so a crash loses at most one
            new_scores, new_times = run_matchers(data, pairs, tables, [matcher], names, progress)
            scores = pd.concat([scores, new_scores], ignore_index=True)
            times = pd.concat([times, new_times], ignore_index=True)
            scores.to_csv(scores_path, index=False)
            times.to_csv(times_path, index=False)

    scores = scores[scores.matcher.isin(matchers)].reset_index(drop=True)
    times = times[times.matcher.isin(matchers)].reset_index(drop=True)
    return scores, times


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Valentine on all candidate pairs.")
    parser.add_argument("--names", default="raw", choices=["raw", "base", "neutral"])
    parser.add_argument("--matchers", nargs="*", default=None, help="subset of MATCHERS (default: all)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    scores, times = column_scores(args.names, args.matchers, args.overwrite)
    print(times.groupby("matcher").seconds.agg(["sum", "mean", "max"]).round(2))
