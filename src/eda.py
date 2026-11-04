"""Measures used in the exploratory analysis (notebooks/01_eda.ipynb).

They approximate the two signals Valentine relies on:

* data matching: how many distinct values of a target column also occur in
  a source column (containment);
* schema matching: how similar the column names are.
"""

from difflib import SequenceMatcher

import networkx as nx
import pandas as pd

VALUE_MATCH = 0.8   # containment from which a column counts as "copied"


def containment(target_col, source_col):
    """Share of the distinct target values that also occur in the source."""
    target = set(target_col.dropna())
    if not target:
        return 0.0
    return len(target & set(source_col.dropna())) / len(target)


def name_similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def column_profile(source, target):
    """For every target column: the source column that holds most of its
    values, how much of it, and how close the two names are."""
    rows = []
    for col in target.columns:
        best_col, best = max(((s, containment(target[col], source[s])) for s in source.columns),
                             key=lambda x: x[1])
        rows.append({
            "target_column": col,
            "best_source_column": best_col,
            "containment": best,
            "name_similarity": name_similarity(col, best_col),
            "best_name_similarity": max(name_similarity(col, s) for s in source.columns),
        })
    profile = pd.DataFrame(rows)
    copied = profile.containment >= VALUE_MATCH
    same_name = profile.target_column.str.lower() == profile.best_source_column.str.lower()
    profile["category"] = "values changed"
    profile.loc[copied & same_name, "category"] = "same name, same values"
    profile.loc[copied & ~same_name, "category"] = "renamed, same values"
    return profile


def table_scores(source, target):
    """Table-pair scores from the column profile: the mean and the maximum
    of the best value containment over the target's columns (mean rewards
    full copies, max also catches a source that gave only one column to a
    join), and the mean best name similarity."""
    profile = column_profile(source, target)
    return {
        "value_mean": profile.containment.mean(),
        "value_max": profile.containment.max(),
        "name_mean": profile.best_name_similarity.mean(),
    }


def usable(split):
    """Tables Valentine can actually read: a file exists and it has rows."""
    return {name: df for name, df in split.tables.items() if len(df) > 0}


def evaluation_links(split):
    """Lineage links between usable tables. A link through an object
    without data (a dropped temporary table) is bridged: A -> ##Temp -> B
    becomes A -> B."""
    tables = usable(split)
    graph = nx.DiGraph(split.links)
    links = {}
    for source, target in split.links:
        if source in tables and target in tables:
            links[(source, target)] = "direct"
    for hidden in [n for n in graph if n not in tables]:
        for source in graph.predecessors(hidden):
            for target in graph.successors(hidden):
                if source in tables and target in tables:
                    links.setdefault((source, target), "bridged")
    return links


PAIR_LABELS = ["lineage", "reverse", "indirect", "sibling", "unlogged", "unrelated"]


def pair_label(split, links, source, target):
    """How an ordered pair of tables is related in the ground truth:
    lineage   source -> target is a (direct or bridged) lineage link
    reverse   target -> source is a lineage link
    indirect  source is a further ancestor of target, or the other way round
    sibling   both were derived from a common source
    unlogged  one of them is a derived object whose lineage is not logged
    unrelated none of the above."""
    graph = nx.DiGraph(list(links))
    if (source, target) in links:
        return "lineage"
    if (target, source) in links:
        return "reverse"
    if source in graph and target in graph:
        if nx.has_path(graph, source, target) or nx.has_path(graph, target, source):
            return "indirect"
        if set(graph.predecessors(source)) & set(graph.predecessors(target)):
            return "sibling"
    if "unlogged" in (split.kind(source), split.kind(target)):
        return "unlogged"
    return "unrelated"


def pair_scores(split):
    """Scores and ground-truth label for every ordered pair of usable tables."""
    tables = usable(split)
    links = evaluation_links(split)
    rows = []
    for s in tables:
        for t in tables:
            if s == t:
                continue
            rows.append({"split": split.name, "source": split.display[s], "target": split.display[t],
                         **table_scores(tables[s], tables[t]),
                         "label": pair_label(split, links, s, t)})
    return pd.DataFrame(rows)
