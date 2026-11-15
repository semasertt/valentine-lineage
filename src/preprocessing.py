"""Pre-processing of the Northwind lineage benchmark (notebooks/03_preprocessing.ipynb).

``lineage_data.load_split`` reads the CSVs as they are. This module turns
them into the input of the matching step:

1. repair rows that were split by line breaks inside a value;
2. remove columns that are not business data: binary images and the
   lineage bookkeeping IDs written by the tracking framework;
3. normalise values (padding, decimal commas, trailing zeros, midnight
   timestamps) so that equal values are written the same way;
4. build the candidate table pairs with their ground-truth labels;
5. prepare the table names for the "table names on/off" experiment.

Each step returns the changes it made, so the notebook can report them.
"""

import re
from dataclasses import replace
from itertools import permutations
from pathlib import Path

import pandas as pd

from eda import containment, evaluation_links, pair_label, usable

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# Columns added by the lineage tracking of the benchmark, not by the
# scenarios: equal IDs in two tables reveal the link (leakage).
LINEAGE_COLUMNS = {"globalid", "parentgid"}

BINARY_SHARE = 0.5       # share of values with control characters
CONTROL = re.compile(r"[\x00-\x08\x0e-\x1f\x7f]|\xff\xff")
NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?")
TIMESTAMP = re.compile(r"(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})(?:\.\d+)?")
SMALL_INT = 100          # keys such as EmployeeID 1-9 or CategoryID 1-8
PREFIX = re.compile(r"^(?:##)?(?:vw_?|Table_?|[Tt]emp_?|Final_?)(?=[A-Z])")


# --- 1. Broken rows ----------------------------------------------------------

def repair_broken_rows(df):
    """Merge rows that were cut by a line break inside a value.

    In ``Employees.csv`` the address of two employees spans two lines, so
    the reader creates a row with the first half and a second row whose
    first field is the rest of the address and whose other fields are
    shifted by one. Such rows are recognised by a non-numeric value in a
    numeric key column. Returns (repaired table, number of merged rows)."""
    key = df.columns[0]
    is_id = df[key].map(lambda v: isinstance(v, str) and v.isdigit())
    if len(df) == 0 or is_id.all() or is_id.mean() < 0.5:
        return df, 0
    rows, merged = [], 0
    for values, ok in zip(df.itertuples(index=False), is_id):
        values = list(values)
        if ok or not rows:
            rows.append(values)
            continue
        previous = rows[-1]
        last = max(i for i, v in enumerate(previous) if not pd.isna(v))
        previous[last] = f"{previous[last]} {values[0]}"
        for i, value in enumerate(values[1:len(previous) - last], start=last + 1):
            previous[i] = value
        merged += 1
    return pd.DataFrame(rows, columns=df.columns), merged


# --- 2. Non-data columns -------------------------------------------------------

def is_binary(col):
    values = col.dropna()
    return len(values) > 0 and values.map(lambda v: bool(CONTROL.search(v))).mean() >= BINARY_SHARE


def non_data_columns(df):
    """{column: reason} for columns that carry no business data."""
    reasons = {}
    for col in df.columns:
        if col.lower() in LINEAGE_COLUMNS:
            reasons[col] = "lineage bookkeeping ID"
        elif is_binary(df[col]):
            reasons[col] = "binary content (image)"
    return reasons


# --- 3. Values -------------------------------------------------------------------

def normalise_value(value):
    """One spelling per value: '32,38' and '32.3800' become '32.38',
    '1996-07-04 00:00:00.000' becomes '1996-07-04', padded CHAR values
    are trimmed. Text is otherwise left as it is."""
    if pd.isna(value):
        return value
    value = value.strip()
    if not value:
        return pd.NA
    stamp = TIMESTAMP.fullmatch(value)
    if stamp:
        date, time = stamp.groups()
        return date if time == "00:00:00" else f"{date} {time}"
    if NUMBER.fullmatch(value) and ("." in value or "," in value):
        return value.replace(",", ".").rstrip("0").rstrip(".") or "0"
    return value


def normalise_table(df):
    """Normalised copy and the number of values whose spelling changed."""
    clean = df.apply(lambda col: col.map(normalise_value))
    changed = int((clean.fillna("\0") != df.fillna("\0")).sum().sum())
    return clean, changed


# --- All table-level steps --------------------------------------------------------

def preprocess_split(split, normalise=True):
    """Clean every table of a split. Returns (cleaned split, change log).
    ``normalise=False`` skips step 3 (used to measure its effect). Empty
    tables are kept here and left out later by ``eda.usable``."""
    tables, log = {}, []

    def note(step, name, detail, count):
        log.append({"split": split.name, "step": step, "table": split.display[name],
                    "detail": detail, "count": count})

    for name, df in split.tables.items():
        df, merged = repair_broken_rows(df)
        if merged:
            note("1. repair broken rows", name, f"{len(df)} rows left", merged)

        removed = non_data_columns(df)
        for col, reason in removed.items():
            note("2. remove non-data column", name, f"{col}: {reason}", 1)
        df = df.drop(columns=list(removed))

        df, changed = normalise_table(df) if normalise else (df, 0)
        if changed:
            note("3. normalise values", name, "values rewritten", changed)

        tables[name] = df.reset_index(drop=True)
    return replace(split, tables=tables), pd.DataFrame(log)


def preprocess_all(splits, normalise=True):
    cleaned, logs = {}, []
    for name, split in splits.items():
        cleaned[name], log = preprocess_split(split, normalise)
        logs.append(log)
    return cleaned, pd.concat(logs, ignore_index=True)


# --- 4. Candidate pairs ----------------------------------------------------------

def small_int_column(col):
    values = pd.to_numeric(col.dropna(), errors="coerce")
    return len(values) > 0 and values.notna().all() and (values % 1 == 0).all() \
        and values.between(0, SMALL_INT).all()


def small_int_match(source, target):
    """True when a target column of small integers is fully contained in a
    source column: such keys occur in many unrelated tables (EDA 2.3)."""
    return any(small_int_column(target[t]) and containment(target[t], source[s]) == 1
               for t in target.columns for s in source.columns)


def pair_kind(split, name):
    return "original" if split.kind(name) == "original" else "derived"


def candidate_pairs(split):
    """Every ordered pair of usable tables with its ground-truth relation.

    ``is_lineage`` is the label the matchers are evaluated against;
    ``link`` says whether a lineage link is direct or bridged over a
    temporary table without data; ``hard_negative`` marks non-links that
    are close in the lineage graph (reverse, indirect, sibling)."""
    tables = usable(split)
    links = evaluation_links(split)
    rows = []
    for s, t in permutations(sorted(tables), 2):
        label = pair_label(split, links, s, t)
        rows.append({
            "split": split.name,
            "source": s,
            "target": t,
            "source_name": split.display[s],
            "target_name": split.display[t],
            "direction": f"{pair_kind(split, s)} -> {pair_kind(split, t)}",
            "label": label,
            "is_lineage": label == "lineage",
            "link": links.get((s, t), "-"),
            "hard_negative": label in ("reverse", "indirect", "sibling"),
            "small_int_match": small_int_match(tables[s], tables[t]),
        })
    return pd.DataFrame(rows)


# --- 5. Table names ----------------------------------------------------------------

def base_name(display):
    """Table name without the object-type prefix: vwOrderFulfillment ->
    OrderFulfillment, ##TempAtRisk -> AtRisk, FinalOrderDetails -> OrderDetails."""
    return PREFIX.sub("", display.strip("[]").replace(" ", ""))


def table_names(split, mode):
    """Names passed to Valentine as ``df_names``:
    raw     the object name from the database,
    base    the name without its type prefix,
    neutral None, so Valentine uses names without any similarity."""
    if mode == "neutral":
        return None
    return {n: split.display[n] if mode == "raw" else base_name(split.display[n]) for n in usable(split)}


# --- Saving ---------------------------------------------------------------------

def save(splits, pairs, log, folder=DATA_DIR):
    """Write cleaned tables, candidate pairs and the change log to data/processed/."""
    for split in splits.values():
        out = folder / split.name
        out.mkdir(parents=True, exist_ok=True)
        for name, df in usable(split).items():
            df.to_csv(out / f"{name}.csv", index=False)
    rows = [{"split": s.name, "table": n, "name": s.display[n], "kind": s.kind(n),
             "rows": len(df), "columns": df.shape[1]}
            for s in splits.values() for n, df in usable(s).items()]
    pd.DataFrame(rows).to_csv(folder / "tables.csv", index=False)
    pairs.to_csv(folder / "pairs.csv", index=False)
    log.to_csv(folder / "preprocessing_log.csv", index=False)
    return folder


def load_processed(folder=DATA_DIR):
    """Read the output of ``save``: ({split: {table: DataFrame}}, pairs, tables)."""
    tables = pd.read_csv(folder / "tables.csv")
    data = {split: {name: pd.read_csv(folder / split / f"{name}.csv", dtype=str, keep_default_na=False)
                    .replace("", pd.NA)
                    for name in group.table}
            for split, group in tables.groupby("split")}
    return data, pd.read_csv(folder / "pairs.csv"), tables
