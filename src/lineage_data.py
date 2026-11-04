"""Load the Northwind lineage benchmark (github.com/dudenzz/lineage).

The benchmark comes in two CSV layouts:

* ``csv/train`` and ``csv/test``: ``;`` separated, ``\\r`` line endings, a
  second row with Python type names, nulls written as ``None`` and a
  ``DataLineage.csv`` log.
* ``results_csv/multi_scenario``: ``;`` separated, ``\\n`` line endings, no
  type row, nulls written as ``NULL`` and a ``lineage.csv`` log.

``load_split`` hides these differences and returns every table as a
DataFrame of strings plus the table-level lineage links.
"""

import ast
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

LINEAGE_DIR = Path(__file__).resolve().parent.parent / "lineage"

SPLITS = {
    "train": "csv/train",
    "test": "csv/test",
    "multi": "results_csv/multi_scenario",
}

NORTHWIND_TABLES = {
    "categories", "customercustomerdemo", "customerdemographics", "customers",
    "employees", "employeeterritories", "orderdetails", "orders", "products",
    "region", "shippers", "suppliers", "territories",
}

LOG_FILES = ("DataLineage.csv", "lineage.csv")
SKIPPED_FILES = ("lineageMetadata.csv",)
NULL_MARKERS = ("None", "NULL", "")


def norm(name):
    """Match file names and lineage-log names: vw_OrderFulfillment,
    vwOrderFulfillment.csv, [Order Details] and ##TempAtRisk all become
    lower-case letters and digits only."""
    name = re.sub(r"\.csv$", "", name)
    return re.sub(r"[^0-9a-z]", "", name.lower())


def clean_value(value):
    """Turn byte-string literals such as b'Toms Spezialit\\xe4ten' back
    into text. Returns (value, was_byte_string)."""
    if isinstance(value, str) and len(value) > 3 and value[0] == "b" and value[1] in "'\"" and value[-1] == value[1]:
        try:
            return ast.literal_eval(value).decode("latin-1"), True
        except (ValueError, SyntaxError):
            pass
    return value, False


def read_csv(path):
    """Read one benchmark CSV as strings, whatever its layout."""
    text = Path(path).read_text(encoding="utf-8")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if len(lines) > 1 and lines[1].startswith("<class"):
        del lines[1]
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep=";", dtype=str,
                     keep_default_na=False)
    return df.replace(list(NULL_MARKERS), pd.NA)


@dataclass
class Split:
    name: str
    tables: dict = field(default_factory=dict)        # norm name -> DataFrame
    display: dict = field(default_factory=dict)       # norm name -> name as in the log or file
    lineage_rows: pd.DataFrame = None                 # row-level log, columns source/target/...
    links: set = field(default_factory=set)           # {(source, target)} by norm name
    byte_strings: int = 0                             # values repaired by clean_value

    @property
    def objects(self):
        """Every object named in a file or in the lineage log."""
        names = set(self.tables)
        for s, t in self.links:
            names.update((s, t))
        return sorted(names)

    def kind(self, name):
        """original Northwind table, a derived object type, or a derived
        object left over from another scenario whose lineage is not logged."""
        if name in NORTHWIND_TABLES:
            return "original"
        if name not in {t for _, t in self.links}:
            return "unlogged"
        for prefix, kind in (("vw", "view"), ("temp", "temporary"), ("final", "final")):
            if name.startswith(prefix):
                return kind
        return "derived table"


def load_split(split):
    folder = LINEAGE_DIR / SPLITS[split]
    result = Split(split)

    for path in sorted(folder.glob("*.csv")):
        if path.name in LOG_FILES or path.name in SKIPPED_FILES:
            continue
        df = read_csv(path)
        repaired = 0
        for col in df.columns:
            cleaned = df[col].map(lambda v: clean_value(v) if isinstance(v, str) else (v, False))
            repaired += int(sum(flag for _, flag in cleaned))
            df[col] = [v for v, _ in cleaned]
        result.byte_strings += repaired
        result.tables[norm(path.name)] = df
        result.display[norm(path.name)] = path.stem

    log_path = next(folder / f for f in LOG_FILES if (folder / f).exists())
    log = read_csv(log_path)
    log.columns = ["source", "source_key", "source_value", "target", "target_key", "target_value"]
    for col in ("source_value", "target_value"):
        cleaned = log[col].map(lambda v: clean_value(v) if isinstance(v, str) else (v, False))
        result.byte_strings += int(sum(flag for _, flag in cleaned))
        log[col] = [v for v, _ in cleaned]
    result.lineage_rows = log

    for source, target in log[["source", "target"]].drop_duplicates().itertuples(index=False):
        result.links.add((norm(source), norm(target)))
        result.display.setdefault(norm(source), source)
        result.display.setdefault(norm(target), target)

    return result


def load_all():
    return {split: load_split(split) for split in SPLITS}
