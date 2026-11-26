"""Lineage scenario type of every evaluable link (notebooks/05_second_iteration.ipynb).

The reference paper reports its results per task: an operation (selection,
join, union) combined with a column transformation (projection, linear,
non-linear). The CSV exports do not label the scenarios, so we derive the
same two axes from the data and the row-level lineage log:

* operation: ``join`` when the target has two or more logged sources (every
  target row comes from one row of each source), else ``fewer rows``
  (selection or aggregation) or ``same rows`` (copy or transformation). No
  target in the three splits stacks rows of several sources, so there is no
  union;
* columns: ``projection`` when every target column is found in one of the
  target's sources (value containment >= 0.8, as in the EDA), else
  ``computed columns``;
* bridged links (over a dropped temporary table) form their own group,
  because the intermediate table is not visible.
"""

import pandas as pd

from eda import VALUE_MATCH, column_profile, evaluation_links, usable


def link_scenarios(splits):
    rows = []
    for split in splits.values():
        data = usable(split)
        sources = {}
        for s, t in split.links:
            sources.setdefault(t, set()).add(s)
        for (s, t), kind in sorted(evaluation_links(split).items()):
            source, target = data[s], data[t]
            own = (column_profile(source, target).containment >= VALUE_MATCH).to_numpy()
            found = own.copy()
            for other in sources.get(t, ()):
                if other in data and other != s:
                    found |= (column_profile(data[other], target).containment >= VALUE_MATCH).to_numpy()
            if kind == "bridged":
                operation = "bridged (hidden step)"
            elif len(sources.get(t, ())) >= 2:
                operation = "join"
            elif len(target) < len(source):
                operation = "fewer rows"
            else:
                operation = "same rows"
            rows.append({
                "split": split.name, "source": s, "target": t,
                "source_name": split.display[s], "target_name": split.display[t],
                "operation": operation,
                "columns": "projection" if found.all() else "computed columns",
                "target columns": len(own),
                "columns from this source": int(own.sum()),
                "rows source / target": f"{len(source)} / {len(target)}",
            })
    return pd.DataFrame(rows)
