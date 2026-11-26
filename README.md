<div align="center">

# 🔗 Valentine for Data Lineage Discovery

**Can a schema-matching tool find the missing links between database tables?**

Semantic Web course project · 2026

![Python](https://img.shields.io/badge/python-3.12%20|%203.13%20|%203.14-blue)
![Valentine](https://img.shields.io/badge/valentine-1.1-orange)
![Platform](https://img.shields.io/badge/platform-Windows%20|%20macOS%20|%20Linux-lightgrey)
![Status](https://img.shields.io/badge/status-first%20results-blue)

</div>

---

## 📑 Contents

1. [About the project](#-about-the-project)
2. [Background](#-background)
3. [Data](#-data)
4. [Requirements](#-requirements)
5. [Installation](#-installation)
6. [Verify the installation](#-verify-the-installation)
7. [Exploratory analysis](#-exploratory-analysis)
8. [Tools and pre-processing](#-tools-and-pre-processing)
9. [Experiments and first results](#-experiments-and-first-results)
10. [Second iteration and draft report](#-second-iteration-and-draft-report)
11. [Project structure](#-project-structure)
12. [Troubleshooting](#-troubleshooting)
13. [Roadmap](#-roadmap)
14. [References](#-references)

---

## 🎯 About the project

In real databases, tables and views are constantly **derived** from other tables: they are copied, filtered, joined or transformed. This chain of "what came from where" is called **data lineage**, and it is rarely documented.

```
Customers ─┐
Orders ────┼──► vw_OrderFulfillment ──► Table_FulfillmentLeads ──► ##TempAtRisk ──► Final_AtRiskLog
Shippers ──┘
```

When a link in this chain is missing (**broken lineage**), it is hard to trace errors, audit data or know what breaks when a table changes.

**Goal of this project:** evaluate how well **[Valentine](https://delftdata.github.io/valentine/)**, an open-source column matching library, can recover missing lineage links. We compare it with the knowledge-graph + graph-neural-network approach of the *Relational Database Data Lineage Ontology* paper, using the same Northwind-based data.

| | Reference paper | This project |
|---|---|---|
| Data | Northwind + lineage scenarios | same |
| Method | DB → knowledge graph → GNN link prediction | Valentine column matching |
| Output | "table B is derived from table A" | same |
| Metrics | Precision, Recall, AUC, Hits@10 | same (+ AUROC, MRR) |

## 📚 Background

### Valentine

Valentine takes two tables and scores how likely each pair of columns is to correspond. It offers two families of matchers:

| Matcher | Uses | Idea |
|---|---|---|
| `Coma` | names (+ values) | combines several name/structure similarities |
| `Cupid` | names | linguistic + structural similarity |
| `SimilarityFlooding` | names | similarity propagation over a graph |
| `JaccardDistanceMatcher` | values | overlap of the value sets |
| `DistributionBased` | values | similarity of value distributions |

- **Schema matching** asks whether the column *names* look alike (`CustomerID` ↔ `CustID`).
- **Data matching** asks whether the column *contents* look alike (`Country` ↔ `Region`, both containing *Germany, Sweden*).

### Reference paper

Dutkiewicz, Misiorek and Wrembel (Poznan University of Technology) model a relational database as a **knowledge graph** using their **RDDL ontology**, built on PROV-O and CSVW. They then predict missing `tableDerivedFrom` / `rowDerivedFrom` links with a **path-based Siamese GNN**.

## 🗄️ Data

All data comes from the authors' repository **[dudenzz/lineage](https://github.com/dudenzz/lineage)**. It contains Northwind tables, objects created by lineage scenarios, and a `DataLineage` log that serves as the ground truth.

| Dataset | Folder in `lineage/` | Content |
|---|---|---|
| train | `csv/train` | Northwind + several scenario objects + lineage log |
| test | `csv/test` | Northwind + one scenario chain + lineage log |
| multi | `results_csv/multi_scenario` | Several chains incl. temporary tables + lineage log |

> The full benchmark of the paper (9 tasks × 20 scenarios) is only available as T-SQL scripts that need Microsoft SQL Server. We start with the ready-made CSV exports above.

The data repository is **not** copied into this repo. Every team member clones it locally (see below).

## 🧰 Requirements

| Tool | Version | Why |
|---|---|---|
| [Python](https://www.python.org/downloads/) | **3.12, 3.13 or 3.14** | Valentine needs ≥ 3.12. On Windows/macOS 3.15 is not supported yet. |
| [Git](https://git-scm.com/downloads) | any recent | to clone the repositories |
| Disk space | ~1 GB | Python packages (~400 MB) + data repo (~180 MB) |

No database server is needed.

Check what you have:

```bash
py -0            # Windows: lists installed Python versions
python3 --version   # macOS / Linux
git --version
```

## ⚙️ Installation

The commands below are for **Windows (Command Prompt / CMD)**. macOS/Linux equivalents are given underneath.

### 1. Clone this repository

```cmd
cd /d C:\path\to\your\workspace
git clone https://github.com/semasertt/valentine-lineage.git
cd valentine-lineage
```

### 2. Clone the data repository *inside* it

```cmd
git clone https://github.com/dudenzz/lineage.git
```

This creates a `lineage\` folder (ignored by git). Check it:

```cmd
dir lineage\csv\train
```

You should see `Customers.csv`, `DataLineage.csv`, `vwOrderFulfillment.csv`, …

### 3. Create and activate a virtual environment

```cmd
py -3.13 -m venv .venv
.venv\Scripts\activate.bat
```

The prompt now starts with `(.venv)`. **Activate it every time you open a new terminal.**

<details>
<summary>PowerShell / macOS / Linux</summary>

```powershell
# PowerShell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```
</details>

### 4. Install the packages

```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This installs Valentine together with its dependencies (NLTK, PuLP + CBC solver, POT, RapidFuzz…), plus pandas, scikit-learn, networkx, matplotlib, Jupyter and HiGHS (`highspy`, a fallback solver for Valentine's Distribution-based matcher, see `src/matching.py`).

Valentine downloads a few NLTK resources the first time it runs. If that fails (e.g. behind a proxy), download them once by hand:

```cmd
python -m nltk.downloader punkt_tab omw-1.4 stopwords wordnet
```

## ✅ Verify the installation

Run this one-liner. It matches two tiny tables with Valentine:

```cmd
python -c "import pandas as pd; from valentine import valentine_match; from valentine.algorithms import Coma; a=pd.DataFrame({'CustomerID':['ALFKI','BERGS'],'Country':['Germany','Sweden']}); b=pd.DataFrame({'CustID':['ALFKI','BERGS'],'Region':['Germany','Sweden']}); m=valentine_match([a,b],Coma(use_instances=True)); [print(p.source_column,'<->',p.target_column,round(s,2)) for p,s in m.items()]"
```

Expected output:

```
CustomerID <-> CustID 1.0
Country <-> Region 0.5
```

`Country ↔ Region` is found although the names differ, because the **values** are the same. That is exactly the signal we want to use for lineage.

## 📊 Exploratory analysis

The exploratory analysis (Milestone 2) is in [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb). It is committed with its outputs, so you can read it on GitHub without running anything.

To run it yourself (virtual environment active, `lineage/` cloned):

```cmd
cd notebooks
jupyter notebook 01_eda.ipynb
```

Or open the file in VS Code and pick the `.venv` interpreter as kernel. *Run All* takes about 30 seconds and rewrites the figures in [`docs/figures/`](docs/figures/).

| Part | What it shows |
|---|---|
| 1. Basic analysis | objects per split, sizes, column types, missing values, data-quality issues, lineage graphs |
| 2. Specific analysis | how much shared values (data matching) and similar column names (schema matching) separate true lineage links from other table pairs |
| 3. Main outcomes | the key numbers in one table |

The loading and cleaning code is in [`src/lineage_data.py`](src/lineage_data.py) and the measures in [`src/eda.py`](src/eda.py). The two-page summary of the results is [`docs/eda_report.pdf`](docs/eda_report.pdf).

## 🧪 Tools and pre-processing

Milestone 3. The tools we use and why:

| Tool | Used for | Why this tool |
|---|---|---|
| **Valentine 1.1** | column matching: the method we evaluate | open-source library with both matcher families: schema-based (COMA, Cupid, Similarity Flooding) and instance-based (Jaccard, Distribution-based); the EDA showed that both signals are present in the data |
| **pandas / NumPy** | loading, cleaning, candidate pairs | Valentine works on pandas DataFrames; the benchmark is a set of CSV files |
| **NetworkX** | ground truth: lineage graph, bridging over temporary tables, pair labels | the lineage log is a directed graph |
| **scikit-learn** | Precision, Recall, PR-AUC, AUROC | the metrics of the baseline paper; Hits@k and MRR are implemented on top |
| **Matplotlib** | figures for the reports | |
| **Jupyter, VS Code** | notebooks committed with their outputs | readable on GitHub without running anything |
| **Git, GitHub, GitHub Projects** | code repository and Kanban board | |

Two notebooks belong to this milestone:

| Notebook | What it does |
|---|---|
| [`notebooks/02_tools_intro.ipynb`](notebooks/02_tools_intro.ipynb) | basic use of Valentine (matching two tables, one-to-one matches, the five matchers on a renamed column, column scores → table score), NetworkX (lineage graph, bridging) and scikit-learn (evaluating a ranking, Hits@k, MRR) |
| [`notebooks/03_preprocessing.ipynb`](notebooks/03_preprocessing.ipynb) | the pre-processing: repairs broken rows, removes images and lineage bookkeeping IDs (`GlobalID`, which leaks the links), normalises values, builds the 30 evaluable links and the 842 labelled candidate table pairs, prepares the table-name variants |

The pre-processing code is in [`src/preprocessing.py`](src/preprocessing.py). Running `03_preprocessing.ipynb` (about 45 seconds) writes the cleaned tables and `pairs.csv` to `data/processed/` (git-ignored); later notebooks read them with `preprocessing.load_processed()`.

## 🔬 Experiments and first results

Milestone 4. [`notebooks/04_experiments.ipynb`](notebooks/04_experiments.ipynb) runs Valentine on all 842 candidate pairs and evaluates the result. It needs the output of `03_preprocessing.ipynb`.

```cmd
cd notebooks
jupyter notebook 04_experiments.ipynb
```

The first run matches all pairs with all matchers (**about 4 minutes**, cached in `results/column_scores_raw.csv.gz`); later runs take about 50 seconds. The matching step can also be run from the command line:

```cmd
cd src
python matching.py                 # all matchers, raw table names
python matching.py --names neutral # table names on/off experiment (Milestone 5)
```

| Step | Code |
|---|---|
| 6 Valentine configurations (COMA schema / schema + instances, Cupid, Similarity Flooding, Jaccard, Distribution-based) × 842 ordered pairs | [`src/matching.py`](src/matching.py) |
| column scores → table-pair score (`max`, `top3`, `mean`, `coverage` of the best score per target column) | [`src/evaluation.py`](src/evaluation.py) |
| PR-AUC, AUROC, Hits@k, MRR; precision / recall at a threshold chosen on `train`; baselines; error analysis | [`src/evaluation.py`](src/evaluation.py) |

**Protocol.** Valentine is not trained. The configuration and the threshold are chosen on `train`; `test` + `multi` (19 of the 30 links) are reported as held-out.

**First results (held-out splits)**

| Method | PR-AUC | AUROC | Hits@1 | Hits@3 | MRR |
|---|---|---|---|---|---|
| random ranking | 0.06 | 0.50 | 0.09 | 0.26 | 0.26 |
| table-name similarity (no Valentine) | 0.12 | 0.64 | 0.40 | 0.53 | 0.52 |
| value containment (no Valentine) | 0.22 | 0.86 | 0.33 | 0.73 | 0.54 |
| Valentine COMA (schema) + mean | 0.18 | 0.78 | 0.20 | 0.47 | 0.45 |
| Valentine COMA (schema + instances) + mean | 0.20 | 0.85 | 0.33 | 0.60 | 0.55 |
| **Valentine Jaccard distance + mean** (chosen on train) | **0.21** | **0.86** | **0.40** | **0.73** | **0.59** |

- **Values beat names**, as the EDA predicted; `mean` is the best aggregation for every matcher.
- At the train threshold: precision 0.27, recall 0.32 for the exact link, but **every** false positive is a `reverse` or `indirect` pair, so all predicted pairs lie in a true lineage chain (same-chain precision 1.0). No unrelated pair or small-integer key match is predicted.
- The direction is right for only 18 of 30 links; the missed links are bridged links, single-column join sources and transformed columns.
- A hand-written value-overlap baseline is as good as Valentine's best matcher: Valentine's benefit here is an off-the-shelf, training-free, explainable signal, not a stronger one.
- The GNN of the reference paper is evaluated at row level on a knowledge graph, so the numbers are not directly comparable; section 7 of the notebook gives the closest comparison (Hits@10 against 50 sampled negatives: 0.74).

Figures: [`docs/figures/exp_*.png`](docs/figures/). Metric tables: `results/metrics_*.csv`.

## 🧩 Second iteration and draft report

Milestone 5. [`notebooks/05_second_iteration.ipynb`](notebooks/05_second_iteration.ipynb) tests the four next steps of the first iteration. It needs the cached column scores of all three table-name variants:

```cmd
cd src
python matching.py --names raw
python matching.py --names neutral
python matching.py --names base
```

| Question | Code | Answer (held-out) |
|---|---|---|
| Do table names make the task easier? | [`src/matching.py`](src/matching.py) | **no**: COMA, Cupid, Jaccard give identical scores with raw, base and neutral names |
| Does mixing `mean` and `max` help join sources? | [`src/combination.py`](src/combination.py) | **no** gain |
| Does combining matchers help? | [`src/combination.py`](src/combination.py) | stacking (logistic regression on all 24 scores, fitted on train): PR-AUC 0.21 → 0.25, Hits@3 0.73 → 0.93 |
| Can a second signal give the direction? | [`src/combination.py`](src/combination.py) | object type order: 29/30 directions right (score alone 18/30); PR-AUC → **0.36**, MRR → **0.76**; relies on the benchmark's naming |
| Which kinds of lineage are hard? | [`src/scenarios.py`](src/scenarios.py) | joins with single-column sources and bridged links |

| System | PR-AUC | AUROC | Hits@1 | Hits@3 | MRR |
|---|---|---|---|---|---|
| first iteration (Jaccard distance + mean) | 0.21 | 0.86 | 0.40 | 0.73 | 0.59 |
| stacking | 0.25 | 0.86 | 0.40 | 0.93 | 0.65 |
| **stacking + type direction** | **0.36** | **0.92** | **0.60** | **0.93** | **0.76** |

Figures: [`docs/figures/it2_*.png`](docs/figures/). Metrics: `results/metrics_second_iteration.csv`.

The **draft technical report** is [`docs/report_draft.pdf`](docs/report_draft.pdf) (LaTeX source in [`docs/report/`](docs/report/)). Parts that are still to be written are marked with TODO boxes; who does what is in [`plan.md`](plan.md).

## 🗂️ Project structure

```
valentine-lineage/
├── README.md            ← you are here
├── plan.md              ← project plan, status and who does what
├── requirements.txt     ← Python packages
├── .gitignore
├── docs/                ← reports, Kanban pictures
│   ├── report/          ← LaTeX source of the technical report
│   ├── report_draft.pdf ← draft technical report (Milestone 5)
│   └── figures/         ← figures produced by the notebooks
├── notebooks/
│   ├── 01_eda.ipynb            ← exploratory data analysis (Milestone 2)
│   ├── 02_tools_intro.ipynb    ← introduction to the tools (Milestone 3)
│   ├── 03_preprocessing.ipynb  ← pre-processing (Milestone 3)
│   ├── 04_experiments.ipynb    ← matching, evaluation, error analysis (Milestone 4)
│   └── 05_second_iteration.ipynb ← combination, names on/off, direction, scenario types (Milestone 5)
├── src/
│   ├── lineage_data.py  ← load the benchmark CSVs
│   ├── eda.py           ← overlap and name-similarity measures, ground truth
│   ├── preprocessing.py ← cleaning, candidate pairs, table names
│   ├── matching.py      ← runs the Valentine matchers on all pairs (cached)
│   ├── evaluation.py    ← table scores, metrics, baselines, error analysis
│   ├── combination.py   ← hybrid aggregation, matcher combination, stacking, direction
│   └── scenarios.py     ← scenario type of every lineage link
├── data/processed/      ← output of 03_preprocessing (git-ignored)
├── results/             ← column scores and metric tables of 04_experiments (git-ignored)
├── lineage/             ← data repo, cloned locally (git-ignored)
└── .venv/               ← virtual environment (git-ignored)
```

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `No suitable Python runtime found` | That Python version is not installed. Run `py -0` and use a listed 3.12–3.14 version, or install one from python.org. |
| `cd $HOME\code` → *path not found* | `$HOME` only works in PowerShell. In CMD use `%USERPROFILE%`. |
| `running scripts is disabled` (PowerShell) | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again. |
| `No matching distribution found for …` | Usually a typo in the package name, or a Python version outside 3.12–3.14. |
| `ModuleNotFoundError: valentine` | The virtual environment is not active. Run `.venv\Scripts\activate.bat`. |
| `git` is not recognised | Install Git (`winget install -e --id Git.Git`) and reopen the terminal. |
| `Failed to download required NLTK resource(s)` | Run `python -m nltk.downloader punkt_tab omw-1.4 stopwords wordnet` once. |
| `PulpSolverError: ... cbc` (Distribution-based) | Import Valentine through `src/matching.py`, which patches the solver call; install `highspy` for the fallback solver. |

## 🗺️ Roadmap

- [x] Topic and reference paper chosen
- [x] Environment set up (Python, Git, Valentine)
- [x] Data source identified (`dudenzz/lineage` CSV exports)
- [x] Data loading and ground truth (table-level lineage)
- [x] Exploratory data analysis
- [x] Tool selection and introductory notebook
- [x] Pre-processing and candidate pairs
- [x] Run all Valentine matchers on all table pairs
- [x] Turn column scores into table scores and evaluate (Precision, Recall, PR-AUC, Hits@k)
- [x] Comparison with the GNN baseline paper (first iteration)
- [x] Error analysis: which kinds of lineage are found or missed (by relation and scenario type)
- [x] Combine aggregations and matchers (second iteration)
- [x] Extra experiments: table names on/off, lineage direction
- [ ] Draft technical report (skeleton and sections 1-5 done, results text in progress, see `plan.md`)
- [ ] Optional: add Valentine matches to the RDDL knowledge graph
- [ ] Final report and presentation

## 📖 References

- J. Dutkiewicz, P. Misiorek, R. Wrembel. **Relational Database Data Lineage Ontology.** arXiv:2605.16068, 2026. · [code & data](https://github.com/dudenzz/lineage)
- J. Dutkiewicz, P. Misiorek, R. Wrembel. **Data Lineage Discovery in Databases Based on Knowledge Graph Link Prediction.** EDBT/ICDT Workshops, CEUR-WS vol. 4192, 2026.
- C. Koutras et al. **Valentine: Evaluating Matching Techniques for Dataset Discovery.** IEEE ICDE 2021. · [documentation](https://delftdata.github.io/valentine/)
- Microsoft. **Northwind sample database.** [sql-server-samples](https://github.com/microsoft/sql-server-samples/tree/master/samples/databases/northwind-pubs)
