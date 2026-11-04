<div align="center">

# 🔗 Valentine for Data Lineage Discovery

**Can a schema-matching tool find the missing links between database tables?**

Semantic Web course project · 2026

![Python](https://img.shields.io/badge/python-3.12%20|%203.13%20|%203.14-blue)
![Valentine](https://img.shields.io/badge/valentine-1.1-orange)
![Platform](https://img.shields.io/badge/platform-Windows%20|%20macOS%20|%20Linux-lightgrey)
![Status](https://img.shields.io/badge/status-setup-yellow)

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
8. [Project structure](#-project-structure)
9. [Troubleshooting](#-troubleshooting)
10. [Roadmap](#-roadmap)
11. [References](#-references)

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

This installs Valentine together with its dependencies (NLTK, PuLP + CBC solver, POT, RapidFuzz…), plus pandas, scikit-learn, networkx, matplotlib and Jupyter.

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

## 🗂️ Project structure

```
valentine-lineage/
├── README.md            ← you are here
├── requirements.txt     ← Python packages
├── .gitignore
├── docs/                ← notes, report drafts, slides
│   └── figures/         ← figures produced by the notebooks
├── notebooks/
│   └── 01_eda.ipynb     ← exploratory data analysis (Milestone 2)
├── src/
│   ├── lineage_data.py  ← load and clean the benchmark CSVs
│   └── eda.py           ← overlap and name-similarity measures
├── lineage/             ← data repo, cloned locally (git-ignored)
└── .venv/               ← virtual environment (git-ignored)
```

Code, results and the report will be added step by step (see roadmap).

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `No suitable Python runtime found` | That Python version is not installed. Run `py -0` and use a listed 3.12–3.14 version, or install one from python.org. |
| `cd $HOME\code` → *path not found* | `$HOME` only works in PowerShell. In CMD use `%USERPROFILE%`. |
| `running scripts is disabled` (PowerShell) | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again. |
| `No matching distribution found for …` | Usually a typo in the package name, or a Python version outside 3.12–3.14. |
| `ModuleNotFoundError: valentine` | The virtual environment is not active. Run `.venv\Scripts\activate.bat`. |
| `git` is not recognised | Install Git (`winget install -e --id Git.Git`) and reopen the terminal. |

## 🗺️ Roadmap

- [x] Topic and reference paper chosen
- [x] Environment set up (Python, Git, Valentine)
- [x] Data source identified (`dudenzz/lineage` CSV exports)
- [x] Data loading and ground truth (table-level lineage)
- [x] Exploratory data analysis
- [ ] Run all Valentine matchers on all table pairs
- [ ] Turn column scores into table scores and evaluate (Precision, Recall, PR-AUC, Hits@k)
- [ ] Error analysis: which kinds of lineage are found or missed
- [ ] Extra experiments: table names on/off, lineage direction
- [ ] Optional: add Valentine matches to the RDDL knowledge graph
- [ ] Report and presentation

## 📖 References

- J. Dutkiewicz, P. Misiorek, R. Wrembel. **Relational Database Data Lineage Ontology.** arXiv:2605.16068, 2026. · [code & data](https://github.com/dudenzz/lineage)
- J. Dutkiewicz, P. Misiorek, R. Wrembel. **Data Lineage Discovery in Databases Based on Knowledge Graph Link Prediction.** EDBT/ICDT Workshops, CEUR-WS vol. 4192, 2026.
- C. Koutras et al. **Valentine: Evaluating Matching Techniques for Dataset Discovery.** IEEE ICDE 2021. · [documentation](https://delftdata.github.io/valentine/)
- Microsoft. **Northwind sample database.** [sql-server-samples](https://github.com/microsoft/sql-server-samples/tree/master/samples/databases/northwind-pubs)
