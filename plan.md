# Project plan and status

*Evaluating lineage link prediction with Valentine* · Group 6 · state on **26.11.2026**

This file says where the project stands and who does what until the end. The Kanban picture of the same plan is
[`docs/kanban_m5.png`](docs/kanban_m5.png).

## Milestones

| # | Milestone | Due | Status | Where |
|---|---|---|---|---|
| 1 | Topic, literature, team | 26.10 | ✅ done | course document |
| 2 | Dataset, exploratory analysis | 09.11 | ✅ done (10/10) | `notebooks/01_eda.ipynb`, `docs/eda_report.pdf` |
| 3 | Tools, pre-processing | 16.11 | ✅ done | `notebooks/02_tools_intro.ipynb`, `notebooks/03_preprocessing.ipynb` |
| 4 | First iteration of experiments | 30.11 | ✅ done (9.5/10) | `notebooks/04_experiments.ipynb` |
| 5 | Second iteration + **draft report** | **14.12** | 🟡 experiments and draft report done, presentation open | `notebooks/05_second_iteration.ipynb`, `docs/report/` |
| 6 | Final presentation | 11-18.01 | ⬜ | |
| – | Final report + final code | 11-18.01 | ⬜ | |

## Milestone 5: what is done

**Second iteration of experiments** (`notebooks/05_second_iteration.ipynb`, code in `src/combination.py` and
`src/scenarios.py`). All four next steps of Milestone 4 are tested; held-out results (test + multi):

| system | PR-AUC | AUROC | Hits@1 | Hits@3 | MRR |
|---|---|---|---|---|---|
| first iteration (Jaccard distance + mean) | 0.21 | 0.86 | 0.40 | 0.73 | 0.59 |
| hybrid mean/max aggregation | 0.20 | 0.84 | 0.40 | 0.73 | 0.60 |
| average of 6 matchers | 0.21 | 0.85 | 0.40 | 0.87 | 0.64 |
| stacking (logistic regression on all scores, C by CV on train) | 0.25 | 0.86 | 0.40 | 0.93 | 0.65 |
| **stacking + object-type direction (final)** | **0.36** | **0.92** | **0.60** | **0.93** | **0.76** |

Key findings:

1. **Table names on/off:** no effect. COMA, Cupid and Jaccard give identical scores with raw, base and neutral
   names; Similarity Flooding and Distribution-based change by < 0.01 PR-AUC. Our results are not inflated by the
   benchmark's naming.
2. **Hybrid mean/max:** no gain. Pure `max` is bad; any share of `mean` gives the `mean` result.
3. **Stacking** (logistic regression on the 24 matcher × aggregation scores, fitted on train): PR-AUC 0.21 → 0.25,
   Hits@3 0.73 → 0.93. Only 11 training positives, so treat it as indicative.
4. **Direction:** the score gets 18/30 directions right, the object type order (base table < view < table < temp <
   final) 29/30. Demoting reverse pairs raises PR-AUC to 0.36. **Caveat:** the order comes from the benchmark's
   naming convention; a catalog-only prior (base tables first) barely helps.
5. **Scenario types:** joins (single-column lookup sources such as `Shippers`) and bridged links are the hardest,
   like the join tasks in the paper. Final system: Hits@3 = 1.0 for "fewer rows" and "same rows" links.
6. Paper-style Hits@10 against 50 / 100 sampled negatives: 0.88 / 0.75 (first iteration 0.74 / 0.53).

**Draft report** (`docs/report/report.tex` → `docs/report_draft.pdf`, LaTeX, 9 pages): abstract,
1 Introduction, 2 Related work, 3 Data, 4 Method (with the pipeline figure), 5 Experimental setup, 6 Results (first
iteration, second iteration, comparison with the paper), 7 Discussion and limitations, 8 Conclusions, 13 checked
references. For the final report (January) it needs a review by the whole team.

**Kanban:** `docs/kanban_m5.png`.

## Milestone 5: what is left (due 14.12)

Everything except the presentation is done: the second iteration, the complete draft report
(`docs/report_draft.pdf`, all sections 1-8 written), this plan, the Kanban and the Milestone 5 fields of the course
document. **Only the presentation is left:**

- [ ] **All (Sema, Yiğit, Alessandro, Achraf)**: Milestone 5 presentation (second-iteration results) and its link
      in the course document, due **14.12**. Material: Table 3-5 and Figures 1-3 of `docs/report_draft.pdf`,
      `docs/figures/it2_*.png`, the summary table above.

Suggested slides: (1) recap of the problem and of the first iteration; (2) the four questions of the second
iteration; (3) table names on/off; (4) combining aggregations and matchers (stacking); (5) direction of a link
(Table 4) and its caveat; (6) results by scenario type (Figure 3); (7) final numbers vs. the first iteration
(`it2_02_progress.png`) and vs. the paper (Table 5); (8) limitations and next steps.

## After Milestone 5

| Task | Owner | Due |
|---|---|---|
| Optional: write the predicted links as `tableDerivedFrom` triples into the RDDL knowledge graph (rdflib) | Achraf | 21.12 |
| Final report: review and extend the draft (graded on references, figures, tables, sections, formatting) | All | 11.01 |
| Final presentation | All | 11.01 |
| Reproducibility check from a fresh clone, README cleanup | Sema | 11.01 |

Open research points that could go into the final report: a direction signal that needs only catalog
information (creation time), a cross-split check of the stacking weights (train on multi, test on train), and the
bridged links over dropped temporary tables.

## How to run

```
# once: clone this repo, clone dudenzz/lineage inside it, create .venv, pip install -r requirements.txt (README)
cd notebooks
jupyter notebook          # run 03 -> 04 -> 05 in this order
```

`05_second_iteration.ipynb` needs the cached column scores for three table-name variants. If
`results/column_scores_*.csv.gz` are missing, run once (about 2 minutes each):

```
cd src
python matching.py --names raw
python matching.py --names neutral
python matching.py --names base
```

On Windows PowerShell without activating the environment: `..\.venv\Scripts\python.exe matching.py --names raw`.

## Building the report

The report is LaTeX: `docs/report/report.tex`, bibliography `docs/report/refs.bib`, pipeline figure
`docs/report/pipeline.tex`, other figures from `docs/figures/`.

* **Overleaf (easiest):** upload `docs/report/` and `docs/figures/` keeping the folder structure (`report.tex`
  uses `\graphicspath{{../figures/}{./}}`), compile `pipeline.tex` once, then `report.tex` with pdfLaTeX + BibTeX.
* **Locally:** `cd docs/report`, then `pdflatex pipeline`, `pdflatex report`, `bibtex report`,
  `pdflatex report`, `pdflatex report`, and copy `report.pdf` to `docs/report_draft.pdf`.

## Note for the course document

The Milestone 1 entry for the Magneto paper links to the Valentine paper (IEEE 9458921). The correct reference is
PVLDB 18(8): 2681-2694, 2025, <https://doi.org/10.14778/3742728.3742757>.
