# Q1 Assistance Log

Record every instance of outside help used for this quarter's work: AI tools,
tutorials, documentation examples, forum answers, or help from other people.
Add entries as you go, newest at the bottom.

## How to log an entry

| Field | What to write |
|---|---|
| **Date** | YYYY-MM-DD |
| **Source** | Tool or person (e.g. ChatGPT, Claude, Stack Overflow link, a classmate) |
| **Task** | Which assignment, notebook, or file |
| **What I asked / used** | The question, prompt summary, or resource consulted |
| **What I used from it** | Code, explanation, debugging help, wording, etc. |
| **What I changed or verified** | How I checked, tested, or modified it |
| **Files affected** | Paths in this repository |

## Entries

### 2026-09-14 · Repository and environment setup

- **Source:** Claude Code (Anthropic), AI coding assistant
- **Task:** Initial repository scaffolding and Python environment setup (no assignment work)
- **What I asked / used:** Create the folder structure, READMEs, `.gitignore`,
  Week 1 notebook skeleton, and a verified PyTorch/Jupyter environment.
- **What I used from it:** Repository layout, documentation templates,
  `environment/verify_environment.py`, `requirements.txt`, and the empty
  section outline in `notebooks/week01_fashion_mnist_baseline.ipynb`.
- **What I changed or verified:** Environment checks passed (see
  `environment/system_info.txt`); JupyterLab launched. The Week 1 assignment
  itself was not implemented.
- **Files affected:** entire initial commit

### 2026-10-28 · Dropout experiment analysis and v2 redesign

- **Source:** Claude Code (Anthropic), AI coding assistant
- **Task:** Q1 dropout research (`week5_Q1_dropout_research.ipynb`, Week 4-5 manuscripts)
- **What I asked / used:** Asked for an analysis of the Week 5 dropout results and
  a stronger experimental design, then for the code to implement it.
- **What I used from it:** Diagnosis that the v1 models were undertrained (validation
  loss still falling at epoch 10, best epoch 9-10, small train/val gap), so v1 becomes
  the pilot study. The v2 design (early stopping, wider MLP, dropout 0/0.2/0.5,
  4 training sizes, 10 seeds, per-dropout learning-rate pilot, official test set used
  once, paired tests with Holm correction) and the notebook
  `notebooks/week6_Q1_dropout_v2.ipynb` that implements it.
- **What I changed or verified:** The assistant ran a short smoke test (6 epochs,
  outputs written outside the repo and discarded) to confirm every cell runs.
  _TODO: add what I reviewed, changed, and checked myself._
- **Files affected:** `notebooks/week6_Q1_dropout_v2.ipynb`, this log

### 2026-10-28 · Methods section draft

- **Source:** Claude Code (Anthropic), AI coding assistant
- **Task:** NHSJS paper, Methodology section
- **What I asked / used:** Asked for a draft of the Methods section following the
  NHSJS headings, based on the v2 protocol, code, and results.
- **What I used from it:** `manuscript/Week6MethodsDraft.md` (first draft, all numbers
  taken from the repository files).
- **What I changed or verified:** _TODO: describe my revisions and checks._
- **Files affected:** `manuscript/Week6MethodsDraft.md`, this log

### 2026-10-28 · Exploratory analysis cell and Results draft

- **Source:** Claude Code (Anthropic), AI coding assistant
- **Task:** NHSJS paper, Results section; v2 notebook
- **What I asked / used:** Asked for the exploratory overfitting analysis to be added to
  the notebook and for a draft of the Results section.
- **What I used from it:** Section 7 of `notebooks/week6_Q1_dropout_v2.ipynb` (exploratory,
  not pre-registered: gap at epoch 20 and validation-loss rise after the best epoch; reads
  saved histories only), `results/raw/v2/exploratory_overfitting_tests.csv`,
  `figures/q1v2_exploratory_overfitting.png`, and `manuscript/Week6ResultsDraft.md`.
- **What I changed or verified:** _TODO: describe my revisions and checks._
- **Files affected:** files listed above, `manuscript/Week6MethodsDraft.md` (epoch 20 fix), this log

### 2026-10-29 · Discussion draft

- **Source:** Claude Code (Anthropic), AI coding assistant
- **Task:** NHSJS paper, Discussion section
- **What I asked / used:** Asked for a draft of the Discussion based on the v2 results.
- **What I used from it:** `manuscript/Week6DiscussionDraft.md`
- **What I changed or verified:** _TODO: describe my revisions; verify the Srivastava et al.
  dataset-size claims in 4.3 against the paper._
- **Files affected:** `manuscript/Week6DiscussionDraft.md`, this log

### 2026-10-29 · Conclusion draft

- **Source:** Claude Code (Anthropic), AI coding assistant
- **Task:** NHSJS paper, Conclusion section
- **What I asked / used:** Asked for a draft of the Conclusion following the NHSJS parts.
- **What I used from it:** `manuscript/Week6ConclusionDraft.md`
- **What I changed or verified:** _TODO: describe my revisions and checks._
- **Files affected:** `manuscript/Week6ConclusionDraft.md`, this log

### 2026-10-29 · Introduction draft

- **Source:** Claude Code (Anthropic), AI coding assistant
- **Task:** NHSJS paper, Introduction section
- **What I asked / used:** Asked for a draft of the Introduction following the NHSJS parts,
  based on my Week 4 research proposal.
- **What I used from it:** `manuscript/Week6IntroductionDraft.md` (hypotheses copied
  unchanged from the frozen protocol)
- **What I changed or verified:** _TODO: describe my revisions; verify citation claims._
- **Files affected:** `manuscript/Week6IntroductionDraft.md`, this log

### 2026-10-29 · Abstract draft

- **Source:** Claude Code (Anthropic), AI coding assistant
- **Task:** NHSJS paper, Abstract
- **What I asked / used:** Asked for a 200-250 word abstract in the NHSJS format.
- **What I used from it:** `manuscript/Week6AbstractDraft.md`
- **What I changed or verified:** _TODO: describe my revisions and checks._
- **Files affected:** `manuscript/Week6AbstractDraft.md`, this log

### 2026-10-29 · Literature review, citations, and references

- **Source:** Claude Code (Anthropic), AI coding assistant
- **Task:** NHSJS paper: literature review, in-text citations, reference list, abstract format
- **What I asked / used:** Asked for a literature review section and citations throughout
  the paper.
- **What I used from it:** Literature review (Introduction 1.2), numbered superscript
  citations in the Introduction, Methods, Discussion, and Conclusion drafts,
  `manuscript/Week6References.md` (20 references; each was checked by the assistant
  through web searches, and Srivastava et al. Section 7.4 was read directly),
  `manuscript/Week6AcknowledgmentsDraft.md`, and the abstract reformatted to one paragraph.
  "Ahmad et al. (2026)" from my proposal could not be found and was replaced with Hobbs &
  Shah (2026).
- **What I changed or verified:** _TODO: open and check each reference myself._
- **Files affected:** `manuscript/Week6*.md`, this log

### 2026-10-29 · Title, NHSJS formatting, and PDF

- **Source:** Claude Code (Anthropic), AI coding assistant
- **Task:** NHSJS paper: title, full manuscript assembly, PDF
- **What I asked / used:** Asked for a title, the paper put into NHSJS format, and one PDF
  with the tables included.
- **What I used from it:** Title; `manuscript/NHSJS_Manuscript_Standard.md` (master file in
  NHSJS order, Conclusion merged into the end of the Discussion as 4.7-4.10, NHSJS figure and
  table captions); `.html`; and the PDF named after the title (18 pages, letter size, 12 pt,
  single spacing, 6 tables and 4 figures).
- **What I changed or verified:** _TODO: fill in author block, advisor name, and my revisions._
- **Files affected:** `manuscript/NHSJS_Manuscript_Standard.*`, the PDF, this log

## 2026-10-29 — Manuscript rewritten into third person / passive voice
- Tool: Claude Code (Opus 5.5)
- What it did: rewrote 61 first-person sentences ("I", "my", "we") in `Q1/manuscript/NHSJS_Manuscript_Standard.md` into third person or passive voice (Introduction, Methods, Discussion, Acknowledgments), after reviewer feedback; rebuilt the PDF. Backup of the previous version kept in the session scratchpad.
- TODO (student): reread the changed sentences and make sure the wording still sounds like you.

## 2026-10-29 — Rewrote AI-sounding passages
- Tool: Claude Code (Opus 5.5), using the stop-slop checklist
- What it did: rewrote 15 paragraphs (1.3, 1.4, Discussion 4.1–4.5, 4.7, 4.8, 4.10) plus one sentence in 1.1 and the last two sentences of the Abstract, removing stock phrases ("In other words", "Even so", "The most important lesson"), "not X but Y" contrasts, and quotable closing lines. Kept third person / passive, all numbers, and citations. Rebuilt the PDF.
- Note: this is still AI-written text. TODO (student): reword these paragraphs in your own voice.

## 2026-10-29 — Rewrote manuscript prose in the student's own writing style
- Tool: Claude Code (Opus 5.5)
- What it did: built a style profile from the student's earlier writing (UNSAT-core research summary and paper, AP Chemistry lab answers, lab report, essays), the student approved two sample paragraphs, then 49 prose paragraphs of `NHSJS_Manuscript_Standard.md` were rewritten in that style (Abstract, Introduction, explanation parts of Methods and Results, all of the Discussion). Kept third person / passive, all numbers, tables, and citation order. Rebuilt the PDF (19 pages). Backup of the previous version kept in the session scratchpad.
- Note: the text is still AI-generated, imitating the student's style. The AI disclosure in the Acknowledgments still applies.
- TODO (student): read the full paper and reword anything that doesn't sound like you.
