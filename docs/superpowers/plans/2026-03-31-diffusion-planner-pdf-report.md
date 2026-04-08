# Diffusion-Planner PDF Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Chinese technical-report-style PDF that explains the Diffusion-Planner paper, maps it to the current repository, and clearly presents the user-added MPC extension and its optimization formulation.

**Architecture:** Extract implementation-critical facts from the paper PDF and local repository, write a structured Chinese report in LaTeX, compile it with XeLaTeX for reliable Chinese and math rendering, then visually inspect the rendered PDF and fix layout issues. The report will separate paper-original content from repository-confirmed content and from the user-added MPC extension.

**Tech Stack:** `pdftotext`, local repository source files, LaTeX/XeLaTeX, Poppler `pdftoppm`

---

### Task 1: Collect Paper And Code Evidence

**Files:**
- Modify: `docs/superpowers/plans/2026-03-31-diffusion-planner-pdf-report.md`
- Read: `Zheng 等 - 2024 - Diffusion-based planning for autonomous driving with flexible guidance.pdf`
- Read: `train_predictor.py`
- Read: `diffusion_planner/**/*.py`

- [ ] **Step 1: Extract paper text for method and appendix sections**

Run: `pdftotext 'Zheng 等 - 2024 - Diffusion-based planning for autonomous driving with flexible guidance.pdf' -`
Expected: full paper text, including method, appendix hyperparameters, and guidance equations.

- [ ] **Step 2: Re-read code paths for training, inference, guidance, and MPC**

Run: `sed -n` / `nl -ba` on:
- `train_predictor.py`
- `diffusion_planner/train_epoch.py`
- `diffusion_planner/loss.py`
- `diffusion_planner/model/module/encoder.py`
- `diffusion_planner/model/module/decoder.py`
- `diffusion_planner/planner/planner.py`
- `diffusion_planner/optimization/mpc_refiner.py`

Expected: verified call graph and line-level references.

- [ ] **Step 3: Build a fact split**

Create three buckets for the report:
- paper-stated
- code-confirmed
- implementation inference

Expected: no MPC detail is accidentally attributed to the paper.

### Task 2: Draft Report Content

**Files:**
- Create: `output/pdf/diffusion_planner_technical_report_zh.tex`

- [ ] **Step 1: Write report skeleton**

Include sections:
- Abstract
- Problem Setting
- Paper Method
- Paper-to-Code Mapping
- Training And Inference Pipeline
- Guidance Mechanism
- User-Added MPC Extension
- MPC Optimization Problem
- Diffusion-MPC Coupling
- Differences From Original Paper
- Reproduction Risks
- Conclusion

- [ ] **Step 2: Fill paper sections with implementation-oriented detail**

Include:
- diffusion target
- architecture blocks
- losses
- training details
- inference details
- guidance energies

- [ ] **Step 3: Fill code mapping and MPC sections**

Include:
- file-level mapping
- actual tensor and mask semantics
- candidate sampling logic
- MPC variables, dynamics, constraints, cost terms, acceptance logic
- runtime writeback path into `outputs["prediction"]`

- [ ] **Step 4: Mark evidence provenance in prose**

Expected: phrases distinguish “论文明确写出”, “代码实现确认”, and “根据实现推断”.

### Task 3: Compile PDF

**Files:**
- Modify: `output/pdf/diffusion_planner_technical_report_zh.tex`
- Create: `output/pdf/diffusion_planner_technical_report_zh.pdf`

- [ ] **Step 1: Compile with XeLaTeX**

Run: `cd /data/wyf/lgq/Diffusion-Planner/output/pdf && xelatex -interaction=nonstopmode diffusion_planner_technical_report_zh.tex`
Expected: PDF generated without fatal errors.

- [ ] **Step 2: Re-compile once more for stable references**

Run: same command again
Expected: clean table of contents, references, and page numbering.

### Task 4: Visual Verification

**Files:**
- Read: `output/pdf/diffusion_planner_technical_report_zh.pdf`

- [ ] **Step 1: Render PDF pages to images**

Run: `mkdir -p /data/wyf/lgq/Diffusion-Planner/tmp/pdfs && pdftoppm -png /data/wyf/lgq/Diffusion-Planner/output/pdf/diffusion_planner_technical_report_zh.pdf /data/wyf/lgq/Diffusion-Planner/tmp/pdfs/dp_report`
Expected: one PNG per page.

- [ ] **Step 2: Inspect page count and spot-check rendering**

Check:
- Chinese glyphs render correctly
- formulas are legible
- headings and spacing are consistent
- long code paths do not overflow margins

- [ ] **Step 3: Fix layout if needed and recompile**

Expected: final PDF is readable and stable.

### Task 5: Deliver

**Files:**
- Read: `output/pdf/diffusion_planner_technical_report_zh.pdf`

- [ ] **Step 1: Confirm final artifact paths**

Provide:
- `.tex` source path
- `.pdf` path

- [ ] **Step 2: Summarize scope and caveats**

Mention:
- MPC is user-added, not paper-original
- report distinguishes paper facts from repo facts
- report was compiled and visually checked
