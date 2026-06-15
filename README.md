# Prompt Framing Determines Whether Frontier Language Models Discover an Algorithmic Reformulation in Scientific Code: The TRACE Prompt-Ablation Experiment

**Experiment ID:** TRACE_20260611

## Authors

Pablo Garcia¹, Roberto Duran-Fernandez¹,²,\*, and David Figueroa¹

¹ Inter-American Development Bank, Washington, DC, USA  
² Université Paris 1 Panthéon-Sorbonne, Paris, France  
\* Corresponding author: Roberto Duran-Fernandez — robertodu@iadb.org

## Abstract

Large language models are increasingly used to assist scientific programming, yet their ability to identify non-obvious algorithmic reformulations depends on how the computational problem is framed. This Letter reports TRACE (Testing Reasoning About Computational Efficiency), a 2,880-run prompt-ablation experiment testing whether prompt framing changes whether frontier models discover the transformation of repeated row-wise spatial-neighbor lookup into reusable sparse graph or adjacency aggregation. The experimental task came from an R spatial downscaling pipeline in which neighborhood features were required for 344,208 cells over 28 years and five predictor variables. The unoptimized formulation repeatedly rebuilt a time-invariant topology, yielding an 86.8-hour analytic runtime estimate and observed incomplete runs exceeding that estimate on local, cloud, and long-duration pre-optimization attempts. TRACE evaluated 12 prompt families, four temperature settings, 30 stochastic replicates per family-temperature-provider cell, and two response-model conditions: OpenAI GPT 5.0 and Anthropic Opus 4.6. Automated structured scoring produced 2,880 valid scored responses from 2,880 attempted runs. Strict discovery occurred in 2,014 attempted runs (69.9%). Success was zero when prompts exposed only the downstream Random Forest wrapper, but reached 99.2% under sparse-graph framing and 95.4% under explicit topology-invariance framing. These results support a bounded methodological claim: prompt-induced computational framing can determine whether frontier models move from local code advice to discovering reusable computational structure.

---

This repository contains the complete replication materials for the experiment. The experiment tests whether prompt framing determines whether frontier large language models (LLMs) discover a non-obvious algorithmic reformulation in scientific code.

---

## Research Question

Can prompt framing determine whether frontier LLMs discover the transformation of repeated row-wise spatial-neighbor lookup into reusable sparse graph / adjacency aggregation?

The target discovery requires recognising that:

- **Spatial neighbor topology is static across years** (time-invariant structure).
- **Yearly cell attributes are dynamic** (time-varying data).
- The correct reformulation separates these two concerns: build the neighbor graph once, then apply yearly attributes to it — rather than reconstructing cell-year string keys and neighbor lookups on every iteration.

---

## Headline Results

| Metric | Value |
|--------|-------|
| Attempted model-response observations | 2,880 |
| Valid scored responses | 2,880 |
| Strict discovery successes | 2,014 (69.9 %) |
| Mechanistic success (mechanism score ≥ 8) | 2,019 (70.1 %) |
| Mean mechanism score | 8.475 / 10 |

Provider breakdown:

| Provider / Model | Strict discoveries | Rate |
|------------------|--------------------|------|
| Anthropic Opus 4.6 | 1,274 / 1,440 | 88.5 % |
| OpenAI GPT 5.0 moderate | 740 / 1,440 | 51.4 % |

Temperature effect: chi-square p = 0.429; likelihood-ratio improvement p = 0.109. Manually selected Copilot Studio temperature settings did not produce a statistically reliable or practically large effect after accounting for prompt family and provider.

---

## Terminology Note

The database field `publication_grade_success` and the paper term **strict discovery** refer to the same outcome. In the database, the field is named `publication_grade_success`; throughout the paper it is called "strict discovery." The two are identical. The stricter internal field `discovery_success` (requiring mechanism score ≥ 9) is used as a secondary diagnostic and is also included in the database.

---

## Experimental Design

- **12 prompt families** (F01–F12), varying in code visibility, semantic framing, and diagnostic constraints.
- **4 Copilot Studio temperature-setting labels**: 0.0, 0.3, 0.7, 1.0 (see temperature note below).
- **30 replicates** per family × temperature-label × provider cell.
- **2 providers**: Anthropic Opus 4.6 and OpenAI GPT 5.0 moderate.
- **Automation layer**: Microsoft Copilot Studio.

### Prompt Family Ladder

| ID | Label | Key feature |
|----|-------|-------------|
| F01 | RF wrapper only | Upstream neighbor construction hidden; 0 strict discoveries |
| F02 | Code only, no hint | Neighbor code visible, neutral framing |
| F03 | Code + RF frame | Neighbor code visible, misleading Random Forest framing |
| F04 | Neighbor bottleneck hint | Explicit spatial-neighbor cue |
| F05 | String-key probe | Isolates repeated cell-year key lines |
| F06 | Raster/kernel analogy | Raster local-neighborhood bridge |
| F07 | Raster + irregular constraint | Raster analogy with explicit topology-safety warning |
| F08 | Topology invariance | States topology is static, attributes vary |
| F09 | Adjacency-table option | Suggests adjacency table as candidate representation |
| F10 | Sparse graph frame | Explicitly frames task as sparse graph aggregation |
| F11 | False RF diagnosis | Adversarial: false RF bottleneck + audit instruction |
| F12 | False rbind diagnosis | Adversarial: false rbind bottleneck + audit instruction |

### Temperature-Setting Disclosure

Temperature values were selected manually in the Copilot Studio UI before each batch run. No batch-level log file verifying that the selected values were applied as runtime parameters is available. Temperature labels are therefore treated as Copilot Studio setting identifiers, not as independently verified controlled experimental factors. All temperature-related results should be interpreted as descriptive summaries of repeated automated submissions, not as controlled temperature-response inference.

---

## Repository Contents

### `01_Experiment/` — Prompts and Raw LLM Outputs

- `Experiment_Prompt/` — Original `.md` prompt files, organised by provider (`Anthropic/`, `OpenAI/`) and temperature-setting label (`Temp_0.0/`, `Temp_0.3/`, `Temp_0.7/`, `Temp_1.0/`).
- `Experiment_Output/` — Raw LLM responses, organised the same way. Files carry an `output_` prefix.

Prompt filename pattern:

```
F01_Anthropic_temp_00_rep_001.md
F12_OpenAI_temp_10_rep_030.md
```

Output filename pattern:

```
output_F01_Anthropic_temp_00_rep_001.md
output_F12_OpenAI_temp_10_rep_030.md
```

Temperature code mapping:

| Copilot UI label | Filename code |
|-----------------|---------------|
| 0.0 | temp_00 |
| 0.3 | temp_03 |
| 0.7 | temp_07 |
| 1.0 | temp_10 |

### `02_Scoring/` — Scoring Prompts and Outputs

- `Score_Input/` — 2,880 scoring prompts, one per raw model response, stored flat for a single Copilot Studio scoring batch. Filename pattern: `S__Anthropic_Opus46__F01__temp_00__rep_001.md`.
- `Score_Output_Normalized/` — Validated `.json` copies of all 2,880 scoring outputs (raw outputs from Copilot had `.md` extensions; these are the parsed and validated JSON equivalents used for database construction).

No scoring rerun was required: all 2,880 scoring outputs parsed as valid JSON on the first pass. Derived fields (`mechanism_score`, `discovery_success`, `publication_grade_success`) were recomputed deterministically from component scores during database construction.

### `03_ Metadata/` — Manifests and Methodology

- `experiment_manifest.csv` — One row per original prompt file; encodes provider, model, prompt family, temperature label, replicate, and file path.
- `prompt_family_manifest.csv` — One row per prompt family; short description of each family's design rationale.
- `score_input_manifest.csv` — One row per scoring prompt, linking it to its raw response file and experiment-manifest row.
- `score_output_validation_audit.csv` — Parse and schema audit for every raw scoring output.
- `missing_experiment_outputs.csv` — Any prompt-manifest rows without a matching output file (empty if all outputs are present).
- `score_rerun_manifest.csv` — Scoring rerun tracking (empty; no rerun was required).
- `Runs.txt` — Structured run-stage metadata record.
- `SCORING_METHODOLOGY.md` — Full scoring rubric, component definitions, derived-outcome rules, and output discipline requirements for the scoring model.

### `04_Scripts/` — Reproducibility

- `generate_trace_20260611_experiment.ps1` — PowerShell script that generates all prompt files in `01_Experiment/Experiment_Prompt/` from the family templates and manifest.

### `05_Dbase/` — Final Database

- `DataBase.xlsx` — Excel workbook containing the full scored database, summary tables by prompt family, provider, and temperature label, and supporting worksheets for statistical tests and run metadata. This is the primary file for replicating the reported results.

### `06_Original Context Windows/` — Discovery History

This folder documents the real human–LLM interaction sequence that motivated the paper. It is not part of the controlled experiment; it is the historical record of how the algorithmic discovery was first made and then lost and recovered across different models and sessions.

```
01 Initial Algorithm/
    Cell_RF_imputation.R              — original R script (the code the experiment is based on)
    Cell_RF_imputation_optimized.R    — an earlier partial optimisation attempt

02 First Claude Optimization/
    Cell_RF_imputation.R
    Cell_RF_imputation_optimized.R
    Original Claude Optimization.txt  — Claude's first optimization session transcript

03 Chat GPT Discovery/
    ChatGPT.txt                       — ChatGPT session that first produced the sparse-graph discovery

04 Prompt From GTP to Claude/
    prompt_ganador.txt                — the prompt that transferred the discovery framing to Claude

05 Final Claude Discovery and Succces/
    interation 1 / cell_imputation_model_5_local_v3.R
    interation 1 / validate_and_benchmark_v2_vs_v3.R
    Interation 2 / ...
    Interation 3 / ...
    Claude.txt                        — Claude session transcript achieving the full correct solution
```

The sequence shows: original code → initial Claude optimisation attempt (partial) → ChatGPT independently discovers the sparse-graph reformulation → that framing is transferred back to Claude → Claude produces the full correct implementation. This chain is the empirical origin of the F09/F10/F11 prompt-family designs and the paper's central hypothesis.

---

## Scoring Rubric Summary

Each model response is evaluated on five **mechanism components** (summed to `mechanism_score`, 0–10) and four **diagnostic dimensions**:

| Component | Max | What earns full credit |
|-----------|-----|------------------------|
| `bottleneck_identification` | 2 | Identifies row-wise neighbor lookup or cell-year string-key construction as the bottleneck |
| `topology_invariance` | 2 | Explicitly separates static neighbor topology from dynamic yearly attributes |
| `solution_architecture` | 2 | Proposes a reusable adjacency table, edge list, sparse graph, or spatial weights structure |
| `yearly_attribute_application` | 2 | Applies per-year or per-variable values through the fixed topology |
| `numerical_equivalence` | 2 | Preserves same rook-neighbor definition, same-year statistics, NA handling, and max/min/mean estimands |

**Strict discovery** (`publication_grade_success` = 1 in the database) requires: mechanism score ≥ 8, `solution_architecture` = 2, `topology_invariance` = 2, and `numerical_equivalence` ≥ 1.

The stricter field `discovery_success` additionally requires mechanism score ≥ 9 and `rf_handling` ≥ 1.

The diagnostic dimensions (`raster_handling`, `rf_handling`, `implementation_quality`, `resists_false_framing`) are retained for quality control and are not included in the mechanism score. Full rubric: `03_ Metadata/SCORING_METHODOLOGY.md`.

---

## File and Size Summary

| Folder | Files | Size |
|--------|-------|------|
| `01_Experiment/` | 5,761 | 25.8 MB |
| `02_Scoring/` | 5,760 | 37.4 MB |
| `03_ Metadata/` | 8 | 3.7 MB |
| `04_Scripts/` | 1 | < 0.1 MB |
| `05_Dbase/` | 1 | 0.9 MB |
| `06_Original Context Windows/` | 14 | 0.2 MB |
| **Total** | **11,546** | **~68 MB** |

All individual files are well under GitHub's 100 MB file limit. Total repository size (~68 MB) is within GitHub's standard limits. Directories with many files (Score_Input, Score_Output_Normalized, Experiment_Output, Experiment_Prompt sub-folders) will be slow to browse in the GitHub web UI but are fully functional via `git clone`.

---

## Citation

Garcia, P., Duran-Fernandez, R., and Figueroa, D. (2026). Prompt Framing Determines Whether Frontier Language Models Discover an Algorithmic Reformulation in Scientific Code: The TRACE Prompt-Ablation Experiment. *AI Letters*. *(Citation to be completed upon publication.)*

---

## License

*License to be confirmed by the authors before public release. Review all files for restricted paths, confidential institutional material, or non-shareable source data before publishing.*
