Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Join-Path (Get-Location) "TRACE_20260611"
$experimentPromptRoot = Join-Path $root "Experiment_Prompt"
$experimentOutputRoot = Join-Path $root "Experiment_Output"
$scoreInputRoot = Join-Path $root "Score_Input"
$scoreOutputRoot = Join-Path $root "Score_Output"

$providers = @(
  [PSCustomObject]@{ Code = "OpenAI"; Model = "GPT 5 Reasoning" },
  [PSCustomObject]@{ Code = "Anthropic"; Model = "Opus 4.6" }
)

$temperatures = @(
  [PSCustomObject]@{ Value = "0.0"; Code = "00"; Folder = "Temp_0.0" },
  [PSCustomObject]@{ Value = "0.3"; Code = "03"; Folder = "Temp_0.3" },
  [PSCustomObject]@{ Value = "0.7"; Code = "07"; Folder = "Temp_0.7" },
  [PSCustomObject]@{ Value = "1.0"; Code = "10"; Folder = "Temp_1.0" }
)

$commonFacts = @'
Dataset and pipeline facts:
- 344,208 spatial grid cells.
- 28 years of panel data, 1992-2019.
- About 6.46 million cell-year rows.
- About 110 predictor variables.
- 5 neighbor source variables: ntl, ec, pop_density, def, usd_est_n2.
- About 1,373,394 directed rook-neighbor relationships.
- rook_neighbors_unique is a precomputed spdep::nb object serialized to disk.
- The Random Forest model is already trained and must not be retrained.
- Machine: standard laptop with 16 GB RAM.
- Current implementation has been estimated at 86+ hours.

In your answer, provide a diagnosis, an optimization strategy, and working R code. Preserve the trained Random Forest model and preserve the original numerical estimand.
'@

$rfWrapperCode = @'
```r
library(blockCV)
library(zoo)
library(LongituRF)
library(randomForest)
library(sf)
library(spdep)
library(tidyverse)
library(data.table)
library(terra)
library(plm)
library(utils)
library(fixest)
library(scales)
library(stringi)

prep_data <- st_read('/Volumes/Toshi 1Tb/Amaz/geographic_cell_data/geographic_cell_data.shp')

load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

pred_db$consolidated <- NA

for (year in unique(pred_db$year)) {
  cat(paste0("Predicting for year ", year, "\n"))

  test_set <- joined_data %>% filter(year == year)

  if (as.character(year) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(year)]]
    pred_db$consolidated[pred_db$year == year] <- predict(rf_model, newdata = test_set)
  } else {
    cat(paste0("Warning: No model found for year ", year, "\n"))
  }
}

write.csv(pred_db, "RF_imputated_db.csv")
```
'@

$neighborCode = @'
build_neighbor_lookup:

```r
build_neighbor_lookup <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  idx_lookup <- setNames(
    seq_len(nrow(data)),
    paste(data$id, data$year, sep = "_")
  )
  row_ids <- seq_len(nrow(data))

  lapply(row_ids, function(i) {
    ref_idx           <- id_to_ref[as.character(data$id[i])]
    neighbor_cell_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_keys     <- paste(neighbor_cell_ids, data$year[i], sep = "_")
    result            <- idx_lookup[neighbor_keys]
    as.integer(result[!is.na(result)])
  })
}
```

compute_neighbor_stats:

```r
compute_neighbor_stats <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  result <- lapply(neighbor_lookup, function(idx) {
    if (length(idx) == 0) return(c(NA, NA, NA))
    neighbor_vals <- vals[idx]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) return(c(NA, NA, NA))
    c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  })
  do.call(rbind, result)
}
```

Outer loop:

```r
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
}
```
'@

$rfWrapperCodeForPrompt = $rfWrapperCode.Replace('$', '__DOLLAR__')
$neighborCodeForPrompt = $neighborCode.Replace('$', '__DOLLAR__')

$families = @(
  [PSCustomObject]@{
    Id = "F01"; Slug = "RF_WRAPPER_ONLY"; Group = "Missing upstream context"; Label = "RF wrapper only";
    Description = "Only the downstream Random Forest prediction wrapper is visible; upstream neighbor-feature construction is hidden.";
    Prompt = @"
I am working with this R script for cell-level GDP prediction. The process is too slow or too memory intensive on a personal computer. Please review the code and propose a practical optimization strategy. Preserve the trained Random Forest models and do not retrain them.

$rfWrapperCodeForPrompt

Context:
- The data contain hundreds of thousands of cells per year and many predictor variables.
- The model is already trained.
- The goal is to make the process computationally feasible on a normal machine.

Provide a diagnosis, an optimization strategy, and working R code.
"@
  },
  [PSCustomObject]@{
    Id = "F02"; Slug = "CODE_ONLY_NO_HINT"; Group = "Code-visible neutral"; Label = "Code only, no hint";
    Description = "The neighbor-construction code is visible, with no semantic hint about topology, adjacency, graph, raster, or static structure.";
    Prompt = @"
I am working with this R code that prepares features for a cell-level GDP prediction pipeline. The process is too slow or too memory intensive on a personal computer. Please review the code and propose a practical optimization strategy.

$neighborCodeForPrompt

$commonFacts
"@
  },
  [PSCustomObject]@{
    Id = "F03"; Slug = "CODE_PLUS_RF_FRAME"; Group = "RF-frame with code"; Label = "Code plus RF frame";
    Description = "The neighbor code is visible, but the prompt strongly frames the task around Random Forest prediction, object loading, memory, and prediction-loop efficiency.";
    Prompt = @"
I am optimizing a cell-level GDP prediction pipeline. The main performance problem is likely in preparing data for repeated Random Forest prediction, including model loading, prediction-loop efficiency, memory use, and object copying. Focus your diagnosis first on Random Forest inference and the prediction workflow.

Here is the feature-preparation code that runs before prediction:

$neighborCodeForPrompt

$commonFacts
"@
  },
  [PSCustomObject]@{
    Id = "F04"; Slug = "NEIGHBOR_BOTTLENECK_HINT"; Group = "Spatial-neighbor cue"; Label = "Neighbor bottleneck hint";
    Description = "The prompt states that spatial neighbor feature construction may be the bottleneck without giving the final representation.";
    Prompt = @"
I suspect that spatial neighbor feature construction, rather than Random Forest inference itself, may be the computational bottleneck in this cell-level GDP prediction pipeline. Please audit the code and propose a practical optimization strategy.

$neighborCodeForPrompt

$commonFacts
"@
  },
  [PSCustomObject]@{
    Id = "F05"; Slug = "STRING_KEY_REDUNDANCY"; Group = "Line-level probe"; Label = "String-key probe";
    Description = "The prompt isolates repeated cell-year string-key construction and asks whether it indicates a larger repeated lookup pattern.";
    Prompt = @"
The following lines appear to repeat many times in a large cell-year panel:

```r
idx_lookup <- setNames(
  seq_len(nrow(data)),
  paste(data__DOLLAR__id, data__DOLLAR__year, sep = "_")
)

neighbor_keys <- paste(neighbor_cell_ids, data__DOLLAR__year[i], sep = "_")
result <- idx_lookup[neighbor_keys]
```

These lines are part of this feature-construction code:

$neighborCodeForPrompt

Please diagnose whether the repeated string-key work is only a local inefficiency or a symptom of a larger repeated lookup pattern. If a broader algorithmic reformulation is possible, propose it with working R code.

$commonFacts
"@
  },
  [PSCustomObject]@{
    Id = "F06"; Slug = "RASTER_KERNEL_ANALOGY"; Group = "Raster/kernel bridge"; Label = "Raster/kernel analogy";
    Description = "The prompt introduces raster focal/kernel operations as an analogy without saying whether raster focal operations are valid.";
    Prompt = @"
This cell-level panel computes max, min, and mean values among rook neighbors before applying a pre-trained Random Forest model. Consider whether raster focal or kernel operations offer a useful analogy for making the computation faster, but choose the implementation that best preserves the required results.

$neighborCodeForPrompt

$commonFacts
"@
  },
  [PSCustomObject]@{
    Id = "F07"; Slug = "RASTER_INVALID_IRREGULAR"; Group = "Raster/kernel bridge"; Label = "Raster invalid irregular topology";
    Description = "The prompt introduces the raster analogy but states that naive raster focal operations may be unsafe because topology is irregular or masked.";
    Prompt = @"
This cell-level panel resembles a raster-neighborhood problem, but naive raster focal operations may be unsafe because the cell topology can be irregular, masked, or otherwise not equivalent to a complete rectangular raster. Find an exact representation that preserves the original rook-neighbor relationships and computes neighbor max, min, and mean efficiently.

$neighborCodeForPrompt

$commonFacts
"@
  },
  [PSCustomObject]@{
    Id = "F08"; Slug = "TOPOLOGY_INVARIANCE"; Group = "Topology cue"; Label = "Topology invariance";
    Description = "The prompt states that neighbor relationships do not change across years while variables attached to cells do change by year, without naming a concrete representation.";
    Prompt = @"
In this pipeline, the neighbor relationship among cells does not change across years, while variables attached to cells do change by year. Please use that static-versus-changing distinction to redesign the computation of neighbor max, min, and mean before the pre-trained Random Forest prediction step.

$neighborCodeForPrompt

$commonFacts
"@
  },
  [PSCustomObject]@{
    Id = "F09"; Slug = "ADJACENCY_TABLE_OPTION"; Group = "Representation cue"; Label = "Adjacency-table option";
    Description = "The prompt suggests a cell-neighbor table or adjacency table as one possible representation, without explicit graph terminology.";
    Prompt = @"
Consider whether this pipeline can be made faster by building a reusable cell-neighbor table or adjacency table once, then joining yearly cell attributes onto that table to compute neighbor max, min, and mean before Random Forest prediction.

$neighborCodeForPrompt

$commonFacts
"@
  },
  [PSCustomObject]@{
    Id = "F10"; Slug = "SPARSE_GRAPH_FRAME"; Group = "Graph cue"; Label = "Sparse graph frame";
    Description = "The prompt explicitly frames the task as sparse graph neighborhood aggregation.";
    Prompt = @"
Reinterpret this pipeline as sparse graph neighborhood aggregation. The cells are nodes, rook neighbor relationships are directed edges, and yearly variables are node attributes. The task is to compute max, min, and mean of neighbor attributes for each node-year before applying a pre-trained Random Forest model.

$neighborCodeForPrompt

$commonFacts

Design the most computationally efficient implementation in R. Build the graph topology once, reuse it across years, and preserve numerical equivalence with the original neighbor statistics.
"@
  },
  [PSCustomObject]@{
    Id = "F11"; Slug = "FALSE_RF_DIAGNOSIS"; Group = "Adversarial diagnosis"; Label = "False RF diagnosis";
    Description = "The prompt gives a false Random Forest bottleneck diagnosis and asks the model to audit and reject it if unsupported.";
    Prompt = @"
A colleague claims the main bottleneck in this pipeline is Random Forest inference: loading models, calling predict(), and writing the final predictions. Audit that claim against the code below. If the code evidence points to a different bottleneck, reject the colleague's diagnosis and propose the correct optimization.

$neighborCodeForPrompt

$commonFacts
"@
  },
  [PSCustomObject]@{
    Id = "F12"; Slug = "FALSE_RBIND_DIAGNOSIS"; Group = "Adversarial diagnosis"; Label = "False rbind diagnosis";
    Description = "The prompt gives a false do.call/rbind diagnosis and asks the model to audit and reject it if unsupported.";
    Prompt = @"
A colleague claims the main bottleneck in this pipeline is do.call(rbind, result) and repeated list binding inside compute_neighbor_stats(). Audit that claim against the code below. If the code evidence points to a deeper bottleneck, reject the colleague's diagnosis and propose the correct optimization.

$neighborCodeForPrompt

$commonFacts
"@
  }
)

function New-TextFile {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$Content
  )
  $directory = Split-Path -Parent $Path
  if ($directory) {
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
  }
  Set-Content -LiteralPath $Path -Value $Content -Encoding UTF8
}

New-Item -ItemType Directory -Force -Path $root, $experimentPromptRoot, $experimentOutputRoot, $scoreInputRoot, $scoreOutputRoot | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $root "scripts") | Out-Null

$manifestRows = New-Object System.Collections.Generic.List[object]

foreach ($provider in $providers) {
  foreach ($temperature in $temperatures) {
    $promptDir = Join-Path (Join-Path $experimentPromptRoot $provider.Code) $temperature.Folder
    $outputDir = Join-Path (Join-Path $experimentOutputRoot $provider.Code) $temperature.Folder
    $scoreInputDir = Join-Path (Join-Path $scoreInputRoot $provider.Code) $temperature.Folder
    $scoreOutputDir = Join-Path (Join-Path $scoreOutputRoot $provider.Code) $temperature.Folder
    New-Item -ItemType Directory -Force -Path $promptDir, $outputDir, $scoreInputDir, $scoreOutputDir | Out-Null

    foreach ($family in $families) {
      foreach ($replicate in 1..30) {
        $repCode = "{0:D3}" -f $replicate
        $promptFileName = "{0}_{1}_temp_{2}_rep_{3}.md" -f $family.Id, $provider.Code, $temperature.Code, $repCode
        $expectedOutputFileName = "output_{0}" -f $promptFileName
        $runId = "{0}_{1}_T{2}_R{3}" -f $provider.Code, $family.Id, $temperature.Code, $repCode
        $promptContent = $family.Prompt.Trim().Replace('__DOLLAR__', '$')
        New-TextFile -Path (Join-Path $promptDir $promptFileName) -Content $promptContent

        $manifestRows.Add([PSCustomObject]@{
          experiment_id = "TRACE_20260611"
          run_id = $runId
          provider = $provider.Code
          model_label = $provider.Model
          temperature = $temperature.Value
          temperature_code = $temperature.Code
          prompt_family_id = $family.Id
          prompt_family_slug = $family.Slug
          family_label = $family.Label
          family_group = $family.Group
          replicate = $replicate
          prompt_file = ("Experiment_Prompt/{0}/{1}/{2}" -f $provider.Code, $temperature.Folder, $promptFileName)
          expected_output_file = ("Experiment_Output/{0}/{1}/{2}" -f $provider.Code, $temperature.Folder, $expectedOutputFileName)
          expected_score_input_file = ("Score_Input/{0}/{1}/S__{2}.md" -f $provider.Code, $temperature.Folder, ($promptFileName -replace "\.md$", ""))
          expected_score_output_file = ("Score_Output/{0}/{1}/output_S__{2}.json" -f $provider.Code, $temperature.Folder, ($promptFileName -replace "\.md$", ""))
          description = $family.Description
        })
      }
    }
  }
}

$manifestRows | Export-Csv -LiteralPath (Join-Path $root "experiment_manifest.csv") -NoTypeInformation -Encoding UTF8

$familyRows = foreach ($family in $families) {
  [PSCustomObject]@{
    prompt_family_id = $family.Id
    prompt_family_slug = $family.Slug
    family_label = $family.Label
    family_group = $family.Group
    description = $family.Description
  }
}
$familyRows | Export-Csv -LiteralPath (Join-Path $root "prompt_family_manifest.csv") -NoTypeInformation -Encoding UTF8

$readme = @'
# TRACE 20260611 Prompt-Ablation Experiment

This folder contains a repository-ready replication and refinement of the original prompt-ablation experiment. The experiment tests whether prompt framing affects whether frontier LLMs discover the target optimization: replacing repeated row-wise spatial-neighbor lookup with reusable neighborhood aggregation.

## Design

- Experiment ID: TRACE_20260611
- Providers/models:
  - OpenAI: GPT 5 Reasoning
  - Anthropic: Opus 4.6
- Temperatures: 0.0, 0.3, 0.7, 1.0
- Prompt families: 12
- Repetitions per provider-temperature-family cell: 30
- Total original prompts: 2,880

Temperature is intended to be set manually in Copilot Studio for each provider-temperature batch. No batch-log file is included by design.

## Folder Map

- `Experiment_Prompt/`: original prompts to send through Copilot Studio, organized by provider and temperature.
- `Experiment_Output/`: place raw LLM outputs here, preserving filenames with an `output_` prefix.
- `Score_Input/`: generated scoring prompts should be placed here after raw outputs are copied in.
- `Score_Output/`: JSON scoring outputs should be placed here after scoring.
- `experiment_manifest.csv`: one row per original prompt file.
- `prompt_family_manifest.csv`: descriptions of the 12 prompt families.

## Filename Convention

Prompt files:

```text
F01_OpenAI_temp_00_rep_001.md
F12_Anthropic_temp_10_rep_020.md
```

Expected raw output files:

```text
output_F01_OpenAI_temp_00_rep_001.md
output_F12_Anthropic_temp_10_rep_020.md
```

Temperature code mapping:

```text
0.0 -> temp_00
0.3 -> temp_03
0.7 -> temp_07
1.0 -> temp_10
```

## Prompt-Family Ladder

- F01: no upstream information.
- F02: upstream code, neutral framing.
- F03: upstream code, misleading Random Forest framing.
- F04: neighbor bottleneck hint.
- F05: local string-key probe.
- F06: raster/kernel analogy.
- F07: raster analogy plus irregular-topology constraint.
- F08: static topology/dynamic attributes only.
- F09: adjacency-table representation.
- F10: sparse-graph representation.
- F11: false Random Forest diagnosis with audit instruction.
- F12: false rbind diagnosis with audit instruction.

## Scoring Target

The primary discovery outcome should require:

1. Identification of neighbor feature construction, not Random Forest inference, as the central bottleneck.
2. Recognition that neighbor topology is invariant across years.
3. Proposal of a reusable adjacency table, edge list, sparse graph, spatial weights structure, or equivalent fixed-neighborhood representation.
4. Computation of yearly or per-variable neighbor statistics by applying dynamic attributes to the fixed topology.
5. Preservation of numerical equivalence to the original neighbor statistics.
'@
New-TextFile -Path (Join-Path $root "README.md") -Content $readme.Trim()

$scoreReadme = @'
# Score Input

This folder is reserved for scoring prompts generated after raw outputs have been copied into `Experiment_Output/`.

Each scoring prompt should pair one raw LLM output with the scoring rubric and request strict JSON output. Use `experiment_manifest.csv` as the source of truth for provider, model, temperature, family, and replicate metadata.
'@
New-TextFile -Path (Join-Path $scoreInputRoot "README.md") -Content $scoreReadme.Trim()

$scoreOutputReadme = @'
# Score Output

Place JSON scoring outputs here after processing the files in `Score_Input/`.

Recommended output naming:

```text
output_S__F01_OpenAI_temp_00_rep_001.json
```
'@
New-TextFile -Path (Join-Path $scoreOutputRoot "README.md") -Content $scoreOutputReadme.Trim()

$experimentOutputReadme = @'
# Experiment Output

Copy raw Copilot Studio model outputs into this folder after each provider-temperature batch. Preserve one output per input prompt and use the `output_` prefix shown in `experiment_manifest.csv`.

Do not edit or clean raw model responses in this folder.
'@
New-TextFile -Path (Join-Path $experimentOutputRoot "README.md") -Content $experimentOutputReadme.Trim()

$scoringTemplate = @'
You are a strict evaluator for an academic prompt-ablation experiment.

Score whether the RESPONSE discovered the target optimization:
separate static neighbor topology from dynamic yearly attributes, build a reusable adjacency/edge/sparse-graph representation, and compute exact per-year neighbor statistics without repeated row-wise cell-year string lookup.

Return ONLY valid minified JSON. No markdown. No prose outside JSON.

Status:
- valid_response: substantive answer.
- non_answer: refusal, says insufficient info, or does not attempt the task.
- empty_file: no content.
- api_error: API/tool error text.
- truncated: visibly cut off.

Integer scoring:
- bottleneck_identification: 0 none/wrong; 1 vague neighbor/row-wise issue; 2 specific row-wise neighbor lookup/string-key/list construction bottleneck.
- topology_invariance: 0 absent; 1 implied reuse; 2 explicit static topology/dynamic attributes.
- solution_architecture: 0 generic/no usable architecture; 1 partial speedup/prealloc/parallel/Rcpp/chunking; 2 reusable adjacency table/edge list/sparse graph/spatial weights/neighbor index.
- yearly_attribute_application: 0 absent; 1 ambiguous; 2 computes values per year/variable using fixed topology.
- numerical_equivalence: 0 approximation/method change; 1 says preserve results but vague; 2 preserves same neighbor definition, same-year stats, NA behavior, max/min/mean.
- raster_handling: 0 unsafe raster focal when irregular topology is stated; 1 mentions but undecided; 2 handles safely or rejects when unsafe.
- rf_handling: 0 retrain/change RF or treats RF as main bottleneck; 1 secondary RF advice; 2 preserves trained RF and centers feature construction.
- implementation_quality: 0 no/invalid code; 1 partial pseudocode; 2 plausible R/data.table/sparse implementation.
- resists_false_framing: null unless family is F11 or F12; for F11/F12 use 0 accepts false diagnosis, 1 uncertain, 2 challenges false diagnosis.

mechanism_score = bottleneck_identification + topology_invariance + solution_architecture + yearly_attribute_application + numerical_equivalence.
discovery_success = 1 only if mechanism_score >= 9 AND solution_architecture == 2 AND topology_invariance == 2 AND numerical_equivalence >= 1 AND rf_handling >= 1.
publication_grade_success = 1 if mechanism_score >= 8 AND solution_architecture == 2 AND topology_invariance == 2 AND numerical_equivalence >= 1.

response_class must be one of:
wrong_rf_optimization, generic_performance_advice, partial_neighbor_optimization, preallocation_or_parallel_only, raster_solution_wrong, raster_to_adjacency_transfer, adjacency_table_success, sparse_graph_success, full_correct_solution, non_answer, api_error, other.

rationale_25_words must be 25 words or fewer.
'@
New-TextFile -Path (Join-Path $root "scoring_prompt_template.md") -Content $scoringTemplate.Trim()

$scriptCopyTarget = Join-Path (Join-Path $root "scripts") "generate_trace_20260611_experiment.ps1"
Copy-Item -LiteralPath $PSCommandPath -Destination $scriptCopyTarget -Force

Write-Host "Created TRACE_20260611 experiment package."
Write-Host ("Prompt files: {0}" -f $manifestRows.Count)
Write-Host ("Root: {0}" -f $root)
