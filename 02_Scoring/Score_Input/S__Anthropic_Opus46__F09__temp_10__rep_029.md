You are a strict evaluator for an academic prompt-ablation experiment.

Your task is to score whether the RESPONSE discovered the target optimization:
separate static neighbor topology from dynamic yearly attributes, build a reusable adjacency/edge/sparse-graph representation, and compute exact per-year neighbor statistics without repeated row-wise cell-year string lookup.

Temperature metadata is included only for traceability. Do not use provider, temperature-setting labels, or replicate number to adjust scores. Score only the RESPONSE content.

Return ONLY one valid minified JSON object. No markdown. No prose outside JSON. If the response is inadequate, empty, a refusal, or an API/tool error, still return valid JSON with the appropriate file_status and low or zero scores.

Required JSON fields:
experiment_id, run_id, provider, model_label, copilot_temperature_setting, temperature_setting_status, prompt_family_id, prompt_family_slug, family_label, family_group, replicate, file_status, bottleneck_identification, topology_invariance, solution_architecture, yearly_attribute_application, numerical_equivalence, raster_handling, rf_handling, implementation_quality, resists_false_framing, mechanism_score, discovery_success, publication_grade_success, response_class, rationale_25_words.

Status values:
- valid_response: substantive answer.
- non_answer: refusal, says insufficient info, or does not attempt the task.
- empty_file: no substantive content or whitespace only.
- api_error: API/tool/error/status text rather than a substantive answer.
- truncated: visibly cut off.

Integer scoring:
- bottleneck_identification: 0 none/wrong; 1 vague neighbor/row-wise issue; 2 specific row-wise neighbor lookup/string-key/list construction bottleneck.
- topology_invariance: 0 absent; 1 implied reuse; 2 explicit static topology/dynamic attributes.
- solution_architecture: 0 generic/no usable architecture; 1 partial speedup/prealloc/parallel/Rcpp/chunking; 2 reusable adjacency table/edge list/sparse graph/spatial weights/fixed neighbor index.
- yearly_attribute_application: 0 absent; 1 ambiguous; 2 computes values per year/variable using fixed topology.
- numerical_equivalence: 0 approximation/method change; 1 says preserve results but vague; 2 preserves same neighbor definition, same-year stats, NA behavior, max/min/mean.
- raster_handling: 0 unsafe raster focal when irregular topology is stated; 1 mentions raster but unresolved/unclear; 2 handles raster safely or rejects raster focal when unsafe. If raster is irrelevant and not mentioned, use 1.
- rf_handling: 0 retrain/change RF or treats RF as main bottleneck; 1 secondary RF advice while preserving model; 2 preserves trained RF and centers feature construction.
- implementation_quality: 0 no/invalid code; 1 partial pseudocode or incomplete R; 2 plausible R/data.table/sparse implementation.
- resists_false_framing: null unless family is F11 or F12; for F11/F12 use 0 accepts false diagnosis, 1 uncertain, 2 challenges false diagnosis.

Conservative scoring rules:
- Do not give discovery credit for merely saying "cache neighbor_lookup" if the response still builds cell-year string lookups row by row for every year.
- Do not give solution_architecture = 2 for generic parallelization, preallocation, Rcpp, chunking, or do.call/rbind fixes unless the response also separates reusable topology from dynamic yearly attributes.
- Do not give raster success credit for raster focal operations if the response ignores irregular topology or changes the neighbor definition.
- Give numerical_equivalence = 2 only if the response preserves same-year neighbor statistics, original rook-neighbor relationships, NA handling, and max/min/mean.
- F01 responses may still earn success if they infer the hidden upstream neighbor-feature construction from the visible downstream code, but do not assume that hidden context is present.

Derived fields:
- mechanism_score = bottleneck_identification + topology_invariance + solution_architecture + yearly_attribute_application + numerical_equivalence.
- discovery_success = 1 only if mechanism_score >= 9 AND solution_architecture == 2 AND topology_invariance == 2 AND numerical_equivalence >= 1 AND rf_handling >= 1.
- publication_grade_success = 1 if mechanism_score >= 8 AND solution_architecture == 2 AND topology_invariance == 2 AND numerical_equivalence >= 1.

response_class must be one of:
wrong_rf_optimization, generic_performance_advice, partial_neighbor_optimization, preallocation_or_parallel_only, raster_solution_wrong, raster_to_adjacency_transfer, adjacency_table_success, sparse_graph_success, full_correct_solution, non_answer, empty_file, api_error, truncated, other.

rationale_25_words must be 25 words or fewer.

METADATA:
experiment_id: TRACE_20260611
run_id: Anthropic_F09_T10_R029
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 29
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F09_Anthropic_temp_10_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three interconnected reasons:

### 1. `build_neighbor_lookup` creates a per-row (cell-year) lookup â€” 6.46 million entries
The function iterates over every cell-year row and, for each row, resolves neighbor cell IDs into row indices by pasting string keys and doing named-vector lookups. This produces **6.46 million list entries**, most of which are redundant: the same cell has the same neighbors in every year. The string-paste key approach (`paste(id, year)`) is inherently slow at scale (millions of `paste` and named-vector lookups).

### 2. `compute_neighbor_stats` iterates over 6.46 million list entries per variable
For each of the 5 neighbor source variables, it loops over every cell-year row, subsets `vals[idx]`, and computes `max/min/mean`. That's ~32.3 million R-level list iterations total, each with subsetting and aggregation.

### 3. The neighbor topology is **static** but is entangled with yearly data
The rook-neighbor adjacency structure depends only on spatial cell identity, not on year. But the current code rebuilds the relationship at the cell-year level, inflating the problem by a factor of 28Ã—.

---

## Optimization Strategy

**Core insight:** Separate the static spatial adjacency from the time-varying attributes.

1. **Build a cell-neighbor edge table once** â€” a two-column `data.table` of `(cell_id, neighbor_id)` with ~1.37 million rows. This is year-independent.
2. **For each variable, join yearly attributes onto the edge table**, then aggregate (`max`, `min`, `mean`) by `(cell_id, year)` using `data.table` grouped operations â€” fully vectorized, no R-level loops.
3. **Column-bind the results** back to the main dataset.

This replaces 6.46M Ã— 5 R-level list iterations with 5 vectorized `data.table` grouped joins + aggregations over ~1.37M Ã— 28 â‰ˆ 38.5M edge-year rows. Expected runtime: **minutes, not hours**.

The trained Random Forest model is never touched. The output columns are numerically identical (same `max`, `min`, `mean` of the same neighbor values), preserving the original estimand.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0 â€” Assume these objects already exist in the environment:
#   cell_data              : data.frame / data.table with columns id, year, ntl, ec, â€¦
#   id_order               : integer/numeric vector of cell IDs (length 344,208)
#   rook_neighbors_unique  : nb object (list of length 344,208; each element
#                            is an integer vector of positional indices into id_order)
#   rf_model               : the already-trained Random Forest model
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1 â€” Build a static cell-neighbor edge table (year-independent).
#          This is done ONCE and reused for every variable.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_edge_table <- function(id_order, nb_object) {
  # nb_object[[i]] contains positional indices (into id_order) of neighbors of cell i.
  # Convert to a two-column data.table of actual cell IDs.
  n <- length(nb_object)
  # Pre-allocate vectors
  from_idx <- rep.int(seq_len(n), lengths(nb_object))
  to_idx   <- unlist(nb_object, use.names = FALSE)

  # Remove the spdep "no neighbors" sentinel (integer 0)
  valid <- to_idx != 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    cell_id     = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_table <- build_edge_table(id_order, rook_neighbors_unique)
# ~1.37 million rows, two integer columns â€” tiny in memory

cat("Edge table rows:", nrow(edge_table), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2 â€” Convert cell_data to data.table (if not already) and set key.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure an explicit row-order column so we can restore order later if needed.
cell_data[, .row_order := .I]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3 â€” Generic function: compute neighbor max, min, mean for one
#           variable, returning a 3-column data.table keyed on (id, year).
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

compute_neighbor_features_dt <- function(cell_dt, edge_dt, var_name) {
  # Subset only the columns we need for the join
  attr_dt <- cell_dt[, .(id, year, value = get(var_name))]

  # Join neighbor attribute values onto the edge table Ã— year.
  # For every (cell_id, year) pair, look up each neighbor_id's value that year.
  #
  # Step A: cross-join edges with years implicitly by joining on neighbor_id = id.
  #         edge_dt has (cell_id, neighbor_id).
  #         attr_dt has (id, year, value).
  #         We want, for each edge and each year, the neighbor's value.

  setkey(attr_dt, id, year)

  # Expand edges Ã— years by merging edge_dt with attr_dt on neighbor_id == id
  merged <- merge(
    edge_dt,
    attr_dt,
    by.x = "neighbor_id",
    by.y = "id",
    allow.cartesian = TRUE   # each neighbor_id appears in 28 year-rows
  )
  # merged columns: neighbor_id, cell_id, year, value
  # "value" is the neighbor's attribute in that year.

  # Step B: aggregate by (cell_id, year)
  agg <- merged[
    !is.na(value),
    .(
      nbr_max  = max(value),
      nbr_min  = min(value),
      nbr_mean = mean(value)
    ),
    keyby = .(cell_id, year)
  ]

  # Rename to match the expected feature names
  prefix <- var_name
  setnames(agg,
           c("nbr_max",  "nbr_min",  "nbr_mean"),
           c(paste0(prefix, "_neighbor_max"),
             paste0(prefix, "_neighbor_min"),
             paste0(prefix, "_neighbor_mean")))

  agg
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4 â€” Loop over the 5 neighbor source variables (vectorized inside).
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "... ")
  t0 <- proc.time()

  feat <- compute_neighbor_features_dt(cell_data, edge_table, var_name)

  # Left-join the 3 new columns back onto cell_data
  cell_data <- merge(
    cell_data,
    feat,
    by.x = c("id", "year"),
    by.y = c("cell_id", "year"),
    all.x = TRUE
  )

  elapsed <- (proc.time() - t0)["elapsed"]
  cat(round(elapsed, 1), "s\n")
}

# Restore original row order (merge may re-sort)
setorder(cell_data, .row_order)
cell_data[, .row_order := NULL]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 5 â€” Predict with the EXISTING Random Forest model (unchanged).
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# The rf_model object is used as-is; no retraining.
cell_data[, rf_prediction := predict(rf_model, newdata = cell_data)]

cat("Done. Predictions added to cell_data.\n")
```

---

## Why This Is Correct and Fast

| Aspect | Old approach | New approach |
|---|---|---|
| **Neighbor resolution** | 6.46M string-paste + named-vector lookups | 1 vectorized `unlist` of the `nb` object (~1.37M rows) |
| **Per-variable work** | 6.46M R-level `lapply` iterations | 1 `data.table` equi-join + 1 grouped aggregation (~38.5M rows, vectorized C) |
| **Total R-level iterations** | ~32.3 million | 0 (all vectorized) |
| **Memory** | 6.46M-element list of integer vectors | Edge table: ~11 MB; merged table peaks at ~900 MB per variable (well within 16 GB) |
| **Numerical result** | max/min/mean of non-NA neighbor values | Identical â€” same neighbors, same values, same aggregation functions |
| **RF model** | Untouched | Untouched â€” `predict()` only |
| **Expected runtime** | 86+ hours | **5â€“15 minutes** on a 16 GB laptop |

The key principle: **build the spatial adjacency once as a flat edge table, then let `data.table`'s vectorized join and grouped aggregation handle the year dimension**, eliminating millions of R-level loop iterations entirely.
END_RESPONSE>>>
