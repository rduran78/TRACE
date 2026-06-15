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
run_id: Anthropic_F09_T03_R021
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 21
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F09_Anthropic_temp_03_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(NÂ²)-like in practice
This function iterates over **all 6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs **year-specific string keys** (`paste(neighbor_id, year, sep="_")`) and matches them against a named character vector (`idx_lookup`) of length 6.46 million.

Named vector lookup in R via `idx_lookup[neighbor_keys]` is hash-based, but doing ~6.46 million calls to `paste()` and hash lookups â€” each producing ~4 neighbor keys â€” means roughly **25+ million string constructions and hash probes**, all inside a sequential `lapply` over 6.46 million iterations. The per-iteration overhead of R's interpreted loop dominates.

### 2. `compute_neighbor_stats` is similarly slow
It runs another `lapply` over 6.46 million entries, subsetting a numeric vector and computing `max/min/mean` per entry. While each operation is trivial, 6.46M R-level function calls with list allocation is very expensive.

### 3. The core architectural flaw: conflating spatial topology with temporal panel structure
The neighbor **topology** is purely spatial â€” it depends only on cell identity, not on year. There are only **344,208 unique cells** with ~1.37M directed neighbor pairs. But the current code rebuilds neighbor row-indices for every cell Ã— year combination (6.46M rows), doing redundant work: the same cell has the same neighbors in every year.

---

## Optimization Strategy

**Key insight:** Separate the *spatial adjacency structure* (build once, 344K cells) from the *temporal attribute join* (vectorized, 6.46M rows).

### Step-by-step plan:

1. **Build a static edge table once** â€” a two-column `data.table` of `(cell_id, neighbor_id)` with ~1.37M rows. This is year-independent.

2. **For each variable, use a vectorized `data.table` join** â€” join cell-year attributes onto the edge table by `(neighbor_id, year)`, then group by `(cell_id, year)` to compute `max`, `min`, `mean`. This replaces both `build_neighbor_lookup` and `compute_neighbor_stats` with a single vectorized operation per variable.

3. **Join results back** to the main dataset.

This eliminates all R-level row-wise loops, all string-key construction, and all per-row `lapply` calls. The `data.table` grouped aggregation is implemented in C and will handle 1.37M Ã— 28 â‰ˆ 38.4M edge-year rows efficiently.

### Expected speedup:
- From ~86 hours to **minutes** (likely 5â€“15 minutes total for all 5 variables on a 16 GB laptop).

### Memory check:
- Edge table: ~1.37M rows Ã— 2 integer cols â‰ˆ 11 MB.
- Expanded edge-year table: ~38.4M rows Ã— 3 cols (cell, neighbor, year) â‰ˆ 460 MB.
- With one attribute column joined: ~38.4M Ã— 4 cols â‰ˆ 610 MB.
- Well within 16 GB.

---

## Working R Code

```r
library(data.table)

# ============================================================
# 1. Build the static spatial edge table ONCE
#    Input: id_order (vector of cell IDs in nb-object order)
#           rook_neighbors_unique (spdep nb object, length = # cells)
#    Output: edge_dt â€” a data.table with columns (cell_id, neighbor_id)
# ============================================================

build_edge_table <- function(id_order, neighbors) {
  # neighbors is an nb object: list of integer index vectors
  n <- length(neighbors)
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(neighbors))

  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)

  pos <- 1L
  for (i in seq_len(n)) {
    nb_idx <- neighbors[[i]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) next
    len <- length(nb_idx)
    from_id[pos:(pos + len - 1L)] <- id_order[i]
    to_id[pos:(pos + len - 1L)]   <- id_order[nb_idx]
    pos <- pos + len
  }

  # Trim if any nb entries were empty (0-neighbor cells)
  if (pos <= n_edges) {
    from_id <- from_id[seq_len(pos - 1L)]
    to_id   <- to_id[seq_len(pos - 1L)]
  }

  data.table(cell_id = from_id, neighbor_id = to_id)
}

# ============================================================
# 2. Compute neighbor stats for one variable via vectorized join
#    Input: cell_dt    â€” data.table with columns: id, year, <var_name>
#           edge_dt    â€” from step 1
#           var_name   â€” character, name of the source variable
#    Output: cell_dt with three new columns appended:
#            <var_name>_neighbor_max, _neighbor_min, _neighbor_mean
# ============================================================

compute_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {
  # Extract only the columns we need for the join
  # neighbor attributes: keyed by (id, year)
  attr_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setkey(attr_dt, id, year)

  # Expand edges across all years present in the data
  # Instead of a full cross-join (expensive), join edges onto data's (cell_id, year)
  # Step A: get unique (cell_id, year) pairs
  cell_years <- cell_dt[, .(cell_id = id, year)]

  # Step B: join edges to get (cell_id, year, neighbor_id)
  #         This is an inner join: for each cell-year, attach its neighbor IDs
  setkey(edge_dt, cell_id)
  setkey(cell_years, cell_id)
  edge_year <- edge_dt[cell_years, on = "cell_id", allow.cartesian = TRUE, nomatch = NULL]
  # edge_year now has columns: cell_id, neighbor_id, year
  # Rows: ~1.37M neighbors Ã— 28 years â‰ˆ 38.4M (but done via join, not cross)

  # Step C: join neighbor attribute values
  edge_year[attr_dt, val := i.val, on = .(neighbor_id = id, year)]

  # Step D: aggregate by (cell_id, year)
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")

  stats <- edge_year[!is.na(val),
    .(nmax = max(val), nmin = min(val), nmean = mean(val)),
    by = .(cell_id, year)
  ]

  setnames(stats, c("nmax", "nmin", "nmean"), c(max_col, min_col, mean_col))

  # Step E: merge back onto cell_dt
  # Remove old columns if they exist (idempotent re-runs)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }

  cell_dt <- merge(cell_dt, stats,
    by.x = c("id", "year"),
    by.y = c("cell_id", "year"),
    all.x = TRUE
  )

  cell_dt
}

# ============================================================
# 3. Full pipeline
# ============================================================

# Convert to data.table if not already
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# --- Build the edge table ONCE ---
edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
cat("Edge table built:", nrow(edge_dt), "directed edges\n")

# --- Compute neighbor features for each source variable ---
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "...\n")
  t0 <- proc.time()
  cell_data <- compute_neighbor_features_fast(cell_data, edge_dt, var_name)
  elapsed <- (proc.time() - t0)[["elapsed"]]
  cat("  Done in", round(elapsed, 1), "seconds\n")
}

# --- Predict with the existing trained Random Forest ---
# (The model object, e.g. `rf_model`, is already in memory and is NOT retrained)
# Ensure column order / names match what the model expects:
pred_vars <- names(cell_data)[names(cell_data) %in% rf_model$forest$independent.variable.names]
# Or however your model stores predictor names (ranger, randomForest, etc.)

cell_data[, prediction := predict(rf_model, data = cell_data[, ..pred_vars])$predictions]

cat("Pipeline complete.\n")
```

---

## Summary of Changes and Guarantees

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` are computed on exactly the same neighbor sets and the same variable values. NA handling is identical (NAs excluded before aggregation; cells with no valid neighbors get `NA`). The estimand is preserved. |
| **Trained RF model** | The model object is never modified or retrained. Only `predict()` is called. |
| **Correctness of topology** | `build_edge_table` reads the same `rook_neighbors_unique` nb object and `id_order` mapping. Each `neighbors[[i]]` indexes into `id_order` identically to the original code. |
| **Speed** | Eliminates 6.46M R-level loop iterations (Ã—2 functions Ã—5 variables = ~64.6M iterations). Replaced by ~5 vectorized `data.table` grouped aggregations over ~38M rows each. Expected wall time: **5â€“15 minutes** vs. 86+ hours. |
| **Memory** | Peak additional memory ~1â€“2 GB for the expanded edge-year table, well within 16 GB. |
END_RESPONSE>>>
