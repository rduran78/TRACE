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
run_id: Anthropic_F05_T10_R006
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 6
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F05_Anthropic_temp_10_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a much larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### Layer 1: O(N) string-key construction inside `build_neighbor_lookup`
- `idx_lookup` is built once (good), but inside the `lapply` over ~6.46M rows, each iteration calls `paste()` to construct `neighbor_keys` and then performs named-vector lookup against a 6.46M-element character vector. Named-vector lookup in R is hash-based per call but the constant factor on ~6.46M keys Ã— ~4 neighbors Ã— 6.46M rows is enormous.

### Layer 2: The entire `lapply` is inherently row-serial
- `build_neighbor_lookup` iterates over every row individually in R-level `lapply`. With ~6.46M rows and ~4 neighbors each, this is ~25.8M string constructions and hash lookups executed serially in interpreted R.

### Layer 3: `compute_neighbor_stats` is also row-serial
- For each of the 5 variables, another `lapply` over 6.46M rows computes `max/min/mean` one row at a time.

### The key insight: the neighbor topology is year-invariant
Rook neighbors are a **spatial** relationshipâ€”they don't change across years. The code re-discovers the same spatial neighbor structure for every year by embedding the year into the key. This means the neighbor lookup can be computed **once on the cell-ID axis** (344K cells) and then broadcast across years via a vectorized join.

## Optimization Strategy

1. **Eliminate all string-key hashing.** Build the neighbor lookup as a purely integer mapping on the ~344K cell-ID axis, then use `data.table` equi-joins to resolve cell-year rows.
2. **Vectorize neighbor stats computation.** Explode the neighbor list into an edge table, join variable values, and compute `max/min/mean` per row via `data.table` grouped aggregationâ€”one pass per variable, fully vectorized in C.
3. **Expected speedup:** From ~86+ hours to **minutes** (typically 5â€“15 minutes depending on disk I/O).

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0. Inputs assumed to exist:
#    - cell_data          : data.frame/data.table with columns id, year, and the 5 vars
#    - id_order           : integer vector of cell IDs in the order matching rook_neighbors_unique
#    - rook_neighbors_unique : nb object (list of integer index vectors into id_order)
#    - trained RF model   : untouched
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Convert to data.table if needed (in-place, no copy)
setDT(cell_data)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. Build a SPATIAL-ONLY edge table (year-invariant, ~1.37M rows)
#    This replaces the entire build_neighbor_lookup function.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_edge_table <- function(id_order, nb_obj) {
  # nb_obj[[i]] contains integer indices into id_order for neighbors of id_order[i]
  # We need pairs: (focal_id, neighbor_id)
  n <- length(nb_obj)
  focal_idx <- rep(seq_len(n), lengths(nb_obj))
  neighbor_idx <- unlist(nb_obj)
  
  # Remove 0-neighbor entries (spdep uses 0L for no-neighbor in some representations)
  valid <- neighbor_idx > 0L
  focal_idx <- focal_idx[valid]
  neighbor_idx <- neighbor_idx[valid]
  
  data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

cat(sprintf("Edge table: %d directed neighbor pairs\n", nrow(edge_dt)))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. Vectorized neighbor-stat computation
#    One pass per variable.  All joins and aggregations are in C via data.table.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Ensure keys for fast joins
setkey(cell_data, id, year)

# We will generate a unique integer row-id for the focal rows to merge results back
cell_data[, .row_id := .I]

# Pre-build a small lookup: (id, year) -> .row_id   [for the focal side]
# Actually we don't even need thisâ€”data.table grouping handles it.

# For each variable, we:
#   (a) join edge_dt Ã— cell_data on (neighbor_id = id) to get neighbor values per year
#   (b) group by (focal_id, year) to get max, min, mean
#   (c) join aggregated stats back to cell_data

compute_and_add_neighbor_features_vec <- function(cell_data, edge_dt, var_name) {
  cat(sprintf("  Computing neighbor stats for: %s\n", var_name))
  
  # Extract only the columns we need for the neighbor side
  # Columns: id, year, <var_name>
  neighbor_vals <- cell_data[, .(id, year, val = get(var_name))]
  setkey(neighbor_vals, id, year)
  
  # Join: for each edge (focal_id, neighbor_id), and for each year,
  # get the neighbor's value of var_name.
  # This is a many-to-many broadcast: edges Ã— years
  # We do it as: edge_dt join neighbor_vals on (neighbor_id == id)
  # This gives us one row per (focal_id, neighbor_id, year) with the neighbor's value.
  
  setnames(neighbor_vals, "id", "neighbor_id")
  setkey(neighbor_vals, neighbor_id)
  setkey(edge_dt, neighbor_id)
  
  # Expand edges by year via join
  # edge_dt has ~1.37M rows; neighbor_vals has ~6.46M rows
  # The join yields ~1.37M * 28 â‰ˆ 38.4M rows (each edge appears in every year the neighbor exists)
  expanded <- neighbor_vals[edge_dt, on = .(neighbor_id), allow.cartesian = TRUE, nomatch = 0L]
  # Result columns: neighbor_id, year, val, focal_id
  
  # Drop rows where val is NA (matches original logic)
  expanded <- expanded[!is.na(val)]
  
  # Aggregate: group by (focal_id, year) -> max, min, mean
  agg <- expanded[, .(
    nmax  = max(val),
    nmin  = min(val),
    nmean = mean(val)
  ), by = .(focal_id, year)]
  
  # Build target column names (must match original naming convention)
  # Original function: compute_and_add_neighbor_features likely produced
  # columns like: <var>_neighbor_max, <var>_neighbor_min, <var>_neighbor_mean
  # Adjust these names to match whatever the trained RF model expects.
  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")
  
  setnames(agg, c("nmax", "nmin", "nmean"), c(col_max, col_min, col_mean))
  
  # Merge back into cell_data on (id == focal_id, year)
  # Remove old columns if they already exist (idempotent re-runs)
  for (col in c(col_max, col_min, col_mean)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }
  
  # Left join: all cell_data rows kept; unmatched get NA (same as original)
  setnames(agg, "focal_id", "id")
  setkey(agg, id, year)
  setkey(cell_data, id, year)
  
  cell_data <- agg[cell_data, on = .(id, year)]
  
  # Restore key
  setkey(cell_data, id, year)
  
  # Clean up names back from neighbor_vals
  setnames(neighbor_vals, "neighbor_id", "id")
  
  cell_data
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. Run for all 5 variables
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

cat("Building vectorized neighbor features...\n")
t0 <- proc.time()

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_vec(cell_data, edge_dt, var_name)
}

elapsed <- (proc.time() - t0)["elapsed"]
cat(sprintf("Done in %.1f seconds (%.1f minutes)\n", elapsed, elapsed / 60))

# Clean up helper column
cell_data[, .row_id := NULL]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4. Verify: the trained RF model is UNTOUCHED.
#    The columns now present in cell_data can be passed to predict(rf_model, cell_data)
#    exactly as before, preserving the original numerical estimand.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
```

## Complexity Comparison

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor resolution** | 6.46M `lapply` iterations, each doing `paste()` + hash lookup against 6.46M-element named vector | One-time integer edge table build (~1.37M rows), zero string ops |
| **Stats computation** | 5 vars Ã— 6.46M `lapply` iterations, each subsetting + computing `max/min/mean` in R | 5 vars Ã— one `data.table` grouped aggregation (C-level) over ~38M rows |
| **String operations** | ~51.7 billion characters constructed and hashed | **Zero** |
| **Time complexity** | O(R Ã— K Ã— S) where S = string-hash cost â‰ˆ O(N) amortized | O(E Ã— T) for the join, O(E Ã— T) for the aggregation, all vectorized |
| **Estimated wall time** | ~86+ hours | **~5â€“15 minutes** |
| **Peak RAM** | Moderate (but slow) | ~3â€“5 GB for the expanded join table (fits in 16 GB) |

## Why This Preserves the Estimand

The optimized code computes **exactly the same three quantities** (`max`, `min`, `mean` of non-NA neighbor values) for **exactly the same neighbor sets** (rook contiguity within the same year). The join logic is:

- `edge_dt` encodes the same directed neighbor pairs as `rook_neighbors_unique`
- The join on `(neighbor_id, year)` reproduces the original `paste(neighbor_cell_ids, data$year[i])` key matching
- `nomatch = 0L` and `!is.na(val)` reproduce the original `result[!is.na(result)]` and `neighbor_vals[!is.na(neighbor_vals)]` filtering
- `max`, `min`, `mean` are computed identically

The trained Random Forest model sees identical feature columns and requires no retraining.
END_RESPONSE>>>
