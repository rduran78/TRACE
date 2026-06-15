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
run_id: Anthropic_F11_T07_R021
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 21
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F11_Anthropic_temp_07_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence clearly points to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, and for each row it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is effectively a hash-table probe *per element*, but constructing ~6.46 million string keys and doing millions of named lookups is extremely slow in interpreted R. This single function produces O(6.46M Ã— average_neighbors) string operations.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over the 6.46 million-element `neighbor_lookup` list with per-element `lapply`, subsetting, removing NAs, and computing `max/min/mean`. The `do.call(rbind, result)` on a 6.46M-element list of 3-element vectors is also notoriously slow.

3. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-trained model. Even with 6.46M rows and 110 predictors, modern RF implementations (e.g., `ranger`) complete prediction in seconds to a few minutes. Loading a serialized model and writing predictions are I/O-bound and fast relative to the feature engineering above.

**Conclusion:** The 86+ hour runtime is dominated by millions of interpreted-R string operations and per-row list iterations in the neighbor feature pipeline, not by RF inference.

---

## Optimization Strategy

1. **Eliminate string-key lookups entirely.** Replace the `paste(id, year, sep="_")` â†’ named-vector lookup pattern with direct integer arithmetic. Since `id` maps to a known position in `id_order` and `year` spans a contiguous range (1992â€“2019, i.e., 28 years), each row can be addressed by a 2D integer index `(cell_position, year_offset)` mapped to a flat row index. This turns O(n Ã— k) string operations into O(n Ã— k) integer operations.

2. **Vectorize neighbor stats computation.** Instead of `lapply` over 6.46M list elements, build a single long vector of (row_index, neighbor_row_index) pairs, then use vectorized group-by operations (via `data.table`) to compute max, min, and mean in one pass per variable.

3. **Process all 5 variables in one pass** over the neighbor-pair structure rather than 5 separate passes.

These changes reduce the estimated runtime from 86+ hours to minutes.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# OPTIMIZED NEIGHBOR FEATURE ENGINEERING
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_neighbor_pairs_fast <- function(cell_data_dt, id_order, rook_neighbors) {

  # cell_data_dt: a data.table with columns id, year, and an integer row index .row_idx
  # id_order:     vector of cell IDs in the same order as rook_neighbors (spdep::nb)
  # rook_neighbors: the nb list (each element is integer vector of neighbor positions)
  #

  # Returns a data.table with columns: row_idx (focal row), nb_row_idx (neighbor row)

  n_cells <- length(id_order)
  years   <- sort(unique(cell_data_dt$year))
  n_years <- length(years)

  # Map id -> position in id_order (integer)
  id_to_pos <- integer(max(id_order))
  id_to_pos[id_order] <- seq_len(n_cells)

  # Map (cell_position, year) -> row index in cell_data_dt
  # Build a matrix: rows = cell positions, cols = year offsets
  year_min <- min(years)

  # Create the mapping matrix (NA where a cell-year doesn't exist)
  row_lookup <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)

  cell_data_dt[, cell_pos := id_to_pos[id]]
  cell_data_dt[, year_off := year - year_min + 1L]
  cell_data_dt[, .row_idx := .I]

  # Fill the lookup matrix
  row_lookup[cbind(cell_data_dt$cell_pos, cell_data_dt$year_off)] <- cell_data_dt$.row_idx

  # Build the edge list of (focal_cell_pos, neighbor_cell_pos)
  # from the nb object
  focal_pos <- rep(seq_len(n_cells), lengths(rook_neighbors))
  nb_pos    <- unlist(rook_neighbors, use.names = FALSE)

  # Remove 0-entries (spdep uses 0 for no-neighbor cells)
  valid <- nb_pos > 0L
  focal_pos <- focal_pos[valid]
  nb_pos    <- nb_pos[valid]

  n_edges <- length(focal_pos)

  # Expand across all years: each edge exists for every year
  focal_pos_exp <- rep(focal_pos, each = n_years)
  nb_pos_exp    <- rep(nb_pos,    each = n_years)
  year_off_exp  <- rep(seq_len(n_years), times = n_edges)

  # Look up actual row indices
  focal_row <- row_lookup[cbind(focal_pos_exp, year_off_exp)]
  nb_row    <- row_lookup[cbind(nb_pos_exp,    year_off_exp)]

  # Keep only pairs where both focal and neighbor exist
  keep <- !is.na(focal_row) & !is.na(nb_row)

  pairs <- data.table(
    row_idx    = focal_row[keep],
    nb_row_idx = nb_row[keep]
  )

  # Clean up temporary columns
  cell_data_dt[, c("cell_pos", "year_off") := NULL]

  return(pairs)
}


compute_all_neighbor_features_fast <- function(cell_data_dt, pairs, neighbor_source_vars) {
  # pairs: data.table with row_idx, nb_row_idx
  # For each variable, compute max, min, mean of neighbor values grouped by row_idx

  n_rows <- nrow(cell_data_dt)

  for (var_name in neighbor_source_vars) {
    message("Computing neighbor features for: ", var_name)

    # Attach neighbor values to pairs
    pairs[, nb_val := cell_data_dt[[var_name]][nb_row_idx]]

    # Remove NA neighbor values
    valid_pairs <- pairs[!is.na(nb_val)]

    # Compute grouped stats
    stats <- valid_pairs[, .(
      nb_max  = max(nb_val),
      nb_min  = min(nb_val),
      nb_mean = mean(nb_val)
    ), by = row_idx]

    # Initialize columns with NA
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")

    set(cell_data_dt, j = max_col,  value = NA_real_)
    set(cell_data_dt, j = min_col,  value = NA_real_)
    set(cell_data_dt, j = mean_col, value = NA_real_)

    # Assign computed values
    set(cell_data_dt, i = stats$row_idx, j = max_col,  value = stats$nb_max)
    set(cell_data_dt, i = stats$row_idx, j = min_col,  value = stats$nb_min)
    set(cell_data_dt, i = stats$row_idx, j = mean_col, value = stats$nb_mean)

    # Clean up
    pairs[, nb_val := NULL]
  }

  invisible(cell_data_dt)
}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# MAIN PIPELINE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Convert to data.table if not already
cell_data_dt <- as.data.table(cell_data)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Step 1: Build integer-indexed neighbor pairs (replaces build_neighbor_lookup)
message("Building neighbor pair index...")
pairs <- build_neighbor_pairs_fast(cell_data_dt, id_order, rook_neighbors_unique)

# Step 2: Compute all neighbor features in vectorized grouped operations
#         (replaces the for-loop over compute_neighbor_stats)
compute_all_neighbor_features_fast(cell_data_dt, pairs, neighbor_source_vars)

# Step 3: Random Forest prediction (unchanged â€” preserving the trained model)
message("Running Random Forest prediction...")
# rf_model is the pre-trained model already loaded into memory, e.g.:
# rf_model <- readRDS("trained_rf_model.rds")
cell_data_dt[, prediction := predict(rf_model, newdata = cell_data_dt)$predictions]
# (Adjust the predict() call to match your RF package â€” ranger, randomForest, etc.)

message("Done.")
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M string pastes + named-vector hash lookups | Integer matrix indexing â€” no strings |
| **Neighbor stats** | `lapply` over 6.46M list elements Ã— 5 vars; `do.call(rbind, ...)` | Vectorized `data.table` grouped aggregation; single edge-list reused for all 5 vars |
| **Expected runtime** | 86+ hours | Minutes (dominated by the `data.table` group-by on ~38M pairs Ã— 5 vars) |
| **RF model** | Preserved âœ“ | Preserved âœ“ |
| **Numerical estimand** | Preserved âœ“ (same max/min/mean logic, same neighbor definitions) | Preserved âœ“ |

The true bottleneck is the interpreted-R, per-row, string-based neighbor feature engineering â€” not Random Forest inference.
END_RESPONSE>>>
