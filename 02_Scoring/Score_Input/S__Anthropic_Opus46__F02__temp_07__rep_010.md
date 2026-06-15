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
run_id: Anthropic_F02_T07_R010
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 10
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F02_Anthropic_temp_07_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which runs an `lapply` over ~6.46 million rows, performing per-row string pasting, hash lookups, and NA filtering. This produces a list of 6.46 million integer vectors. Then, `compute_neighbor_stats` iterates over that same 6.46 million-element list once per variable (Ã—5 variables), subsetting and computing summary statistics in pure R loops. Specific problems:

1. **`build_neighbor_lookup` creates ~6.46M character keys** (`paste(id, year)`) and does repeated named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) hashing per call, and doing it millions of times inside `lapply` is extremely slow.
2. **The neighbor lookup list itself is enormous in memory.** 6.46M list elements, each an integer vector, carries heavy R object overhead (~hundreds of bytes per list element even if short), easily consuming 5â€“10+ GB.
3. **`compute_neighbor_stats` is called 5 times**, each time iterating over the 6.46M-element list in pure R â€” another ~32M R-level function calls total.
4. **No vectorization or data.table/matrix operations** are used anywhere; everything is scalar/list-based R.

---

## Optimization Strategy

**Replace the per-row list-based approach with a fully vectorized, edge-list / `data.table` group-by approach:**

- Instead of building a 6.46M-element list mapping each row to its neighbors, construct a **long edge table** (`data.table`) where each row is a `(focal_row, neighbor_row)` pair. With ~1.37M directed neighbor relationships Ã— 28 years â‰ˆ ~38.5M rows, this is large but manageable in columnar form (~300 MB for a two-column integer table).
- Join the variable values onto the neighbor side, then **group-by `focal_row`** to compute `max`, `min`, `mean` in one vectorized `data.table` aggregation per variable.
- This eliminates all per-row `lapply`, all character key construction inside loops, and all R-level list overhead.
- Estimated speedup: from 86+ hours to **minutes**.
- Estimated peak RAM: well within 16 GB.

**The trained Random Forest model is never touched. The numerical outputs (max, min, mean of neighbor values) are identical â€” just computed faster.**

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 1: Build a long edge table (focal_id, neighbor_id) from the nb object
#         This is done ONCE and reused for all variables and all years.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_edge_dt <- function(id_order, neighbors) {
  # neighbors is an spdep::nb list: neighbors[[i]] gives integer indices

  # into id_order for the neighbors of id_order[i].
  n <- length(neighbors)
  
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(neighbors))
  
  focal_id    <- integer(n_edges)
  neighbor_id <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    nb_i <- neighbors[[i]]
    if (length(nb_i) == 0L || (length(nb_i) == 1L && nb_i[1] == 0L)) next
    len <- length(nb_i)
    idx <- pos:(pos + len - 1L)
    focal_id[idx]    <- id_order[i]
    neighbor_id[idx] <- id_order[nb_i]
    pos <- pos + len
  }
  
  # Trim if any nb entries were empty (0-sentinel)
  if (pos - 1L < n_edges) {
    focal_id    <- focal_id[1:(pos - 1L)]
    neighbor_id <- neighbor_id[1:(pos - 1L)]
  }
  
  data.table(focal_id = focal_id, neighbor_id = neighbor_id)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 2: Expand edges across years and map to row indices
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_neighbor_edge_table <- function(cell_data_dt, id_order, neighbors) {
  # cell_data_dt must be a data.table with columns: id, year, and a .ROW_IDX column
  
  # 2a. Spatial edge list (id-level, ~1.37M rows)
  edge_dt <- build_edge_dt(id_order, neighbors)
  
  # 2b. Get the unique years
  years <- sort(unique(cell_data_dt$year))
  
  # 2c. Cross-join edges Ã— years to get (focal_id, neighbor_id, year)
  #     ~1.37M Ã— 28 â‰ˆ 38.5M rows
  year_dt <- data.table(year = years)
  edge_year_dt <- edge_dt[, CJ_idx := 1L][year_dt[, CJ_idx := 1L], 
                                            on = "CJ_idx", 
                                            allow.cartesian = TRUE]
  edge_year_dt[, CJ_idx := NULL]
  
  # 2d. Map focal (id, year) -> row index in cell_data_dt
  id_year_to_row <- cell_data_dt[, .(id, year, .ROW_IDX)]
  
  # Join to get focal row index
  edge_year_dt <- merge(edge_year_dt, id_year_to_row, 
                        by.x = c("focal_id", "year"), 
                        by.y = c("id", "year"), 
                        all.x = FALSE)
  setnames(edge_year_dt, ".ROW_IDX", "focal_row")
  

  # Join to get neighbor row index
  edge_year_dt <- merge(edge_year_dt, id_year_to_row, 
                        by.x = c("neighbor_id", "year"), 
                        by.y = c("id", "year"), 
                        all.x = FALSE)
  setnames(edge_year_dt, ".ROW_IDX", "neighbor_row")
  
  # Keep only what we need
  edge_year_dt[, .(focal_row, neighbor_row)]
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 3: Compute neighbor stats for one variable using vectorized groupby
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

compute_neighbor_stats_fast <- function(cell_data_dt, edge_table, var_name) {
  # edge_table has columns: focal_row, neighbor_row
  # Attach the neighbor's value
  vals <- cell_data_dt[[var_name]]
  
  work <- edge_table[, .(focal_row, nval = vals[neighbor_row])]
  
  # Drop NA neighbor values
  work <- work[!is.na(nval)]
  
  # Aggregate
  agg <- work[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = focal_row]
  
  # Initialize output columns with NA
  n <- nrow(cell_data_dt)
  out_max  <- rep(NA_real_, n)
  out_min  <- rep(NA_real_, n)
  out_mean <- rep(NA_real_, n)
  
  out_max[agg$focal_row]  <- agg$nb_max
  out_min[agg$focal_row]  <- agg$nb_min
  out_mean[agg$focal_row] <- agg$nb_mean
  
  # Return as a data.table with proper column names
  prefix <- paste0("neighbor_", var_name)
  result <- data.table(
    v_max  = out_max,
    v_min  = out_min,
    v_mean = out_mean
  )
  setnames(result, c(paste0(prefix, "_max"), 
                      paste0(prefix, "_min"), 
                      paste0(prefix, "_mean")))
  result
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 4: Main driver â€” drop-in replacement for the original outer loop
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

add_all_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                      neighbor_source_vars = c("ntl", "ec", 
                                        "pop_density", "def", "usd_est_n2")) {
  
  cell_data_dt <- as.data.table(cell_data)
  cell_data_dt[, .ROW_IDX := .I]
  
  message("Building edge table (one-time cost)...")
  edge_table <- build_neighbor_edge_table(cell_data_dt, id_order, rook_neighbors_unique)
  message(sprintf("Edge table: %s rows (%.1f MB)", 
                  format(nrow(edge_table), big.mark = ","),
                  object.size(edge_table) / 1e6))
  
  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))
    stats <- compute_neighbor_stats_fast(cell_data_dt, edge_table, var_name)
    
    # Bind columns to the data.table
    cell_data_dt <- cbind(cell_data_dt, stats)
  }
  
  cell_data_dt[, .ROW_IDX := NULL]
  
  # Return as original class (data.frame or data.table)
  if (!is.data.table(cell_data)) {
    return(as.data.frame(cell_data_dt))
  }
  cell_data_dt
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Usage (drop-in replacement for the original outer loop):
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

cell_data <- add_all_neighbor_features(
  cell_data, 
  id_order, 
  rook_neighbors_unique,
  neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
)

# The trained Random Forest model is unchanged.
# Predictions proceed as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Works and What Changes

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M-element R list via `lapply` + character key hashing | One `data.table` edge table built via vectorized merge (~38.5M rows, two integer columns) |
| **Stats computation** | `lapply` over 6.46M elements, per-variable | `data.table` grouped aggregation (`[, .(max, min, mean), by = focal_row]`), per-variable |
| **Memory model** | ~6.46M R list elements (huge overhead) | ~38.5M Ã— 2 integer columns â‰ˆ ~310 MB + working copies â‰ˆ ~2â€“3 GB peak |
| **Time estimate** | 86+ hours | ~5â€“15 minutes on a 16 GB laptop |
| **Numerical output** | max, min, mean of non-NA neighbor values | Identical max, min, mean of non-NA neighbor values |
| **RF model** | Untouched | Untouched |

The key insight is that the neighbor structure is **year-invariant** (same spatial grid, same rook neighbors every year), so the ~1.37M spatial edges can be crossed with 28 years once, producing a single reusable edge table. All subsequent per-variable computations are then a single vectorized `data.table` group-by â€” orders of magnitude faster than 6.46 million R-level function calls.
END_RESPONSE>>>
