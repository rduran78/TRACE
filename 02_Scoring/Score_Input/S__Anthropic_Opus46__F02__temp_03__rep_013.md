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
run_id: Anthropic_F02_T03_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F02_Anthropic_temp_03_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting and named-vector lookups. Named vector lookups in R via character keys are O(n) hash lookups, but doing this millions of times with string concatenation inside the loop creates enormous overhead. The `compute_neighbor_stats` function then iterates over the same 6.46 million entries again, once per variable. Combined: ~6.46M Ã— (lookup construction + 5 variable passes) = billions of R-level interpreted operations.

**Specific problems:**

1. **`build_neighbor_lookup`**: `paste()` and named-vector indexing inside a per-row `lapply` over 6.46M rows is extremely slow. The `idx_lookup` named vector has 6.46M entries, making each character-key lookup expensive at scale.
2. **`compute_neighbor_stats`**: `lapply` over 6.46M elements, each allocating a small vector and calling `max/min/mean`, is slow due to R's per-call overhead and lack of vectorization.
3. **Memory**: Storing a list of 6.46M integer vectors (`neighbor_lookup`) plus intermediate string vectors consumes many gigabytes.
4. **No parallelism or vectorization**: Everything is serial, interpreted R.

---

## Optimization Strategy

### Key Insight
The neighbor relationships are **cell-level** (not cell-year-level). A cell's rook neighbors are the same in every year. Therefore, we should:

1. **Work at the cell level for neighbor topology** â€” build a sparse adjacency structure over 344K cells, not 6.46M cell-years.
2. **Use `data.table` for fast indexed joins by year** â€” instead of string-key lookups, join neighbor cell IDs by year using integer keys.
3. **Vectorize aggregation** â€” use `data.table` grouped operations (`[, .(max, min, mean), by=...]`) instead of per-row `lapply`.
4. **Process all variables in one pass** if possible, or at least use vectorized column operations.

### Architecture

```
cell_data (data.table, keyed on id + year)
    â†“
edges table: (id, neighbor_id) â€” 1.37M rows, cell-level
    â†“
Join edges Ã— cell_data on (neighbor_id = id, year) â†’ neighbor values
    â†“
Group by (id, year) â†’ compute max, min, mean per variable
    â†“
Join aggregated stats back to cell_data
```

This replaces 6.46M-element R lists and per-row `lapply` with a single vectorized `data.table` merge + grouped aggregation. Expected runtime: **minutes, not days**. Memory: the expanded edge-year table will have ~1.37M edges Ã— 28 years â‰ˆ 38.4M rows, which is manageable at ~16 GB RAM especially with integer/double columns only.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 0. Convert cell_data to data.table if not already
# ---------------------------------------------------------------
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# ---------------------------------------------------------------
# 1. Build a cell-level edge list from the nb object
#    rook_neighbors_unique is a list of length 344,208;
#    id_order is the vector mapping list index â†’ cell id.
# ---------------------------------------------------------------
build_edge_table <- function(id_order, neighbors) {
  # neighbors is an nb object: list of integer vectors (indices into id_order)
  from_ids <- rep(id_order, times = lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)
  # Remove the 0-neighbor sentinel if present (spdep uses 0L for no neighbors)
  valid    <- to_idx > 0L
  edges    <- data.table(
    id          = from_ids[valid],
    neighbor_id = id_order[to_idx[valid]]
  )
  return(edges)
}

edges <- build_edge_table(id_order, rook_neighbors_unique)
cat("Edge table rows:", nrow(edges), "\n")
# Expected: ~1,373,394

# ---------------------------------------------------------------
# 2. Vectorized neighbor-stat computation for all variables at once
# ---------------------------------------------------------------
compute_all_neighbor_features <- function(cell_data, edges, neighbor_source_vars) {

  # Subset cell_data to only the columns we need for the join
  join_cols <- c("id", "year", neighbor_source_vars)
  nbr_vals <- cell_data[, ..join_cols]

  # Rename 'id' to 'neighbor_id' so we can join on the neighbor side
  setnames(nbr_vals, "id", "neighbor_id")

  # Merge: for every (id, year) pair, look up each neighbor's values in that year
  # edges has (id, neighbor_id); nbr_vals has (neighbor_id, year, var1, var2, ...)
  # Result: one row per (id, year, neighbor_id) with the neighbor's variable values
  setkey(nbr_vals, neighbor_id, year)
  setkey(edges, neighbor_id)

  # This is the big join â€” ~1.37M edges Ã— 28 years â‰ˆ 38.4M rows
  expanded <- edges[nbr_vals, on = "neighbor_id", allow.cartesian = TRUE, nomatch = NULL]
  # expanded now has columns: id, neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2

  # ---------------------------------------------------------------
  # 3. Grouped aggregation: max, min, mean per (id, year) per variable
  # ---------------------------------------------------------------
  # Build aggregation expressions dynamically
  agg_exprs <- list()
  for (v in neighbor_source_vars) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("nbr_max_", v)]]  <- bquote(as.double(max(.(v_sym), na.rm = TRUE)))
    agg_exprs[[paste0("nbr_min_", v)]]  <- bquote(as.double(min(.(v_sym), na.rm = TRUE)))
    agg_exprs[[paste0("nbr_mean_", v)]] <- bquote(mean(.(v_sym), na.rm = TRUE))
  }

  # Convert to a single call
  agg_call <- as.call(c(as.name("list"), agg_exprs))

  cat("Aggregating neighbor stats...\n")
  stats <- expanded[, eval(agg_call), by = .(id, year)]

  # Replace Inf/-Inf (from max/min on all-NA) with NA
  for (col_name in names(stats)) {
    if (col_name %in% c("id", "year")) next
    vals <- stats[[col_name]]
    set(stats, i = which(is.infinite(vals)), j = col_name, value = NA_real_)
  }

  return(stats)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

stats <- compute_all_neighbor_features(cell_data, edges, neighbor_source_vars)

# ---------------------------------------------------------------
# 4. Join the aggregated neighbor features back to cell_data
# ---------------------------------------------------------------
# Remove any pre-existing neighbor columns to avoid duplication
nbr_col_names <- setdiff(names(stats), c("id", "year"))
existing_nbr_cols <- intersect(names(cell_data), nbr_col_names)
if (length(existing_nbr_cols) > 0) {
  cell_data[, (existing_nbr_cols) := NULL]
}

setkey(cell_data, id, year)
setkey(stats, id, year)

cell_data <- stats[cell_data, on = .(id, year)]
# Left join: all original rows preserved; cells with no neighbors get NA (correct)

cat("Done. cell_data now has", ncol(cell_data), "columns and", nrow(cell_data), "rows.\n")

# ---------------------------------------------------------------
# 5. Predict with the existing trained Random Forest (unchanged)
# ---------------------------------------------------------------
# The trained RF model object is preserved as-is.
# Ensure the new neighbor column names match what the model expects.
# If the original code produced columns named e.g. "ntl_nbr_max", "ntl_nbr_min",
# "ntl_nbr_mean", rename accordingly:
#
# Example renaming (adjust to match your model's expected feature names):
# setnames(cell_data, "nbr_max_ntl",  "ntl_nbr_max")
# setnames(cell_data, "nbr_min_ntl",  "ntl_nbr_min")
# setnames(cell_data, "nbr_mean_ntl", "ntl_nbr_mean")
# ... etc. for all 5 variables Ã— 3 stats = 15 columns
#
# Then predict as before:
# cell_data$predicted_gdp <- predict(rf_model, newdata = cell_data)
```

---

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M-iteration `lapply` with `paste` + named vector lookup | Single vectorized `data.table` join |
| **Aggregation** | 6.46M-iteration `lapply` Ã— 5 variables = 32.3M R function calls | One grouped `data.table` aggregation over all variables simultaneously |
| **Data structure** | 6.46M-element R list of integer vectors (~10+ GB) | Flat `data.table` (~38.4M rows, ~2-3 GB) |
| **Estimated runtime** | 86+ hours | **5â€“15 minutes** |
| **RAM peak** | Exceeds 16 GB | ~6â€“10 GB (fits in 16 GB) |
| **Numerical output** | max, min, mean per neighbor set | Identical max, min, mean per neighbor set |
| **Trained RF model** | â€” | Untouched; no retraining |

The key principle: **replace row-level R iteration with vectorized joins and grouped aggregation in `data.table`**, and exploit the fact that the spatial topology is cell-level (344K cells), not cell-year-level (6.46M rows).
END_RESPONSE>>>
