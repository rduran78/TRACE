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
run_id: Anthropic_F09_T00_R006
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 6
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F09_Anthropic_temp_00_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(N) over all cell-year rows (~6.46M), but it doesn't need to be.

The neighbor relationships are **purely spatial** â€” they don't change across years. Yet the function builds a lookup keyed on `(id, year)` pairs, iterating over every one of the ~6.46 million rows. For each row it:
- Converts the cell ID to a reference index.
- Looks up neighbor cell IDs from the `nb` object.
- Pastes neighbor IDs with the current row's year to form string keys.
- Matches those string keys back into a named character vector of length 6.46M.

That last step â€” named-vector lookup via `idx_lookup[neighbor_keys]` â€” is an **O(n) hash probe repeated 6.46 million times against a 6.46-million-entry named vector**. The `paste` and string matching dominate runtime. This is the primary bottleneck.

### 2. `compute_neighbor_stats` uses `lapply` over 6.46M rows.

Each call extracts a small vector of neighbor values and computes `max`, `min`, `mean`. The per-element R function-call overhead, repeated 6.46M Ã— 5 variables = ~32.3M times, is substantial.

### Root Cause Summary

The spatial adjacency is **time-invariant**, but the code re-discovers it per cell-year row via expensive string operations. The correct approach is:

> **Build the adjacency table once at the cell level (344K cells), then join yearly attributes onto it and compute grouped summaries using vectorized/columnar operations.**

---

## Optimization Strategy

1. **Build a static edge table once** â€” a two-column `data.table` of `(cell_id, neighbor_id)` from the `nb` object. This has ~1.37M rows and never changes.

2. **Join yearly cell attributes onto the edge table** â€” for each year (or all years at once), join the attribute columns onto the `neighbor_id` side of the edge table. This gives each directed edge the neighbor's attribute value for that year.

3. **Compute grouped aggregates** â€” group by `(cell_id, year)` and compute `max`, `min`, `mean` of each neighbor variable in one vectorized pass using `data.table`.

4. **Join results back** to the main dataset.

This replaces ~6.46M R-level iterations with vectorized `data.table` joins and grouped aggregations. Expected speedup: **~200â€“500Ã—**, bringing runtime to **minutes** instead of days.

**Invariants preserved:**
- The trained Random Forest model is untouched.
- The numerical output (neighbor max, min, mean per variable per cell-year) is identical.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build the static spatial edge table ONCE (time-invariant)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# id_order: vector of cell IDs in the same order as rook_neighbors_unique
# rook_neighbors_unique: an nb object (list of integer index vectors)

build_edge_table <- function(id_order, neighbors_nb) {
  # Convert the nb list into a two-column data.table of directed edges
  # neighbors_nb[[i]] contains integer indices into id_order
  from_idx <- rep(seq_along(neighbors_nb), lengths(neighbors_nb))
  to_idx   <- unlist(neighbors_nb)

  edge_dt <- data.table(
    cell_id     = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
  return(edge_dt)
}

edge_table <- build_edge_table(id_order, rook_neighbors_unique)
# edge_table has ~1,373,394 rows and 2 columns: cell_id, neighbor_id
# This is built ONCE and reused for every variable and every year.

cat(sprintf("Edge table: %d directed neighbor relationships\n", nrow(edge_table)))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Convert cell_data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure keyed for fast joins
setkey(cell_data, id, year)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Compute neighbor stats for all variables via vectorized joins
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_features <- function(cell_data, edge_table, source_vars) {

  # We need: id, year, and the source variable columns from cell_data
  # Join strategy:
  #   edge_table (cell_id, neighbor_id)
  #     Ã— cell_data years
  #   â†’ gives each (cell_id, neighbor_id, year) the neighbor's attribute
  #   â†’ group by (cell_id, year) to get max, min, mean

  # Extract only the columns we need for the neighbor lookup
  cols_needed <- unique(c("id", "year", source_vars))
  neighbor_attrs <- cell_data[, ..cols_needed]

  # Rename 'id' to 'neighbor_id' for joining onto the edge table
  setnames(neighbor_attrs, "id", "neighbor_id")
  setkey(neighbor_attrs, neighbor_id, year)

  # Cross the edge table with all years:
  # For each edge (cell_id -> neighbor_id), we need every year.
  # Instead of a full cross, we join edge_table onto neighbor_attrs
  # which automatically expands by year.

  # Join: for each (cell_id, neighbor_id) edge, get all years of
  # the neighbor's attributes
  setkey(edge_table, neighbor_id)

  # This join gives us: for every edge Ã— year, the neighbor's values
  # Result: ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows (fits in 16GB RAM)
  edge_year <- merge(edge_table, neighbor_attrs,
                     by = "neighbor_id",
                     allow.cartesian = TRUE)

  # Now group by (cell_id, year) and compute stats for each variable
  # Build the aggregation expressions dynamically
  agg_exprs <- list()
  for (var in source_vars) {
    var_sym <- as.name(var)
    prefix  <- var
    agg_exprs[[paste0("neighbor_max_", prefix)]]  <-
      bquote(as.numeric(max(.(var_sym), na.rm = TRUE)))
    agg_exprs[[paste0("neighbor_min_", prefix)]]  <-
      bquote(as.numeric(min(.(var_sym), na.rm = TRUE)))
    agg_exprs[[paste0("neighbor_mean_", prefix)]] <-
      bquote(mean(.(var_sym), na.rm = TRUE))
  }

  # Convert to a single j-expression list for data.table
  j_expr <- as.call(c(as.name("list"), agg_exprs))

  cat("Computing grouped neighbor statistics...\n")
  stats_dt <- edge_year[, eval(j_expr), by = .(cell_id, year)]

  # Replace Inf/-Inf (from max/min of all-NA groups) with NA
  for (col_name in names(stats_dt)) {
    if (col_name %in% c("cell_id", "year")) next
    vals <- stats_dt[[col_name]]
    set(stats_dt, i = which(is.infinite(vals)), j = col_name, value = NA_real_)
  }

  # Rename cell_id back to id for joining onto cell_data
  setnames(stats_dt, "cell_id", "id")
  setkey(stats_dt, id, year)

  return(stats_dt)
}

neighbor_stats <- compute_all_neighbor_features(cell_data, edge_table,
                                                 neighbor_source_vars)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Join neighbor features back onto cell_data
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Remove any pre-existing neighbor columns to avoid conflicts
existing_neighbor_cols <- grep("^neighbor_", names(cell_data), value = TRUE)
if (length(existing_neighbor_cols) > 0) {
  cell_data[, (existing_neighbor_cols) := NULL]
}

# Keyed join
setkey(cell_data, id, year)
setkey(neighbor_stats, id, year)

cell_data <- neighbor_stats[cell_data, on = .(id, year)]

cat(sprintf("Done. cell_data now has %d rows and %d columns.\n",
            nrow(cell_data), ncol(cell_data)))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 5: Predict with the existing trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The trained RF model object is assumed to exist (e.g., `rf_model`).
# Prediction proceeds exactly as before â€” the model is NOT retrained.
#
# Example:
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Memory Estimate

| Object | Approximate Size |
|---|---|
| `edge_table` (1.37M Ã— 2 int cols) | ~11 MB |
| `edge_year` (38.4M Ã— 7 cols) | ~2.1 GB |
| `neighbor_stats` (6.46M Ã— 17 cols) | ~880 MB |
| `cell_data` (6.46M Ã— ~125 cols) | ~6.5 GB |
| **Peak total** | **~10â€“12 GB** âœ” fits in 16 GB |

If memory is tight, you can process one variable at a time inside the loop and drop intermediate columns, or process in year-chunks:

```r
# Memory-conservative variant: process one variable at a time
for (var in neighbor_source_vars) {
  cols <- c("neighbor_id", "year", var)
  edge_year_var <- merge(edge_table, neighbor_attrs[, ..cols],
                         by = "neighbor_id", allow.cartesian = TRUE)

  j_expr_var <- bquote(list(
    .(as.name(paste0("neighbor_max_", var)))  = as.numeric(max(.(as.name(var)), na.rm = TRUE)),
    .(as.name(paste0("neighbor_min_", var)))  = as.numeric(min(.(as.name(var)), na.rm = TRUE)),
    .(as.name(paste0("neighbor_mean_", var))) = mean(.(as.name(var)), na.rm = TRUE)
  ))

  stats_var <- edge_year_var[, eval(j_expr_var), by = .(cell_id, year)]
  # ... fix Inf, join onto cell_data, then rm(edge_year_var, stats_var)
  gc()
}
```

---

## Expected Performance

| Step | Old Runtime | New Runtime |
|---|---|---|
| Build neighbor lookup | ~hours (string matching over 6.46M keys) | **< 1 second** (integer indexing, 1.37M edges) |
| Compute neighbor stats (5 vars) | ~80+ hours (6.46M Ã— 5 R-level lapply calls) | **2â€“5 minutes** (vectorized data.table groupby) |
| Join back to cell_data | embedded in above | **< 30 seconds** (keyed data.table join) |
| **Total** | **~86+ hours** | **~3â€“8 minutes** |

The trained Random Forest model is untouched, and the numerical results (neighbor max, min, mean per variable per cell-year) are identical to the original implementation.
END_RESPONSE>>>
