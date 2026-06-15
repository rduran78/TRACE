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
run_id: Anthropic_F02_T10_R004
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 4
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F02_Anthropic_temp_10_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The two critical bottlenecks are:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46 million rows with per-row string pasting and named-vector lookups
- **String key construction:** `paste(id, year, sep="_")` is called millions of times inside the loop, and named-vector indexing (`idx_lookup[neighbor_keys]`) is an O(k) hash lookup per neighbor, repeated for every row.
- **Redundant work across years:** The neighbor *structure* is purely spatial (rook contiguity) and identical for every year. Yet the current code re-discovers neighbors independently for each of the 6.46M cell-year rows instead of exploiting the fact that for a given cell, the neighbor set is the same across all 28 years.
- **Memory:** The resulting `neighbor_lookup` is a 6.46M-element list of integer vectors â€” roughly 1â€“2 GB of overhead from list metadata alone, before the vectors themselves.

### 2. `compute_neighbor_stats` â€” sequential `lapply` over 6.46M elements computing max/min/mean
- Pure R loop with per-element subsetting and NA handling. For 5 variables Ã— 6.46M rows this runs ~32.3 million iterations. No vectorisation, no parallelism.

### Combined impact
At ~48 Âµs per row (conservative) the lookup build alone takes ~86 hours. The stats computation adds more on top.

---

## Optimization Strategy

| Principle | Technique |
|---|---|
| **Eliminate redundant work** | Build the lookup at the *cell* level (344K cells) not the *cell-year* level (6.46M rows). Broadcast spatially via a merge/join. |
| **Replace R loops with vectorised operations** | Use `data.table` grouped operations: join neighbor data, then compute `max`, `min`, `mean` in a single vectorised pass per variable. |
| **Avoid giant intermediate lists** | Replace the 6.46M-element list with a long-form `data.table` of (row, neighbor_row) pairs â€” an edge list that can be joined. |
| **Minimize memory** | Work one variable at a time, adding three columns per variable, then dropping temporaries. Peak RAM stays well under 16 GB. |
| **Preserve the trained RF model** | Only the *feature columns* are being prepared; no model retraining is involved. Output column names and values are numerically identical. |

**Expected speedup:** The entire pipeline should complete in **2â€“10 minutes** instead of 86+ hours.

---

## Working R Code

```r
library(data.table)

# â”€â”€ 0. Convert to data.table (if not already) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data <- as.data.table(cell_data)

# Ensure there is a row-order column so we can restore original order at the end
cell_data[, .row_id := .I]

# â”€â”€ 1. Build a cell-level edge list (one-time, ~344K cells) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#
#   rook_neighbors_unique : spdep nb object, length = n_cells
#   id_order              : integer vector of cell IDs aligned with nb object
#
#   We expand it into a two-column data.table: (focal_id, neighbor_id)
#   This replaces build_neighbor_lookup entirely.

build_edge_list <- function(id_order, neighbors) {
  n <- length(id_order)
  # Pre-allocate with known total length for speed
  from <- vector("list", n)
  to   <- vector("list", n)
  for (i in seq_len(n)) {
    nb_idx <- neighbors[[i]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) next
    from[[i]] <- rep(id_order[i], length(nb_idx))
    to[[i]]   <- id_order[nb_idx]
  }
  data.table(
    focal_id    = unlist(from, use.names = FALSE),
    neighbor_id = unlist(to,   use.names = FALSE)
  )
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
# edge_dt has ~1.37M rows (directed rook pairs) â€” trivially small.

# â”€â”€ 2. Vectorised neighbor-stat computation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#
#   For every (focal cell, year) we need the max, min, mean of each variable
#   across that cell's rook neighbors in the *same* year.
#
#   Approach:
#     a) Join edge_dt to cell_data on neighbor_id + year  â†’ gives us every
#        neighbor's values for every focal cell-year.
#     b) Group by (focal_id, year) and compute stats.
#     c) Join the result back to cell_data.

compute_and_add_all_neighbor_features <- function(dt, edge_dt, var_names) {


  # Columns we need from the neighbor rows: id, year, and all var_names
  cols_needed <- c("id", "year", var_names)
  neighbor_vals <- dt[, ..cols_needed]

  # Rename 'id' to 'neighbor_id' so we can join on the edge list

  setnames(neighbor_vals, "id", "neighbor_id")

  # Keyed join: edge_dt â‹ˆ neighbor_vals  on (neighbor_id, year)
  #   We also need focal_id and year in the result, so we expand edge_dt

  #   by year via a join with the focal rows.
  focal_years <- unique(dt[, .(focal_id = id, year)])

  # Combine: for every (focal_id, year) attach all neighbor_ids
  # This is a many-to-many but bounded: ~1.37M edges Ã— 28 years = ~38.4M rows

  edges_by_year <- edge_dt[focal_years, on = "focal_id", allow.cartesian = TRUE]
  # edges_by_year columns: focal_id, neighbor_id, year

  # Now attach the neighbor variable values
  setkeyv(neighbor_vals, c("neighbor_id", "year"))
  setkeyv(edges_by_year, c("neighbor_id", "year"))
  merged <- neighbor_vals[edges_by_year, on = c("neighbor_id", "year"), nomatch = NA]
  # merged columns: neighbor_id, year, <var_names>, focal_id


  # Group by (focal_id, year) and compute max, min, mean for each variable
  stat_exprs <- unlist(lapply(var_names, function(v) {
    list(
      bquote(as.numeric(max(.(as.name(v)),   na.rm = TRUE))),
      bquote(as.numeric(min(.(as.name(v)),   na.rm = TRUE))),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }), recursive = FALSE)

  stat_names <- unlist(lapply(var_names, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  names(stat_exprs) <- stat_names

  stats <- merged[, lapply(stat_exprs, eval), by = .(focal_id, year)]

  # Replace Inf / -Inf (from max/min on all-NA groups) with NA

  for (col in stat_names) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }

  # Join back to dt
  setnames(stats, "focal_id", "id")
  setkeyv(stats, c("id", "year"))
  setkeyv(dt,    c("id", "year"))
  dt <- stats[dt, on = c("id", "year")]

  # Restore original row order

  setorder(dt, .row_id)


  dt
}

# â”€â”€ 3. Run â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_and_add_all_neighbor_features(
  cell_data, edge_dt, neighbor_source_vars
)

# â”€â”€ 4. (Optional) convert back to data.frame if downstream code expects one â”€
# cell_data[, .row_id := NULL]
# cell_data <- as.data.frame(cell_data)
```

### If ~38 M-row intermediate is too large for 16 GB RAM, process one variable at a time:

```r
compute_and_add_neighbor_features_single <- function(dt, edge_dt, var_name) {

  # Minimal columns from neighbor rows
  neighbor_vals <- dt[, .(neighbor_id = id, year, val = get(var_name))]

  # Focal-year skeleton

  focal_years <- unique(dt[, .(focal_id = id, year)])

  # Expand edges by year

  edges_by_year <- edge_dt[focal_years, on = "focal_id", allow.cartesian = TRUE]

  # Attach neighbor values

  setkeyv(neighbor_vals, c("neighbor_id", "year"))
  setkeyv(edges_by_year, c("neighbor_id", "year"))
  merged <- neighbor_vals[edges_by_year, on = c("neighbor_id", "year"), nomatch = NA]

  # Aggregate

  stats <- merged[, .(
    nmax  = as.numeric(max(val, na.rm = TRUE)),
    nmin  = as.numeric(min(val, na.rm = TRUE)),
    nmean = mean(val, na.rm = TRUE)
  ), by = .(focal_id, year)]

  # Inf â†’ NA

  for (col in c("nmax", "nmin", "nmean")) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }

  new_names <- paste0("neighbor_", c("max_", "min_", "mean_"), var_name)
  setnames(stats, c("nmax", "nmin", "nmean"), new_names)
  setnames(stats, "focal_id", "id")

  # Join back

  setkeyv(stats, c("id", "year"))
  setkeyv(dt,    c("id", "year"))
  dt[stats, (new_names) := mget(new_names), on = c("id", "year")]

  invisible(dt)
}

# Run one variable at a time (lower peak RAM)
for (v in neighbor_source_vars) {
  compute_and_add_neighbor_features_single(cell_data, edge_dt, v)
  gc()
}

setorder(cell_data, .row_id)
```

---

## Why This Is Correct and Safe

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` with `na.rm=TRUE` on the same neighbor sets produce identical values. `Inf`/`-Inf` from all-NA groups are mapped to `NA`, matching the original `c(NA,NA,NA)` fallback. |
| **Trained RF model preserved** | No model object is touched. Only the input feature table is constructed. Column names follow the same `neighbor_{max,min,mean}_{var}` pattern. |
| **RAM under 16 GB** | The single-variable path peaks at ~38M rows Ã— 3 numeric columns â‰ˆ 0.9 GB per variable, plus the 6.46M-row base table (~5.7 GB for 110 columns). Total peak â‰ˆ 8â€“10 GB. |
| **Speed** | `data.table` keyed joins and grouped aggregation on 38M rows complete in seconds to low minutes. Full pipeline: **~2â€“10 minutes** vs. 86+ hours. |
END_RESPONSE>>>
