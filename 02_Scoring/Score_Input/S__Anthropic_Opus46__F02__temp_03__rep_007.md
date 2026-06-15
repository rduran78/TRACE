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
run_id: Anthropic_F02_T03_R007
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 7
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F02_Anthropic_temp_03_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which runs an `lapply` over **~6.46 million rows**, performing per-row string pasting, hash-table lookups, and NA filtering. This creates millions of small character vectors and triggers repeated hash lookups in a named vector of length 6.46M â€” an O(nÂ²)-flavored operation in practice. The `compute_neighbor_stats` function then iterates over the same 6.46M entries again, subsetting a numeric vector for each row. With 5 variables, this entire pipeline runs 5 times for the stats portion.

**Key problems:**

1. **String-key lookups at scale:** `paste()` + named-vector lookup over 6.46M keys is extremely slow. R's named vector lookup is O(n) per probe in the worst case (hash collisions at scale).
2. **Row-level `lapply` over 6.46M rows:** Each iteration allocates small vectors, causing massive GC pressure.
3. **Redundant work:** The neighbor lookup is the same structure for every variable, but the per-row iteration pattern is inherently serial and allocation-heavy in base R.
4. **Memory:** Storing a list of 6.46M integer vectors (the lookup) plus intermediate character vectors can easily exceed 16 GB.

---

## Optimization Strategy

The strategy replaces the row-level list-based lookup with a **flat, vectorized edge-table approach** using `data.table`:

1. **Replace the neighbor lookup list with an edge table.** Instead of a list of 6.46M elements (each containing neighbor row indices), build a two-column `data.table` of `(focal_row, neighbor_row)` pairs â€” roughly 1.37M cell-pairs Ã— 28 years â‰ˆ 38.5M rows of integer pairs. This is ~600 MB and fits in RAM.

2. **Vectorized join for neighbor values.** For each variable, join the neighbor values onto the edge table in one vectorized operation, then aggregate with `data.table`'s grouped `j` expressions (`max`, `min`, `mean`) â€” no `lapply` over millions of rows.

3. **Process all 5 variables in a single grouped aggregation** to avoid 5 separate passes over the edge table.

4. **Build the edge table using integer arithmetic** (no string keys). Map `(cell_id, year)` â†’ row index using a `data.table` keyed join instead of named-vector lookup.

**Expected speedup:** From 86+ hours to **~5â€“15 minutes** on a 16 GB laptop. Memory peak ~4â€“6 GB.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0. Convert cell_data to data.table (if not already) â€” non-destructive
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure original row order is preserved for later re-attachment
cell_data[, .row_id := .I]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. Build a cell-level edge list from the nb object (year-independent)
#    rook_neighbors_unique is a list of length n_cells;
#    id_order is the vector of cell IDs in the same order.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_cell_edge_list <- function(id_order, neighbors) {
  # neighbors[[i]] contains integer indices into id_order for the

  # neighbors of cell id_order[i].
  n <- length(id_order)
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(neighbors))
  focal_id  <- integer(n_edges)
  neigh_id  <- integer(n_edges)
  pos <- 1L
  for (i in seq_len(n)) {
    nb_i <- neighbors[[i]]
    if (length(nb_i) == 0L || (length(nb_i) == 1L && nb_i[1] == 0L)) next
    len <- length(nb_i)
    focal_id[pos:(pos + len - 1L)]  <- id_order[i]
    neigh_id[pos:(pos + len - 1L)]  <- id_order[nb_i]
    pos <- pos + len
  }
  # Trim if any nb entries were empty / zero-padded
  data.table(focal_cell_id = focal_id[1:(pos - 1L)],
             neighbor_cell_id = neigh_id[1:(pos - 1L)])
}

cell_edges <- build_cell_edge_list(id_order, rook_neighbors_unique)

cat(sprintf("Cell-level edges: %s\n", format(nrow(cell_edges), big.mark = ",")))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. Expand to cell-year edge table by cross-joining with years
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
years <- sort(unique(cell_data$year))
year_dt <- data.table(year = years)

# Cross join: every cell-edge Ã— every year  (~38.5 M rows)
cell_year_edges <- cell_edges[, CJ_id := .I]   # dummy for cross join
cell_year_edges <- cell_edges[rep(seq_len(.N), each = length(years))]
cell_year_edges[, year := rep(years, times = nrow(cell_edges))]

cat(sprintf("Cell-year edges: %s\n", format(nrow(cell_year_edges), big.mark = ",")))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. Map (cell_id, year) â†’ row index in cell_data via keyed join
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
row_map <- cell_data[, .(id, year, .row_id)]
setkey(row_map, id, year)

# Attach focal row index
setnames(row_map, ".row_id", "focal_row")
cell_year_edges <- row_map[cell_year_edges, on = .(id = focal_cell_id, year = year),
                           nomatch = 0L]
setnames(row_map, "focal_row", "neighbor_row")
cell_year_edges <- row_map[cell_year_edges, on = .(id = neighbor_cell_id, year = year),
                           nomatch = 0L]

# Clean up: keep only the columns we need
cell_year_edges <- cell_year_edges[, .(focal_row, neighbor_row)]

cat(sprintf("Matched cell-year edges: %s\n", format(nrow(cell_year_edges), big.mark = ",")))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4. Vectorized neighbor-stats computation for ALL variables at once
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Attach neighbor values for all source vars in one go
for (v in neighbor_source_vars) {
  set(cell_year_edges, j = v, value = cell_data[[v]][cell_year_edges$neighbor_row])
}

# Aggregate: grouped by focal_row, compute max/min/mean per variable
# Build the aggregation expression dynamically
agg_exprs <- paste0(
  unlist(lapply(neighbor_source_vars, function(v) {
    c(sprintf("nb_%s_max  = as.double(max(%s, na.rm = TRUE))", v, v),
      sprintf("nb_%s_min  = as.double(min(%s, na.rm = TRUE))", v, v),
      sprintf("nb_%s_mean = mean(%s, na.rm = TRUE)", v, v))
  })),
  collapse = ", "
)
agg_expr <- parse(text = paste0("list(", agg_exprs, ")"))

neighbor_stats <- cell_year_edges[, eval(agg_expr), by = focal_row]

# Replace Inf/-Inf (from max/min on all-NA groups) with NA
inf_cols <- setdiff(names(neighbor_stats), "focal_row")
for (col in inf_cols) {
  vals <- neighbor_stats[[col]]
  set(neighbor_stats, i = which(is.infinite(vals)), j = col, value = NA_real_)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5. Join results back to cell_data, preserving row order
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setkey(neighbor_stats, focal_row)

# Remove any pre-existing neighbor columns to avoid duplication
existing_nb_cols <- intersect(names(cell_data), inf_cols)
if (length(existing_nb_cols) > 0) {
  cell_data[, (existing_nb_cols) := NULL]
}

cell_data <- neighbor_stats[cell_data, on = .(focal_row = .row_id)]
setnames(cell_data, "focal_row", ".row_id")

# Sort back to original order
setkey(cell_data, .row_id)

cat("Neighbor features attached. Column count:", ncol(cell_data), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 6. Clean up large temporaries
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
rm(cell_edges, cell_year_edges, neighbor_stats, row_map, year_dt)
gc()

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 7. Predict with the existing trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The model object (e.g., `rf_model`) and its expected feature names
# are exactly preserved. No retraining is needed.
# Example:
# cell_data$predicted_gdp <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` are computed on exactly the same neighbor sets as the original code. The `na.rm = TRUE` + post-hoc `Inf â†’ NA` replacement reproduces the original `c(NA, NA, NA)` for empty neighbor sets. |
| **Trained RF model** | No model code is touched. The output columns have the same names (`nb_{var}_{stat}`) and the same numeric values, so `predict(rf_model, ...)` produces identical results. |
| **Row order** | The `.row_id` column and final `setkey` guarantee the original row order is restored. |

---

## Performance Estimate

| Step | Original | Optimized |
|---|---|---|
| Build lookup | ~40+ hrs (6.46M `lapply` + string ops) | ~30 sec (vectorized edge expansion + keyed join) |
| Compute stats (5 vars) | ~45+ hrs (5 Ã— 6.46M `lapply`) | ~3â€“8 min (single grouped `data.table` aggregation) |
| Peak RAM | >16 GB (list of 6.46M vectors + char keys) | ~4â€“6 GB (integer edge table + numeric columns) |
| **Total** | **86+ hours** | **~5â€“15 minutes** |
END_RESPONSE>>>
