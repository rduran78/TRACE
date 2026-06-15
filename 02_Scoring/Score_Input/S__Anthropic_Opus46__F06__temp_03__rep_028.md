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
run_id: Anthropic_F06_T03_R028
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 28
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F06_Anthropic_temp_03_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) list of per-row lookups via `lapply`

- With ~6.46 million rows, this creates a list of 6.46M elements. Each iteration does string pasting, hash-table lookups (`idx_lookup[neighbor_keys]`), and NA filtering. The string-key approach (`paste(id, year, sep="_")`) is extremely expensive at this scale: it allocates ~6.46M strings just for the keys, then does millions of named-vector lookups (which are O(n) in the worst case for named vectors in R).

### Bottleneck B: `compute_neighbor_stats` â€” `lapply` over 6.46M elements

- For each of the 5 variables, another `lapply` iterates over 6.46M rows, subsetting a numeric vector, removing NAs, and computing `max`, `min`, `mean`. This is repeated 5 times. The `do.call(rbind, result)` on a 6.46M-element list of 3-element vectors is also very slow (millions of small allocations).

### Why it takes 86+ hours

- ~6.46M iterations Ã— 5 variables Ã— expensive per-element R-level operations = billions of interpreted R operations.
- Named vector lookups, string operations, and `lapply` with closures are among the slowest patterns in R.

### Why raster focal/kernel operations are not directly applicable

The data is an **irregular spatial panel** (spdep `nb` object with variable numbers of neighbors per cell), not a regular raster grid. Focal operations assume a fixed rectangular kernel on a regular grid. Using them would either require rasterizing (losing the exact neighbor topology) or padding/masking, which risks altering the numerical results. **We must preserve the original numerical estimand**, so we use the actual neighbor graph but vectorize the computation.

---

## 2. Optimization Strategy

### Strategy: Fully vectorized sparse-matrix approach

1. **Replace string-key lookups with integer indexing.** Build a fast integer mapping from `(id, year)` â†’ row index using `data.table` or `match` on a compound integer key.

2. **Expand the neighbor graph into a cell-year edge list once.** Instead of building a per-row list, create a two-column integer matrix `(from_row, to_row)` representing all directed neighbor-year edges. With ~1.37M directed neighbor pairs Ã— 28 years â‰ˆ ~38.5M edges, this fits easily in RAM.

3. **Compute stats via vectorized group-by operations.** Use `data.table` to group `to_row` values by `from_row`, then compute `max`, `min`, `mean` per group â€” all in C-level `data.table` internals. This replaces 6.46M Ã— 5 = 32.3M `lapply` iterations with 5 fast `data.table` aggregations.

4. **Expected speedup:** From 86+ hours to **minutes** (typically 2â€“10 minutes on a 16 GB laptop).

---

## 3. Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build a fast (id, year) â†’ row_index mapping
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Convert cell_data to data.table (in-place if possible to save RAM)
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Preserve original row order for later
cell_data[, .row_idx := .I]

# Build integer lookup: for every (id, year) â†’ row index
# Using a keyed data.table for O(log n) binary-search joins
id_year_map <- cell_data[, .(id, year, .row_idx)]
setkey(id_year_map, id, year)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Expand neighbor graph into a cell-year edge list
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# id_order is the vector of cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique is an nb object: a list of integer index vectors

# Build directed edge list at the cell level (not yet year-expanded)
# Each element rook_neighbors_unique[[i]] contains indices into id_order
message("Building cell-level edge list...")
n_cells <- length(id_order)

# Pre-compute lengths for pre-allocation
n_lengths <- vapply(rook_neighbors_unique, length, integer(1))
total_edges <- sum(n_lengths)  # ~1.37M

from_cell_idx <- rep(seq_len(n_cells), times = n_lengths)
to_cell_idx   <- unlist(rook_neighbors_unique, use.names = FALSE)

# Map cell indices to actual cell IDs
from_cell_id <- id_order[from_cell_idx]
to_cell_id   <- id_order[to_cell_idx]

# Create cell-level edge data.table
cell_edges <- data.table(from_id = from_cell_id, to_id = to_cell_id)

# Expand across all years via cross join
years <- sort(unique(cell_data$year))
message(sprintf("Expanding %d cell edges across %d years...", nrow(cell_edges), length(years)))

# Memory-efficient expansion: cross join edges Ã— years
cell_edges_expanded <- cell_edges[, .(year = years), by = .(from_id, to_id)]

# Now join to get row indices for both from and to
# Join for 'from' rows
setkey(cell_edges_expanded, from_id, year)
cell_edges_expanded[id_year_map, from_row := i..row_idx, on = .(from_id = id, year = year)]

# Join for 'to' rows (neighbors whose values we read)
setkey(cell_edges_expanded, to_id, year)
cell_edges_expanded[id_year_map, to_row := i..row_idx, on = .(to_id = id, year = year)]

# Drop edges where either side has no matching row
edge_dt <- cell_edges_expanded[!is.na(from_row) & !is.na(to_row), .(from_row, to_row)]

# Clean up intermediates
rm(cell_edges, cell_edges_expanded, from_cell_idx, to_cell_idx,
   from_cell_id, to_cell_id, n_lengths)
gc()

message(sprintf("Edge list ready: %d directed cell-year edges.", nrow(edge_dt)))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Vectorized neighbor stats via data.table grouping
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor stats for: %s", var_name))
  
  # Extract neighbor values via vectorized indexing
  edge_dt[, val := cell_data[[var_name]][to_row]]
  
  # Aggregate: max, min, mean of non-NA neighbor values, grouped by from_row
  stats <- edge_dt[!is.na(val),
                   .(nb_max  = max(val),
                     nb_min  = min(val),
                     nb_mean = mean(val)),
                   keyby = from_row]
  
  # Prepare output columns (NA for rows with no valid neighbors)
  max_col  <- paste0(var_name, "_nb_max")
  min_col  <- paste0(var_name, "_nb_min")
  mean_col <- paste0(var_name, "_nb_mean")
  
  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]
  
  # Assign results back by row index
  cell_data[stats$from_row, (max_col)  := stats$nb_max]
  cell_data[stats$from_row, (min_col)  := stats$nb_min]
  cell_data[stats$from_row, (mean_col) := stats$nb_mean]
  
  message(sprintf("  Done: %s â€” %d rows received neighbor stats.", var_name, nrow(stats)))
}

# Clean up edge value column
edge_dt[, val := NULL]

# Remove helper column
cell_data[, .row_idx := NULL]

message("All neighbor features computed.")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Predict with the pre-trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The trained RF model object (e.g., `rf_model`) is used as-is:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## 4. Summary of Guarantees

| Requirement | How it is preserved |
|---|---|
| **Numerical estimand** | Identical `max`, `min`, `mean` computed over exactly the same rook-neighbor sets and the same variable values. No approximation, no rasterization, no kernel substitution. |
| **Trained RF model** | Untouched. Only the feature-engineering step is optimized. Column names match the original (`{var}_nb_max`, `{var}_nb_min`, `{var}_nb_mean`). |
| **RAM â‰¤ 16 GB** | Edge list: ~38.5M rows Ã— 2 int cols â‰ˆ 308 MB. `cell_data`: ~6.46M Ã— 110 cols â‰ˆ 5.7 GB. Total peak â‰ˆ 8â€“10 GB. Fits comfortably. |
| **Runtime** | Expected **2â€“10 minutes** vs. 86+ hours. The entire computation is pushed into C-level `data.table` group-by operations with no per-row R interpretation. |
END_RESPONSE>>>
