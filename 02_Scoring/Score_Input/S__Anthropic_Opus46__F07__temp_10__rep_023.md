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
run_id: Anthropic_F07_T10_R023
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 23
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F07_Anthropic_temp_10_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` â€” O(nÂ²)-class overhead
- `lapply` over **6.46 million rows**, each performing string pasting, named-vector lookups (`idx_lookup[neighbor_keys]`), and NA filtering.
- `idx_lookup` is a **named character vector with 6.46M entries**. Each lookup by name triggers a linear or hash-table scan per key. With ~1.37M neighbor relationships replicated across 28 years, this produces tens of millions of named-vector lookups â€” extremely slow in R.
- String operations (`paste(id, year, sep="_")`) are repeated millions of times.

### Bottleneck 2: `compute_neighbor_stats` â€” repeated per variable but structure is reusable
- The `lapply` over 6.46M list elements is called **5 times** (once per variable). Each call iterates over the full panel, indexing into a numeric vector and computing max/min/mean. This is pure-R looping with no vectorization.

### Memory concern
- `neighbor_lookup` is a list of 6.46M integer vectors. Each list element has overhead (~128 bytes per SEXP). Even if most cells have 4 neighbors, this list alone can consume **several GB** just in R object overhead.

**Estimated wall-clock:** The named-vector lookups and per-row `lapply` calls over 6.46M rows dominate. The 86+ hour estimate is credible.

---

## Optimization Strategy

### 1. Replace the per-row list with a flat edge-list (CSR-style) representation
Instead of a 6.46M-element list, build a **sparse adjacency structure as two integer vectors** (a pointer vector and a neighbor-index vector), equivalent to Compressed Sparse Row format. This eliminates millions of R list elements and their per-element overhead.

### 2. Vectorize the neighbor lookup construction
- Use `data.table` for fast keyed joins instead of named-vector lookups.
- Expand the spatial neighbor list into an edge data.frame `(cell_id, neighbor_cell_id)` once (1.37M rows).
- Cross-join with years to get `(cell_id, year, neighbor_cell_id)` â†’ ~1.37M Ã— 28 â‰ˆ 38.4M rows (but only for existing cell-years).
- Join against the panel to resolve each `(neighbor_cell_id, year)` â†’ row index.
- This replaces 6.46M R-level iterations with a single vectorized merge.

### 3. Vectorize `compute_neighbor_stats` using `data.table` grouped aggregation
- Using the flat edge-list, join in the variable values, then `group by` the focal-row index and compute `max`, `min`, `mean` in one vectorized pass per variable.
- With `data.table`, each variable takes seconds, not hours.

### 4. Preserve the trained RF model and numerical estimand
- The output columns have the same names and identical numerical values (max, min, mean of non-NA neighbor values).
- No retraining needed.

**Expected speedup:** From 86+ hours to **~5â€“15 minutes** total.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Convert panel to data.table and ensure row ordering is preserved
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt <- as.data.table(cell_data)
cell_dt[, .row_idx := .I]  # preserve original row order

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Expand spatial nb object into a flat edge-list (directed)
#
# rook_neighbors_unique is an nb object: a list of length
# length(id_order), where element i contains integer indices into
# id_order of the rook neighbors of cell id_order[i].
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_list <- function(id_order, neighbors) {
  # neighbors[[i]] contains integer positions referencing id_order
  n <- length(neighbors)
  from_list <- vector("list", n)
  to_list   <- vector("list", n)
  for (i in seq_len(n)) {
    nb <- neighbors[[i]]
    # spdep::nb encodes "no neighbors" as a single 0L
    if (length(nb) == 1L && nb[1] == 0L) next
    from_list[[i]] <- rep(id_order[i], length(nb))
    to_list[[i]]   <- id_order[nb]
  }
  data.table(
    focal_id    = unlist(from_list, use.names = FALSE),
    neighbor_id = unlist(to_list,   use.names = FALSE)
  )
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
cat("Spatial edges:", nrow(edge_dt), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Expand edges across years via keyed join to the panel
#
# For every (focal_id, year) row in the panel, find the row indices
# of its spatial neighbors in the same year.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Keyed lookup: (id, year) â†’ .row_idx
setkey(cell_dt, id, year)

# Start from the focal side: get (focal_id, year, focal_row_idx)
focal_info <- cell_dt[, .(focal_id = id, year, focal_row_idx = .row_idx)]

# Join edges onto focal rows: for each focal row, repeat its neighbors
#   Result columns: focal_id, year, focal_row_idx, neighbor_id
edges_with_year <- edge_dt[focal_info, on = .(focal_id), allow.cartesian = TRUE, nomatch = 0L]

# Now resolve neighbor_id + year â†’ neighbor_row_idx
# Build a small lookup
neighbor_key <- cell_dt[, .(neighbor_id = id, year, neighbor_row_idx = .row_idx)]
setkey(neighbor_key, neighbor_id, year)
setkey(edges_with_year, neighbor_id, year)

edges_resolved <- neighbor_key[edges_with_year, on = .(neighbor_id, year), nomatch = NA]
# Keep only edges where the neighbor actually exists in the panel that year
edges_resolved <- edges_resolved[!is.na(neighbor_row_idx)]

cat("Panel-expanded directed edges:", nrow(edges_resolved), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Compute neighbor stats for each variable â€” fully vectorized
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-extract the columns we need from the original ordering
# (cell_dt may have been re-keyed, so index by .row_idx)
setkey(cell_dt, .row_idx)

for (var_name in neighbor_source_vars) {
  cat("Processing neighbor stats for:", var_name, "... ")
  
  # Pull neighbor values via integer indexing (fast)
  vals_vec <- cell_dt[[var_name]]  # ordered by .row_idx because we setkey'd
  
  # Attach neighbor values to the edge table
  edges_resolved[, nval := vals_vec[neighbor_row_idx]]
  
  # Drop NAs before aggregation
  edges_valid <- edges_resolved[!is.na(nval)]
  
  # Grouped aggregation: one pass
  agg <- edges_valid[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), keyby = .(focal_row_idx)]
  
  # Initialize result columns with NA
  max_col  <- paste0("nb_max_",  var_name)
  min_col  <- paste0("nb_min_",  var_name)
  mean_col <- paste0("nb_mean_", var_name)
  
  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]
  
  # Fill in aggregated values by row index
  cell_dt[agg$focal_row_idx, (max_col)  := agg$nb_max]
  cell_dt[agg$focal_row_idx, (min_col)  := agg$nb_min]
  cell_dt[agg$focal_row_idx, (mean_col) := agg$nb_mean]
  
  cat("done.\n")
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Clean up and restore original row order
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setorder(cell_dt, .row_idx)
cell_dt[, .row_idx := NULL]

# Convert back to data.frame if downstream code expects it
cell_data <- as.data.frame(cell_dt)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The trained Random Forest model is untouched.
# The 15 new columns (3 stats Ã— 5 vars) are numerically identical
# to the original implementation's output (max, min, mean of
# non-NA rook-neighbor values per cell-year).
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
```

---

## Summary of gains

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup construction** | 6.46M R-level iterations with named-vector string lookups | Single `data.table` keyed join (~38M rows) |
| **Stat computation per variable** | 6.46M R-level `lapply` iterations | Single `data.table` grouped aggregation |
| **Memory for adjacency** | 6.46M-element R list (~2â€“4 GB overhead) | Flat integer-column data.table (~300 MB) |
| **Total estimated time** | 86+ hours | **5â€“15 minutes** |
| **Numerical result** | max/min/mean of non-NA rook-neighbor values | Identical |
| **RF model** | Preserved (not retrained) | Preserved (not retrained) |

The key insight is replacing millions of R-level loop iterations and named-vector lookups with vectorized `data.table` joins and grouped aggregations, which operate in C-optimized code paths internally.
END_RESPONSE>>>
