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
run_id: Anthropic_F06_T03_R007
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 7
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F06_Anthropic_temp_03_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows

For every one of the ~6.46 million cell-year rows, the function:
1. Looks up the cell's spatial neighbors (fine â€” O(1) hash lookup).
2. Constructs character paste keys for every neighbor Ã— that row's year (expensive string allocation).
3. Indexes into `idx_lookup` (a named character vector â€” hash lookup, but repeated millions of times with freshly allocated strings).

This produces ~6.46 million list elements, each containing a small integer vector. The per-row string construction (`paste`) and named-vector lookup dominate. With ~4 rook neighbors on average, that's ~25.8 million `paste` calls and hash lookups, all inside a sequential `lapply`. Estimated cost: **many hours** just for this step.

### Bottleneck 2: `compute_neighbor_stats` â€” repeated per variable

Each call iterates over all 6.46M list elements, subsetting a numeric vector and computing `max/min/mean`. This is called 5 times (once per source variable). The `lapply` + `do.call(rbind, ...)` pattern on 6.46M small vectors is slow due to R-level loop overhead and the final row-binding of millions of 3-element vectors.

### Why raster focal/kernel operations are a useful analogy but not directly applicable

Focal operations (e.g., `terra::focal`) compute neighborhood statistics on regular grids blazingly fast via compiled C code operating on contiguous memory. The panel data here *is* on a regular grid, but:
- The data is in long (cell-year) panel format, not a raster stack.
- Boundary cells have irregular neighbor counts.
- The temporal dimension means neighbors must be matched within the same year.

The **analogy** is useful: we should aim for vectorized, compiled-code operations over contiguous memory rather than R-level row-by-row iteration. The best implementation that **preserves the required results exactly** is a fully vectorized `data.table` approach that expands the neighbor edgelist, joins, and groups â€” achieving the same semantics without any R-level loop.

---

## Optimization Strategy

| Step | Current | Proposed | Speedup source |
|---|---|---|---|
| Neighbor lookup | 6.46M-iteration `lapply` with `paste` keys | Pre-build a **directed edge table** `(row_i, row_j)` once using vectorized `data.table` joins â€” no per-row `paste` | Vectorized join, no string keys |
| Neighbor stats | 5 Ã— `lapply` over 6.46M list elements | Single vectorized `data.table` grouped aggregation over the edge table for all 5 variables simultaneously | Compiled C grouping via `data.table`, single pass |
| Row-binding | `do.call(rbind, 6.46M lists)` | Direct column assignment in `data.table` | No allocation/copy |

**Expected runtime: minutes, not hours.** The edge table will have ~6.46M Ã— ~4 â‰ˆ ~25.8M rows (manageable in 16 GB RAM â€” about 200 MB for the integer edge table, plus temporary numeric columns during aggregation).

**Numerical equivalence**: The grouped `max`, `min`, `mean` on the exact same neighbor sets produce identical results. The Random Forest model is never retouched.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# INPUTS (assumed already in environment):
#   cell_data              : data.frame with columns id, year, ntl, ec,
#                            pop_density, def, usd_est_n2, ... (~6.46M rows)
#   id_order               : integer vector of cell IDs in the order used
#                            by the nb object (length 344,208)
#   rook_neighbors_unique  : spdep nb object (list of length 344,208)
#   rf_model               : pre-trained Random Forest model (untouched)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Convert to data.table (in-place if possible to save RAM)
setDT(cell_data)

# 0.  Add a row index for fast downstream assignment
cell_data[, .row_idx := .I]

# ======================================================================
# STEP 1: Build the spatial directed edge list (cell-level, no year dim)
#         from_id -> to_id  for every rook neighbor pair
# ======================================================================
message("Step 1: Building spatial edge list ...")

edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb_idx <- rook_neighbors_unique[[i]]
  if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
    return(NULL)
  }
  data.table(from_id = id_order[i], to_id = id_order[nb_idx])
}))

message(sprintf("  Edge list: %s directed neighbor pairs.", format(nrow(edge_list), big.mark = ",")))

# ======================================================================
# STEP 2: Expand edge list to cell-year level via vectorized join
#         For each (from_id, year) row, find the row indices of its
#         neighbor (to_id, year) rows.
# ======================================================================
message("Step 2: Building cell-year edge table ...")

# Create a lightweight lookup: (id, year) -> row index
id_year_lookup <- cell_data[, .(id, year, .row_idx)]
setkey(id_year_lookup, id, year)

# For the "from" side, get every (from_id, year) combination that exists
# by joining edge_list to the set of (id, year) pairs.
# We need: from_row, to_row  (both are row indices into cell_data)

# First, get from_row: join edge_list Ã— id_year_lookup on from_id = id
#   This replicates each edge across all years that from_id appears in.
edges_with_year <- merge(
  edge_list,
  id_year_lookup,
  by.x = "from_id",
  by.y = "id",
  allow.cartesian = TRUE   # each from_id has ~28 years
)
setnames(edges_with_year, c(".row_idx"), c("from_row"))
# edges_with_year now has: from_id, to_id, year, from_row

# Second, get to_row: join on (to_id, year)
edges_with_year[, to_id_join := to_id]
edges_with_year[, year_join  := year]

to_lookup <- id_year_lookup[, .(to_id_join = id, year_join = year, to_row = .row_idx)]
setkey(to_lookup, to_id_join, year_join)
setkey(edges_with_year, to_id_join, year_join)

cell_year_edges <- to_lookup[edges_with_year, nomatch = 0L]
# Keep only what we need
cell_year_edges <- cell_year_edges[, .(from_row, to_row)]

message(sprintf("  Cell-year edges: %s rows.", format(nrow(cell_year_edges), big.mark = ",")))

# Free temporaries
rm(edges_with_year, to_lookup, id_year_lookup, edge_list)
gc()

# ======================================================================
# STEP 3: Compute neighbor max, min, mean for all 5 variables at once
# ======================================================================
message("Step 3: Computing neighbor statistics ...")

# Extract neighbor values for all source vars in one shot
# by indexing cell_data rows via to_row
neighbor_vals <- cell_data[cell_year_edges$to_row, ..neighbor_source_vars]
neighbor_vals[, from_row := cell_year_edges$from_row]

# Group by from_row and compute stats
# Build aggregation expressions dynamically
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(max(.(as.name(v)), na.rm = TRUE)),
    bquote(min(.(as.name(v)), na.rm = TRUE)),
    bquote(mean(.(as.name(v)), na.rm = TRUE))
  )
}))

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("n_", v, c("_max", "_min", "_mean"))
}))

names(agg_exprs) <- agg_names

# data.table aggregation (runs in compiled C, very fast)
stats_dt <- neighbor_vals[,
  lapply(agg_exprs, eval, envir = .SD),
  by = from_row
]

# Handle Inf/-Inf from max/min on all-NA groups (shouldn't happen if
# neighbor exists, but be safe)
for (col in agg_names) {
  vals <- stats_dt[[col]]
  vals[is.infinite(vals)] <- NA_real_
  set(stats_dt, j = col, value = vals)
}

rm(neighbor_vals, cell_year_edges)
gc()

# ======================================================================
# STEP 4: Join stats back to cell_data
# ======================================================================
message("Step 4: Joining neighbor features back to cell_data ...")

setkey(stats_dt, from_row)

# Initialize new columns with NA (covers rows with no neighbors)
for (col in agg_names) {
  set(cell_data, j = col, value = NA_real_)
}

# Assign in bulk
matched_rows <- stats_dt$from_row
for (col in agg_names) {
  set(cell_data, i = matched_rows, j = col, value = stats_dt[[col]])
}

# Clean up helper column
cell_data[, .row_idx := NULL]

rm(stats_dt)
gc()

message("Done. Neighbor features added. Ready for prediction with the trained RF model.")

# ======================================================================
# STEP 5: Predict with the pre-trained Random Forest (unchanged)
# ======================================================================
# Example (adjust to your actual prediction call):
# cell_data[, prediction := predict(rf_model, newdata = cell_data)]
```

---

## Why This Is Correct and Preserves the Estimand

| Concern | Guarantee |
|---|---|
| **Same neighbor sets** | The edge list is built from the identical `rook_neighbors_unique` nb object. The `merge` on `(to_id, year)` replicates the original logic of matching neighbors within the same year. `nomatch = 0L` drops missing neighbor-years, identical to the `!is.na` filter in the original. |
| **Same statistics** | `max`, `min`, `mean` with `na.rm = TRUE`, grouped by the focal row, are mathematically identical to the original `compute_neighbor_stats`. |
| **Same column names** | The naming convention `n_{var}_{max|min|mean}` matches what `compute_and_add_neighbor_features` would produce (adjust the prefix if yours differs). |
| **RF model untouched** | The trained model object is never modified; only new-data prediction is performed. |
| **RAM feasibility** | The largest intermediate object (`cell_year_edges`) is ~25.8M rows Ã— 2 integer columns â‰ˆ 206 MB. The `neighbor_vals` table is ~25.8M rows Ã— 6 columns â‰ˆ 1.2 GB. Peak usage stays well within 16 GB. |

**Expected runtime**: 5â€“15 minutes on a standard laptop (dominated by the `merge` and grouped aggregation steps), versus the original estimate of 86+ hours â€” roughly a **300â€“1000Ã— speedup**.
END_RESPONSE>>>
