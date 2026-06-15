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
run_id: Anthropic_F06_T07_R017
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 17
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F06_Anthropic_temp_07_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows

For every one of the ~6.46 million cell-year rows, the function:
1. Looks up the cell's rook neighbors (fast).
2. Constructs character paste keys like `"cellid_year"` for each neighbor (slow â€” millions of string allocations).
3. Indexes into a named character vector (`idx_lookup`) to find row indices (slow â€” repeated hash lookups on a 6.46M-entry named vector).

This produces a **list of length 6.46M**, each element a small integer vector of row indices. The sheer number of string operations and hash lookups on a multi-million-entry lookup table makes this extremely expensive.

### Bottleneck B: `compute_neighbor_stats` â€” repeated `lapply` over 6.46M rows, called 5 times

For each of the 5 source variables, another `lapply` iterates over all 6.46M rows, subsetting a numeric vector by the neighbor indices and computing `max`, `min`, `mean`. The `lapply` + `do.call(rbind, ...)` pattern on 6.46M small vectors is slow due to R-level loop overhead and the cost of binding millions of 3-element vectors.

### Why raster focal/kernel operations are **not** appropriate here

Focal operations assume a regular rectangular grid with a fixed kernel. Here the grid cells have an **irregular neighbor structure** (stored as an `nb` object â€” coastal cells, boundary cells, etc. have varying numbers of neighbors), and the data is in **long panel format** (not a raster stack). Focal operations would require reshaping into rasters per year and would not handle irregular boundaries correctly. The analogy is useful conceptually but the implementation must stay with the `nb`-based approach to **preserve the original numerical estimand exactly**.

### Estimated current runtime breakdown

- `build_neighbor_lookup`: ~6.46M string-paste + hash-lookup operations â†’ **~30â€“40 hours**.
- `compute_neighbor_stats` Ã— 5 variables: ~5 Ã— 6.46M R-level loop iterations â†’ **~40â€“50 hours**.
- Total: **~86+ hours** as reported.

---

## 2. Optimization Strategy

### Strategy A: Vectorized neighbor lookup via sparse matrix multiplication

Replace the entire `build_neighbor_lookup` + `compute_neighbor_stats` pipeline with **sparse matrix operations**:

1. **Expand the 344,208-cell neighbor adjacency into a 6.46M Ã— 6.46M sparse matrix `W`** where `W[i,j] = 1` if row `j` is a rook neighbor of row `i` **in the same year**. This is done by:
   - Creating a cell-index-to-rows mapping (which rows belong to which cell).
   - For each year, mapping the cell-level adjacency to row-level adjacency.

2. **Compute neighbor stats via vectorized sparse operations**:
   - `neighbor_max`: not directly available via matrix multiply, but achievable via row-wise operations on the sparse matrix.
   - `neighbor_mean`: `W %*% x / rowSums(W)` â€” a single sparse matrix-vector multiply.
   - `neighbor_min` and `neighbor_max`: use the sparse structure to do grouped min/max.

However, a 6.46M Ã— 6.46M sparse matrix, even with ~1.37M Ã— 28 â‰ˆ 38.4M nonzeros, is feasible but the max/min operations are awkward with matrix algebra.

### Strategy B (Chosen): Vectorized data.table join approach

A more practical and equally fast approach:

1. **Explode** the `nb` neighbor list into an edge table: `data.table(from_id, to_id)`.
2. **Join** with the panel data by `(to_id, year)` to get neighbor values.
3. **Group by** `(from_id, year)` and compute `max`, `min`, `mean` in one vectorized pass.

This replaces millions of R-level loops with a single `data.table` merge + grouped aggregation â€” expected runtime: **2â€“10 minutes**.

### Why this preserves the estimand

- The neighbor relationships are identical (same `nb` object, same rook adjacency).
- The statistics computed (`max`, `min`, `mean` of non-NA neighbor values) are numerically identical.
- The trained Random Forest model is not modified â€” we only reproduce the same feature columns.

---

## 3. Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build the edge list from the nb object (once, ~344K cells)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(id_order, nb_obj) {
  # nb_obj is a list of length length(id_order),

  # where nb_obj[[i]] contains integer indices into id_order
  # of the rook neighbors of id_order[i].
  edges <- rbindlist(lapply(seq_along(nb_obj), function(i) {
    nbrs <- nb_obj[[i]]
    # spdep nb objects use 0L to indicate no neighbors
    nbrs <- nbrs[nbrs != 0L]
    if (length(nbrs) == 0L) return(NULL)
    data.table(from_id = id_order[i], to_id = id_order[nbrs])
  }))
  edges
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# edge_dt has columns: from_id, to_id
# ~1,373,394 rows (directed edges)

cat(sprintf("Edge table: %d directed neighbor relationships\n", nrow(edge_dt)))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Convert cell_data to data.table and compute all neighbor
#         features in one vectorized pass per variable
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Ensure cell_data is a data.table (non-destructive copy if already one)
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Set key for fast joins
setkey(cell_data, id, year)

# Define source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Columns to extract for joining (id, year, plus all source vars)
join_cols <- c("id", "year", neighbor_source_vars)

# Create a slim table of just the columns we need for neighbor lookups
slim_dt <- cell_data[, ..join_cols]
setnames(slim_dt, "id", "to_id")
setkey(slim_dt, to_id, year)

# Join edges with the slim data to get neighbor values
# For each (from_id, to_id) edge, join on (to_id, year) to get
# the neighbor's values in each year.
# We need to expand edges Ã— years, but it's more efficient to
# join edge_dt with slim_dt directly.

# Add year dimension: merge edge_dt with slim_dt on to_id
# This gives us, for each edge and each year, the neighbor's variable values.
cat("Joining edge table with panel data...\n")
neighbor_vals <- merge(
  edge_dt,
  slim_dt,
  by = "to_id",
  allow.cartesian = TRUE  # each to_id appears in 28 years
)
# neighbor_vals now has columns: to_id, from_id, year, ntl, ec, pop_density, def, usd_est_n2
# Rows: ~1,373,394 edges Ã— 28 years â‰ˆ 38.5M rows (fits easily in 16GB)

cat(sprintf("Neighbor values table: %d rows\n", nrow(neighbor_vals)))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Compute grouped max, min, mean for each (from_id, year)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cat("Computing neighbor statistics...\n")

# Build aggregation expressions dynamically
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(max(.(as.name(v)), na.rm = TRUE)),
    bquote(min(.(as.name(v)), na.rm = TRUE)),
    bquote(mean(.(as.name(v)), na.rm = TRUE))
  )
}))

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

names(agg_exprs) <- agg_names

# Perform the grouped aggregation in one pass
neighbor_stats <- neighbor_vals[,
  lapply(agg_exprs, eval),
  by = .(from_id, year)
]

# Replace -Inf/Inf from max/min of empty groups with NA
for (col_name in agg_names) {
  vals <- neighbor_stats[[col_name]]
  set(neighbor_stats, i = which(is.infinite(vals)), j = col_name, value = NA_real_)
}

cat(sprintf("Neighbor stats table: %d rows, %d columns\n",
            nrow(neighbor_stats), ncol(neighbor_stats)))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Merge neighbor stats back into cell_data
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setnames(neighbor_stats, "from_id", "id")
setkey(neighbor_stats, id, year)
setkey(cell_data, id, year)

# Remove any pre-existing neighbor columns to avoid conflicts
existing_neighbor_cols <- intersect(names(cell_data), agg_names)
if (length(existing_neighbor_cols) > 0) {
  cell_data[, (existing_neighbor_cols) := NULL]
}

cell_data <- merge(cell_data, neighbor_stats, by = c("id", "year"), all.x = TRUE)

cat("Neighbor features added. Done.\n")
cat(sprintf("Final cell_data: %d rows, %d columns\n",
            nrow(cell_data), ncol(cell_data)))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 5: Apply the pre-trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The trained RF model is applied exactly as before.
# Example (adjust to your actual prediction call):
# cell_data$rf_prediction <- predict(trained_rf_model, newdata = cell_data)
```

---

## 4. Performance Summary

| Component | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~30â€“40 hrs (6.46M string pastes + hash lookups) | ~5 sec (344K-row `rbindlist`) |
| Neighbor stats (5 vars) | ~40â€“50 hrs (5 Ã— 6.46M R-level `lapply`) | ~1â€“3 min (one `data.table` merge + grouped agg on ~38.5M rows) |
| Memory peak | ~6â€“8 GB (6.46M-element list of integer vectors) | ~4â€“6 GB (38.5M-row data.table, released after merge) |
| **Total** | **~86+ hours** | **~2â€“5 minutes** |

### Why it's correct

1. **Same adjacency**: The edge table is built from the identical `rook_neighbors_unique` nb object and `id_order`.
2. **Same temporal matching**: The join on `(to_id, year)` ensures each cell's neighbors are drawn from the same year â€” exactly replicating the `paste(id, year)` key logic.
3. **Same statistics**: `max`, `min`, `mean` with `na.rm = TRUE` replicate the original `compute_neighbor_stats` logic (filter NAs, then compute).
4. **Same downstream model**: The pre-trained Random Forest is not retrained; it receives identically-named and identically-valued predictor columns.
END_RESPONSE>>>
