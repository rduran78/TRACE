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
run_id: Anthropic_F06_T10_R018
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 18
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F06_Anthropic_temp_10_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` â€” O(nÂ²) row-level `lapply`

This function iterates over **~6.46 million rows**, and for each row it:
1. Looks up the cell's reference index in `id_to_ref`.
2. Retrieves neighbor cell IDs.
3. Constructs string keys (`paste(id, year, sep="_")`).
4. Looks up those keys in `idx_lookup`.

The string-paste-and-match pattern across 6.46M rows is extremely expensive. The named-vector lookup `idx_lookup[neighbor_keys]` has hidden O(n) hashing costs at scale, and the `lapply` over millions of rows in R's interpreted loop is inherently slow.

### Bottleneck 2: `compute_neighbor_stats` â€” Repeated row-level R loops

For each of the 5 variables, `compute_neighbor_stats` runs an `lapply` over 6.46M rows, subsetting a numeric vector by index, removing NAs, and computing `max`, `min`, `mean`. That's **5 Ã— 6.46M = 32.3M** R-level function calls with per-element subsetting.

### Why raster focal/kernel operations aren't directly applicable

Raster focal operations (e.g., `terra::focal`) assume a regular rectangular grid with uniform kernel windows. While the 344K cells *might* be on a regular grid, the panel structure (28 years) and the fact that `rook_neighbors_unique` is a precomputed `spdep::nb` object (which can handle irregular boundaries, missing cells, coastal cells, etc.) means a focal approach could silently change results at edges and boundaries. The instructions say to **preserve the original numerical estimand**, so we must keep the exact neighbor structure. However, the *concept* of vectorized spatial operations inspires the solution: **vectorize using a sparse adjacency matrix and matrix algebra**.

---

## Optimization Strategy

### Strategy: Sparse Matrix Multiplication

The key insight: computing `mean` of neighbor values is equivalent to multiplying a **row-normalized sparse adjacency matrix** by the value vector. Similarly, `max` and `min` can be computed via sparse-matrix-guided grouped operations using `data.table`.

**Step-by-step plan:**

1. **Replace `build_neighbor_lookup`** with construction of a sparse adjacency matrix (cell Ã— cell) using the `Matrix` package, then expand it to the panel level via a year-merge â€” or better, operate at the cell level per year using `data.table` grouping.

2. **Replace `compute_neighbor_stats`** with vectorized `data.table` grouped operations: for each cell-year, join to neighbors and compute `max`, `min`, `mean` in bulk.

3. **All 5 variables at once** in a single join pass, rather than 5 separate loops.

**Expected speedup:** From ~86 hours to ~2â€“10 minutes, because:
- The neighbor join is done once via `data.table` keyed merge (vectorized C code).
- Aggregation uses `data.table`'s optimized `GForce` for `max`, `min`, `mean`.
- No R-level `lapply` over millions of rows.

---

## Working R Code

```r
library(data.table)

# â”€â”€ Step 0: Convert to data.table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# cell_data must have columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: spdep::nb object (list of integer neighbor indices)
# id_order: vector of cell IDs in the order corresponding to rook_neighbors_unique

dt <- as.data.table(cell_data)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# â”€â”€ Step 1: Build an edge list from the nb object â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Each element of rook_neighbors_unique[[i]] gives the indices (into id_order)
# of neighbors of cell id_order[i].

build_edge_list <- function(id_order, nb_obj) {
  # Pre-allocate: count total edges
  n_cells <- length(nb_obj)
  edge_counts <- vapply(nb_obj, function(x) {
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1))
  total_edges <- sum(edge_counts)
  
  from_id <- integer(total_edges)
  to_id   <- integer(total_edges)
  
  pos <- 1L
  for (i in seq_len(n_cells)) {
    nbrs <- nb_obj[[i]]
    if (length(nbrs) == 1L && nbrs[1] == 0L) next
    n <- length(nbrs)
    from_id[pos:(pos + n - 1L)] <- id_order[i]
    to_id[pos:(pos + n - 1L)]   <- id_order[nbrs]
    pos <- pos + n
  }
  
  data.table(focal_id = from_id, neighbor_id = to_id)
}

edges <- build_edge_list(id_order, rook_neighbors_unique)
# edges now has columns: focal_id, neighbor_id
# This should have ~1,373,394 rows

cat("Edge list built:", nrow(edges), "directed edges\n")

# â”€â”€ Step 2: Compute all neighbor stats in one vectorized pass per year â”€â”€â”€â”€â”€â”€â”€â”€
# Strategy: join edges to data by (neighbor_id, year), then group by (focal_id, year)
# to compute max, min, mean for each variable.

# Ensure dt is keyed for fast joins
setkey(dt, id, year)

# Create neighbor data: for each edge and each year, look up the neighbor's values
# We do a single large join: edges Ã— years

# Prepare a "neighbor lookup" table: for each (focal_id, year), we need all
# neighbor values. We achieve this by joining edges with dt on neighbor_id = id.

# Rename for clarity in the join
neighbor_dt <- dt[, c("id", "year", neighbor_source_vars), with = FALSE]
setnames(neighbor_dt, "id", "neighbor_id")
setkey(neighbor_dt, neighbor_id, year)

# Expand edges by year via join: each edge gets one row per year the neighbor exists
# This is the big join: ~1.37M edges Ã— 28 years â‰ˆ 38.4M rows (upper bound)
# In practice, not all cells exist in all years, so it may be less.

setkey(edges, neighbor_id)

# Perform the join: for each (focal_id, neighbor_id) pair, get all years
# where the neighbor has data
cat("Performing edge-year join...\n")
edge_years <- merge(edges, neighbor_dt, by = "neighbor_id", allow.cartesian = TRUE)
# edge_years columns: neighbor_id, focal_id, year, ntl, ec, pop_density, def, usd_est_n2

cat("Edge-year table:", nrow(edge_years), "rows\n")

# â”€â”€ Step 3: Aggregate by (focal_id, year) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

# Use data.table's efficient grouped aggregation
# Construct the j expression
agg_call <- as.call(c(as.name("list"),
  setNames(agg_exprs, agg_names)
))

neighbor_stats <- edge_years[, eval(agg_call), by = .(focal_id, year)]

# Handle Inf/-Inf from max/min on all-NA groups (shouldn't happen if edges exist,
# but be safe)
for (col in agg_names) {
  vals <- neighbor_stats[[col]]
  vals[is.infinite(vals)] <- NA_real_
  set(neighbor_stats, j = col, value = vals)
}

cat("Neighbor stats computed:", nrow(neighbor_stats), "rows,",
    ncol(neighbor_stats) - 2, "new features\n")

# â”€â”€ Step 4: Merge back into original data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setnames(neighbor_stats, "focal_id", "id")
setkey(neighbor_stats, id, year)
setkey(dt, id, year)

dt <- merge(dt, neighbor_stats, by = c("id", "year"), all.x = TRUE)

cat("Final dataset:", nrow(dt), "rows,", ncol(dt), "columns\n")

# â”€â”€ Step 5: Convert back to data.frame if needed for predict() â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data <- as.data.frame(dt)

# â”€â”€ Step 6: Apply the pre-trained Random Forest (unchanged) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The trained model object (e.g., `rf_model`) is used as-is.
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Verification Script

To confirm numerical equivalence with the original implementation on a subset:

```r
# â”€â”€ Verify numerical equivalence on a sample â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
verify_equivalence <- function(cell_data_original, cell_data_optimized,
                                neighbor_source_vars, n_sample = 1000) {
  set.seed(42)
  sample_rows <- sample(nrow(cell_data_original), min(n_sample, nrow(cell_data_original)))
  
  suffixes <- c("max_", "min_", "mean_")
  all_ok <- TRUE
  
  for (v in neighbor_source_vars) {
    for (s in suffixes) {
      col <- paste0("neighbor_", s, v)
      orig <- cell_data_original[[col]][sample_rows]
      optim <- cell_data_optimized[[col]][sample_rows]
      
      both_na <- is.na(orig) & is.na(optim)
      both_val <- !is.na(orig) & !is.na(optim)
      
      if (any(is.na(orig) != is.na(optim))) {
        cat("MISMATCH NA pattern in", col, "\n")
        all_ok <- FALSE
        next
      }
      
      if (any(both_val)) {
        max_diff <- max(abs(orig[both_val] - optim[both_val]))
        if (max_diff > 1e-10) {
          cat("MISMATCH values in", col, "- max diff:", max_diff, "\n")
          all_ok <- FALSE
        }
      }
    }
  }
  
  if (all_ok) cat("All neighbor statistics match within tolerance.\n")
  all_ok
}
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | `lapply` over 6.46M rows, string key matching | One-time edge list build from `nb` object (~1.37M rows) |
| **Stats computation** | 5 Ã— `lapply` over 6.46M rows, per-row subsetting | Single `data.table` grouped aggregation on ~38M-row join |
| **Estimated runtime** | ~86+ hours | ~2â€“10 minutes |
| **Peak RAM** | Moderate (list of 6.46M vectors) | ~4â€“6 GB (edge-year table in memory) |
| **Numerical equivalence** | Baseline | Exact (same `max`, `min`, `mean` over same neighbor sets) |
| **RF model** | Unchanged | Unchanged â€” no retraining |

The optimization replaces two R-level interpreted loops over millions of rows with `data.table`'s vectorized C-level merge and `GForce`-optimized grouped aggregation, yielding orders-of-magnitude speedup while preserving exact numerical results.
END_RESPONSE>>>
