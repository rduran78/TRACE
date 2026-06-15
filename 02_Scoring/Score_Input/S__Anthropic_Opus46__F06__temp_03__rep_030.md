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
run_id: Anthropic_F06_T03_R030
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 30
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F06_Anthropic_temp_03_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows

For every single cell-year row, the function:
1. Looks up the spatial cell's rook neighbors (fast).
2. Constructs character key strings by pasting `id_year` (slow â€” 6.46M string allocations).
3. Performs named-vector lookups via `idx_lookup[neighbor_keys]` â€” this is an **O(k)** hash lookup per row, but the constant factor of character matching in a 6.46M-element named vector is enormous.

The result is a **list of 6.46 million integer vectors**. This alone consumes massive memory (~several GB for the list overhead) and takes hours because of repeated string construction and hash lookups.

### Bottleneck B: `compute_neighbor_stats` â€” O(n) `lapply` over 6.46M rows, repeated 5 times

For each of the 5 source variables, another full pass over 6.46M rows computes `max`, `min`, `mean` of neighbor values. The `lapply` + `do.call(rbind, ...)` pattern on 6.46M elements is extremely slow in base R.

### Why raster focal/kernel operations don't directly apply

The comment in the prompt asks whether raster focal operations are a useful analogy. They are conceptually analogous (a neighborhood summary over a grid), but:
- The data is in **long panel format** (cell Ã— year), not a raster stack.
- The neighbor structure is precomputed as an `nb` object with irregular coastal/boundary cells.
- Focal operations would require reshaping to raster for each year and variable, applying `focal()`, then reshaping back â€” introducing complexity and potential floating-point discrepancies at boundaries.

**The better strategy** is to stay in tabular form but replace the row-level R loops with **vectorized sparse-matrix multiplication and grouped operations**.

---

## 2. Optimization Strategy

### Key Insight: Separate spatial and temporal dimensions

The neighbor relationships are **purely spatial** (they don't change across years). There are only **344,208 cells** with rook neighbors, not 6.46M cell-years. We can:

1. **Build a sparse adjacency matrix `W`** (344,208 Ã— 344,208) from the `nb` object â€” done once.
2. **Reshape each variable into a matrix** of shape (344,208 cells Ã— 28 years).
3. **Compute neighbor stats using sparse matrix operations:**
   - `neighbor_mean = (W %*% X) / (W %*% 1_{non-NA})` â€” sparse matrix multiply, vectorized.
   - `neighbor_max` and `neighbor_min` require iterating over the sparse structure, but can be done efficiently in C++ via a small `Rcpp` function, or approximated with repeated sparse operations. Alternatively, we use `data.table` grouped operations on an edge list.
4. **Join results back** to the long panel.

### Expected speedup

| Component | Current | Optimized |
|---|---|---|
| Neighbor lookup construction | ~hours (6.46M string ops) | ~seconds (sparse matrix from nb) |
| Neighbor stats (per variable) | ~15+ hours | ~seconds (sparse mat-mul) or ~1-2 min (data.table edge-list) |
| Total (5 variables) | 86+ hours | **< 10 minutes** |

---

## 3. Working R Code

```r
# ==============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Replaces: build_neighbor_lookup + compute_neighbor_stats + outer loop
# Preserves: exact numerical results (max, min, mean of rook-neighbor values)
# Preserves: trained Random Forest model (no retraining)
# ==============================================================================

library(data.table)
library(Matrix)

# --------------------------------------------------------------------------
# STEP 0: Ensure cell_data is a data.table with key columns: id, year
# --------------------------------------------------------------------------
cell_data <- as.data.table(cell_data)

# Establish a consistent integer mapping for spatial cell IDs
# id_order is the vector of cell IDs matching rook_neighbors_unique (the nb object)
n_cells <- length(id_order)
id_to_idx <- setNames(seq_len(n_cells), as.character(id_order))

# --------------------------------------------------------------------------
# STEP 1: Build edge list from the nb object (done ONCE)
#
# rook_neighbors_unique is an nb object: a list of length n_cells,
# where each element is an integer vector of neighbor indices (into id_order).
# --------------------------------------------------------------------------
build_edge_list <- function(nb_obj) {
  # Pre-count total edges for pre-allocation
  n_edges <- sum(vapply(nb_obj, function(x) {
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1)))
  
  from_idx <- integer(n_edges)
  to_idx   <- integer(n_edges)
  pos <- 1L
  
  for (i in seq_along(nb_obj)) {
    nbrs <- nb_obj[[i]]
    if (length(nbrs) == 1L && nbrs[1] == 0L) next
    k <- length(nbrs)
    from_idx[pos:(pos + k - 1L)] <- i
    to_idx[pos:(pos + k - 1L)]   <- nbrs
    pos <- pos + k
  }
  
  data.table(from_idx = from_idx, to_idx = to_idx)
}

cat("Building edge list from nb object...\n")
edge_dt <- build_edge_list(rook_neighbors_unique)

# Add actual cell IDs to edge list
edge_dt[, from_id := id_order[from_idx]]
edge_dt[, to_id   := id_order[to_idx]]

cat(sprintf("Edge list: %d directed edges\n", nrow(edge_dt)))

# --------------------------------------------------------------------------
# STEP 2: For each source variable, compute neighbor max/min/mean
#          using a data.table merge-and-aggregate approach.
#
# Strategy:
#   - Extract the (id, year, var) columns from cell_data.
#   - Join edge_dt (from_id -> to_id) with the variable values at to_id,year.
#   - Group by (from_id, year) and compute max, min, mean.
#   - Join results back to cell_data.
# --------------------------------------------------------------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Set key on cell_data for fast joins
setkey(cell_data, id, year)

compute_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {
  cat(sprintf("  Computing neighbor features for: %s\n", var_name))
  
  # Extract only needed columns: the neighbor cell's value
  # We need to look up var_name at (to_id, year) for each edge, for each year.
  val_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setkey(val_dt, id, year)
  
  # Get unique years
  years <- sort(unique(cell_dt$year))
  
  # Cross join edges Ã— years, then look up neighbor values
  # But edges Ã— years = 1.37M Ã— 28 â‰ˆ 38.4M rows â€” manageable in 16GB RAM
  
  # More memory-efficient: process year by year
  results_list <- vector("list", length(years))
  
  for (yi in seq_along(years)) {
    yr <- years[yi]
    
    # Get values for this year
    yr_vals <- val_dt[year == yr, .(id, val)]
    setkey(yr_vals, id)
    
    # Join: for each edge, get the neighbor's value in this year
    # edge_dt has (from_id, to_id); we want val at to_id
    edge_vals <- yr_vals[edge_dt[, .(from_id, to_id)], on = .(id = to_id), nomatch = NA]
    # edge_vals now has columns: id (=to_id), val, from_id
    
    # Aggregate by from_id
    agg <- edge_vals[!is.na(val), .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ), by = from_id]
    
    agg[, year := yr]
    results_list[[yi]] <- agg
  }
  
  results <- rbindlist(results_list)
  setnames(results, "from_id", "id")
  
  # Create proper column names matching original pipeline
  max_col  <- paste0(var_name, "_max")
  min_col  <- paste0(var_name, "_min")
  mean_col <- paste0(var_name, "_mean")
  setnames(results, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
  
  setkey(results, id, year)
  
  return(results)
}

cat("Computing neighbor features for all source variables...\n")
t0 <- Sys.time()

for (var_name in neighbor_source_vars) {
  feat_dt <- compute_neighbor_features_fast(cell_data, edge_dt, var_name)
  
  # Remove old columns if they exist (in case of re-run)
  max_col  <- paste0(var_name, "_max")
  min_col  <- paste0(var_name, "_min")
  mean_col <- paste0(var_name, "_mean")
  
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) {
      cell_data[, (col) := NULL]
    }
  }
  
  # Merge back to cell_data
  cell_data <- feat_dt[cell_data, on = .(id, year)]
}

t1 <- Sys.time()
cat(sprintf("Done. Total time: %.1f minutes\n", as.numeric(difftime(t1, t0, units = "mins"))))

# --------------------------------------------------------------------------
# STEP 3: Predict with the pre-trained Random Forest (unchanged)
# --------------------------------------------------------------------------
# The trained RF model object is used as-is. No retraining.
# Example (adjust to your actual model object and prediction call):
#
#   predictions <- predict(rf_model, newdata = cell_data)
#
# The 15 new neighbor feature columns (5 vars Ã— 3 stats) have the same
# names and identical numerical values as the original implementation,
# so the RF predictions are exactly preserved.
# --------------------------------------------------------------------------
```

---

## 4. Why This Preserves Exact Results

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | The `max()`, `min()`, `mean()` calls operate on the identical set of non-NA rook-neighbor values per (cell, year). The grouping key `(from_id, year)` exactly replicates the original `neighbor_lookup` logic. |
| **NA handling** | `edge_vals[!is.na(val), ...]` mirrors the original `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]`. Cells with zero valid neighbors get `NA` for all three stats (via the left join â€” they simply won't appear in `agg`, so the join leaves `NA`). |
| **Trained RF model** | Untouched. The 15 derived columns have identical names and values, so `predict(rf_model, newdata = ...)` produces identical output. |

## 5. Performance Estimate

| Step | Time (est.) | Memory (est.) |
|---|---|---|
| Edge list construction | < 5 sec | ~50 MB |
| Per-variable, per-year join + aggregate | ~10 sec/var | ~500 MB peak |
| Total (5 variables, 28 years) | **3â€“8 minutes** | < 4 GB peak |

This reduces the runtime from **86+ hours to under 10 minutes** on the same 16 GB laptop.
END_RESPONSE>>>
