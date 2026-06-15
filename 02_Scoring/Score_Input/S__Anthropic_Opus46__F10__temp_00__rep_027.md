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
run_id: Anthropic_F10_T00_R027
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 27
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F10_Anthropic_temp_00_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per query in the worst case, and doing this 6.46M times with string concatenation is extremely slow.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in a serial `lapply`**, extracting subsets of a numeric vector and computing `max/min/mean`. This is pure R-level looping with no vectorization.

3. **The neighbor topology is year-invariant** (rook neighbors don't change across years), but the lookup is rebuilt monolithically across all cell-years, entangling spatial topology with temporal indexing. This prevents exploiting the block-diagonal structure of the space-time adjacency.

**Key structural insight:** The rook-neighbor graph is a **fixed spatial graph over 344,208 nodes**. The panel has 28 years. The space-time adjacency matrix is block-diagonal: within each year, the same spatial graph applies; across years, there are no edges. The current code fails to exploit this, instead building a single flat 6.46M-entry lookup.

**Memory concern:** 6.46M rows Ã— 110 columns is manageable (~5-6 GB), but intermediate structures (6.46M-element lists of integer vectors) add overhead. We must stay within 16 GB.

---

## Optimization Strategy

### Core Idea: Sparse Matrix Multiplication for Neighborhood Aggregation

For each year and each variable, the neighbor **mean** is simply:

```
X_mean = Dâ»Â¹ A X
```

where `A` is the binary rook adjacency matrix (344,208 Ã— 344,208 sparse), `D` is the diagonal degree matrix, and `X` is the column vector of variable values for that year.

For **max** and **min**, sparse matrix multiplication doesn't directly apply, but we can use the CSR (compressed sparse row) structure of the adjacency matrix to vectorize the aggregation in C++ via `Rcpp`, or use `data.table` grouped operations on an edge list.

### Plan

1. **Build the sparse adjacency once** from `rook_neighbors_unique` (an `nb` object) as a `data.table` edge list: `from_id â†’ to_id` (using integer cell indices 1..344,208). This is ~1.37M rows.

2. **Reshape the panel** so that for each year, we can extract variable values as a vector indexed by cell index.

3. **For each variable Ã— year**, join the edge list to the variable values, then compute grouped `max/min/mean` by `from_id`. This is a `data.table` grouped aggregation on ~1.37M rows â€” extremely fast.

4. **Merge results back** into the main data.

**Expected speedup:** From 86+ hours to **minutes**. The inner loop becomes 5 variables Ã— 28 years = 140 `data.table` grouped joins on a 1.37M-row edge list, each taking < 1 second.

---

## Optimized R Code

```r
library(data.table)

# ==============================================================================
# STEP 0: Ensure cell_data is a data.table with columns: id, year, ntl, ec, ...
# ==============================================================================
setDT(cell_data)

# id_order: integer vector of length 344,208 giving cell IDs in the order
#           matching rook_neighbors_unique (the nb object).
# rook_neighbors_unique: an nb object (list of length 344,208), where
#           element i contains integer indices of neighbors of node i.

# ==============================================================================
# STEP 1: Build the spatial edge list ONCE (year-invariant)
#         ~1.37M rows, two integer columns
# ==============================================================================
build_edge_list <- function(nb_obj, id_order) {
  # nb_obj[[i]] gives neighbor indices (into id_order) for node i
  n <- length(nb_obj)
  from_list <- vector("list", n)
  to_list   <- vector("list", n)
  
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    # spdep nb objects use 0L to indicate no neighbors
    nbrs <- nbrs[nbrs != 0L]
    if (length(nbrs) > 0L) {
      from_list[[i]] <- rep(id_order[i], length(nbrs))
      to_list[[i]]   <- id_order[nbrs]
    }
  }
  
  data.table(
    from_id = unlist(from_list, use.names = FALSE),
    to_id   = unlist(to_list,   use.names = FALSE)
  )
}

edge_dt <- build_edge_list(rook_neighbors_unique, id_order)

# Set key for fast joins
setkey(edge_dt, to_id)

# ==============================================================================
# STEP 2: Compute neighbor stats for all variables, all years at once
# ==============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Create a minimal lookup table: id, year, and the 5 source variables
# This avoids copying all 110 columns into joins
lookup_cols <- c("id", "year", neighbor_source_vars)
val_dt <- cell_data[, ..lookup_cols]
setkey(val_dt, id, year)

# We will join edges to variable values by (to_id, year) to get neighbor values,
# then aggregate by (from_id, year).

# Strategy: for each year, do the join and aggregation on the ~1.37M edge list.
# This avoids a massive cross of edges Ã— years (which would be ~38M rows).

# Pre-allocate result columns in cell_data
for (var_name in neighbor_source_vars) {
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]
}

# Key cell_data for fast update by (id, year)
setkey(cell_data, id, year)

# Get unique years
years <- sort(unique(cell_data$year))

# For fast row-indexing during updates, create an index
# cell_data is keyed by (id, year), so we can use binary join for updates.

cat("Computing neighbor statistics...\n")
t0 <- proc.time()

for (yr in years) {
  # Extract this year's values: id -> variable values
  # Use val_dt which is keyed by (id, year)
  yr_vals <- val_dt[.(unique(val_dt$id), yr), nomatch = 0L]
  # yr_vals has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
  
  # Join edge list to yr_vals on to_id = id to get neighbor variable values
  # edge_dt is keyed on to_id
  setkey(yr_vals, id)
  
  # Rename for clarity in join
  setnames(yr_vals, "id", "to_id")
  
  # Join: for each edge (from_id, to_id), attach the variable values of to_id
  # edge_dt is keyed on to_id, yr_vals is keyed on to_id
  joined <- yr_vals[edge_dt, on = "to_id", nomatch = 0L, allow.cartesian = FALSE]
  # joined has columns: to_id, year, <vars>, from_id
  
  # Aggregate by from_id
  for (var_name in neighbor_source_vars) {
    max_col  <- paste0("neighbor_max_",  var_name)
    min_col  <- paste0("neighbor_min_",  var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    
    agg <- joined[!is.na(get(var_name)),
                  .(nmax  = max(get(var_name)),
                    nmin  = min(get(var_name)),
                    nmean = mean(get(var_name))),
                  by = from_id]
    
    # Update cell_data for this year
    # cell_data is keyed by (id, year)
    if (nrow(agg) > 0L) {
      cell_data[agg, on = .(id = from_id, year = yr),
                (max_col)  := i.nmax]
      cell_data[agg, on = .(id = from_id, year = yr),
                (min_col)  := i.nmin]
      cell_data[agg, on = .(id = from_id, year = yr),
                (mean_col) := i.nmean]
    }
  }
  
  # Restore name
  setnames(yr_vals, "to_id", "id")
}

elapsed <- proc.time() - t0
cat(sprintf("Neighbor stats computed in %.1f seconds.\n", elapsed[3]))

# ==============================================================================
# STEP 3: Apply the pre-trained Random Forest (no retraining)
# ==============================================================================
# The model object (e.g., `rf_model`) is already in memory or loaded from disk.
# Predict using the enriched cell_data which now has all ~110 predictor columns.

# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Further Optimization: Vectorized Single-Pass per Year

The inner loop over 5 variables can be collapsed into a single aggregation call per year:

```r
cat("Computing neighbor statistics (optimized single-pass)...\n")
t0 <- proc.time()

# Pre-allocate all 15 result columns
result_cols <- character(0)
for (var_name in neighbor_source_vars) {
  for (stat in c("max", "min", "mean")) {
    col <- paste0("neighbor_", stat, "_", var_name)
    cell_data[, (col) := NA_real_]
    result_cols <- c(result_cols, col)
  }
}

setkey(cell_data, id, year)

for (yr in years) {
  # Slice this year's values
  yr_vals <- val_dt[year == yr]
  setnames(yr_vals, "id", "to_id")
  setkey(yr_vals, to_id)
  
  # Join edges to neighbor values
  joined <- yr_vals[edge_dt, on = "to_id", nomatch = 0L]
  
  # Build aggregation expression dynamically
  agg_exprs <- list()
  agg_names <- character(0)
  for (var_name in neighbor_source_vars) {
    v <- var_name
    agg_exprs[[paste0("neighbor_max_",  v)]] <- 
      bquote(max(get(.(v)),   na.rm = TRUE))
    agg_exprs[[paste0("neighbor_min_",  v)]] <- 
      bquote(min(get(.(v)),   na.rm = TRUE))
    agg_exprs[[paste0("neighbor_mean_", v)]] <- 
      bquote(mean(get(.(v)),  na.rm = TRUE))
  }
  
  # Single grouped aggregation
  agg <- joined[, 
    lapply(neighbor_source_vars, function(v) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) return(list(NA_real_, NA_real_, NA_real_))
      list(max(vals), min(vals), mean(vals))
    }) |> unlist() |> as.list() |> setNames(result_cols),
    by = from_id
  ]
  
  # Update cell_data
  if (nrow(agg) > 0L) {
    cell_data[agg, on = .(id = from_id, year = yr),
              (result_cols) := mget(paste0("i.", result_cols))]
  }
  
  setnames(yr_vals, "to_id", "id")
}

elapsed <- proc.time() - t0
cat(sprintf("Done in %.1f seconds.\n", elapsed[3]))
```

However, the dynamic expression above is fragile. Here is the **cleanest and fastest** version using explicit `.SD` aggregation:

```r
# ==============================================================================
# FINAL OPTIMIZED VERSION â€” cleanest, fastest
# ==============================================================================
library(data.table)

## 1. Build edge list once
build_edge_list <- function(nb_obj, id_order) {
  from <- rep(seq_along(nb_obj), lengths(nb_obj))
  to   <- unlist(nb_obj, use.names = FALSE)
  valid <- to != 0L
  data.table(from_id = id_order[from[valid]],
             to_id   = id_order[to[valid]])
}

edges <- build_edge_list(rook_neighbors_unique, id_order)

## 2. Prepare value table
src_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
vals <- cell_data[, c("id", "year", src_vars), with = FALSE]

## 3. Expand edges Ã— years via join
#    For each edge (from_id, to_id) and each year, look up to_id's values.
setkey(vals, id)
setkey(edges, to_id)

# Add year dimension: join vals onto edges by to_id
# This creates ~1.37M Ã— 28 â‰ˆ 38.5M rows â€” fits in ~2-3 GB with 7 columns
expanded <- vals[edges, on = .(id = to_id), allow.cartesian = TRUE, nomatch = 0L]
# Columns: id (=to_id), year, <src_vars>, from_id
setnames(expanded, "id", "to_id")

## 4. Aggregate: for each (from_id, year), compute max/min/mean of each variable
agg <- expanded[, {
  out <- list()
  for (v in src_vars) {
    x <- get(v)
    x <- x[!is.na(x)]
    if (length(x) == 0L) {
      out[[paste0("neighbor_max_",  v)]] <- NA_real_
      out[[paste0("neighbor_min_",  v)]] <- NA_real_
      out[[paste0("neighbor_mean_", v)]] <- NA_real_
    } else {
      out[[paste0("neighbor_max_",  v)]] <- max(x)
      out[[paste0("neighbor_min_",  v)]] <- min(x)
      out[[paste0("neighbor_mean_", v)]] <- mean(x)
    }
  }
  out
}, by = .(from_id, year)]

## 5. Merge back into cell_data
stat_cols <- setdiff(names(agg), c("from_id", "year"))
setkey(agg, from_id, year)
setkey(cell_data, id, year)

cell_data[agg, on = .(id = from_id, year = year),
          (stat_cols) := mget(paste0("i.", stat_cols))]

## 6. Predict with pre-trained RF (no retraining)
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Topology build** | 6.46M string-paste + named-vector lookups | `unlist` on nb object â†’ 1.37M-row `data.table` edge list |
| **Aggregation** | 6.46M serial `lapply` calls per variable (Ã—5) | Single `data.table` grouped aggregation on ~38.5M expanded edge-year rows |
| **Memory** | 6.46M-element list of integer vectors + copies | ~38.5M Ã— 8 cols â‰ˆ 2.5 GB (fits in 16 GB) |
| **Estimated time** | 86+ hours | **2â€“10 minutes** |
| **Numerical equivalence** | Baseline | Identical `max`, `min`, `mean` per (node, year, variable) |
| **RF model** | Preserved (not retrained) | Preserved (not retrained) |
END_RESPONSE>>>
