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
run_id: Anthropic_F06_T03_R026
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 26
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F06_Anthropic_temp_03_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows

For every single cell-year row, the function:
1. Looks up the spatial cell's rook neighbors (fine â€” fast hash lookup).
2. Constructs `paste(neighbor_id, year, "_")` key strings for each neighbor.
3. Looks up those keys in `idx_lookup` (a named character vector).

**Problem:** `paste()` and named-vector lookup on 6.46 million rows, each with ~4 neighbors on average, generates ~25M string constructions and hash lookups. The `lapply` loop is single-threaded and allocates millions of small character vectors. The resulting `neighbor_lookup` list itself is a 6.46M-element list of integer vectors â€” enormous memory overhead from list metadata alone.

### Bottleneck B: `compute_neighbor_stats` â€” O(n) `lapply` over 6.46M rows, repeated 5 times

For each of the 5 source variables, the code iterates over all 6.46M rows, subsets a numeric vector by index, removes NAs, and computes `max`, `min`, `mean`. This is called 5 times = 32.3M R-level function calls. Each call has overhead from anonymous function dispatch, `is.na()`, subsetting, and `c()`.

### Combined effect
The two stages together produce an estimated 86+ hours because R's `lapply` over millions of elements with string operations and small-vector allocations is extremely slow.

### Why raster focal/kernel operations don't directly apply
Focal operations (e.g., `terra::focal`) assume a regular rectangular grid with a fixed kernel. Here, the neighbor structure comes from an irregular `spdep::nb` object (rook contiguity on an arbitrary polygon/grid layout), and the data is a panel (cell Ã— year). Focal operations would only work if the spatial grid is perfectly regular **and** you reshape data into a raster stack per year. That's possible but fragile and risks subtle misalignment. The better approach is to vectorize the neighbor computation using sparse matrix algebra, which is both general and fast.

---

## 2. Optimization Strategy

### Strategy: Sparse Matrix Multiplication

The key insight: **max, min, and mean over neighbors can be computed using sparse matrix operations** that process all 6.46M rows simultaneously in compiled C code, eliminating all R-level loops.

**For `mean`:** If `W` is the row-normalized spatial weights matrix (expanded to cell-year dimensions), then `W %*% x` gives the neighbor mean for every row at once.

**For `max` and `min`:** We can't use standard matrix multiplication, but we can use the sparse structure to extract neighbor values in bulk and compute grouped max/min using `data.table` or vectorized operations.

**Step-by-step:**

1. **Build a sparse adjacency matrix** from the `nb` object (344K Ã— 344K) â€” done once.
2. **Expand it to cell-year space** by exploiting the fact that neighbors are looked up within the same year. Instead of building a 6.46M Ã— 6.46M matrix, we compute per-year or use a block-diagonal structure. Even simpler: we work in "spatial ID" space and use `data.table` joins by year.
3. **Vectorized grouped stats** using `data.table`: create an edge list `(row_i, neighbor_j)`, join the variable values by `(neighbor_j, year)`, then compute `max/min/mean` grouped by `(row_i, year)`.

**Expected speedup:** From 86+ hours to **~5â€“15 minutes**. The edge list has ~1.37M spatial edges Ã— 28 years â‰ˆ 38.5M rows â€” easily handled by `data.table` in seconds per variable.

**Memory:** The edge table is ~38.5M rows Ã— 3 columns â‰ˆ ~900 MB. With 16 GB RAM this is feasible, especially since we process one variable at a time.

---

## 3. Working R Code

```r
# ==============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Replaces: build_neighbor_lookup + compute_neighbor_stats loop
# Preserves: exact same numerical results (max, min, mean of rook neighbors)
# Preserves: trained Random Forest model (no retraining)
# ==============================================================================

library(data.table)
library(spdep)

# --------------------------------------------------------------------------
# Step 1: Build spatial edge list from the nb object (done once)
# --------------------------------------------------------------------------
build_spatial_edge_list <- function(id_order, neighbors_nb) {
  # neighbors_nb is an nb object (list of integer index vectors)
  # id_order is the vector of spatial cell IDs in the order matching the nb object
  
  n <- length(neighbors_nb)
  
  # Pre-calculate total edges for pre-allocation
  edge_counts <- vapply(neighbors_nb, length, integer(1))
  total_edges <- sum(edge_counts)
  
  from_id <- integer(total_edges)
  to_id   <- integer(total_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    nb_idx <- neighbors_nb[[i]]
    if (length(nb_idx) > 0 && !(length(nb_idx) == 1 && nb_idx[1] == 0L)) {
      len <- length(nb_idx)
      from_id[pos:(pos + len - 1L)] <- id_order[i]
      to_id[pos:(pos + len - 1L)]   <- id_order[nb_idx]
      pos <- pos + len
    }
  }
  
  # Trim if any nb entries were 0 (no-neighbor sentinel in spdep)
  if (pos <= total_edges) {
    from_id <- from_id[1:(pos - 1L)]
    to_id   <- to_id[1:(pos - 1L)]
  }
  
  data.table(from_id = from_id, to_id = to_id)
}

# --------------------------------------------------------------------------
# Step 2: Compute neighbor stats for one variable (vectorized)
# --------------------------------------------------------------------------
compute_neighbor_stats_fast <- function(cell_dt, edge_dt, var_name) {
  # cell_dt must be a data.table with columns: id, year, <var_name>
  # edge_dt has columns: from_id, to_id
  
  # Create a keyed lookup: for each (to_id, year), the variable value
  lookup <- cell_dt[, .(id, year, val = get(var_name))]
  setkey(lookup, id, year)
  
  # Expand edges across all years via join:
  # For each edge (from_id -> to_id), for each year, get the neighbor's value
  # 
  # Approach: join edge_dt with cell_dt on to_id = id, by year
  # This is effectively a cross of edges Ã— years, but done efficiently via join
  
  # First, get all (from_id, year) combinations that exist in the data
  from_years <- cell_dt[, .(from_id = id, year)]
  
  # Join edges to get (from_id, to_id, year)
  # Use edge_dt and expand by year through the from_id
  edge_year <- edge_dt[from_years, on = .(from_id), allow.cartesian = TRUE, nomatch = 0L]
  # edge_year now has columns: from_id, to_id, year
  
  # Join to get neighbor values
  edge_year[lookup, val := i.val, on = .(to_id = id, year)]
  
  # Remove NA values (as in original code)
  edge_year <- edge_year[!is.na(val)]
  
  # Compute grouped stats
  stats <- edge_year[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = .(from_id, year)]
  
  # Rename columns to match original naming convention
  # Original code produces columns like: ntl_max_neighbor, ntl_min_neighbor, ntl_mean_neighbor
  max_col  <- paste0(var_name, "_max_neighbor")
  min_col  <- paste0(var_name, "_min_neighbor")
  mean_col <- paste0(var_name, "_mean_neighbor")
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
  
  stats
}

# --------------------------------------------------------------------------
# Step 3: Main pipeline
# --------------------------------------------------------------------------
optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique) {
  
  cat("Converting to data.table...\n")
  cell_dt <- as.data.table(cell_data)
  
  cat("Building spatial edge list...\n")
  edge_dt <- build_spatial_edge_list(id_order, rook_neighbors_unique)
  cat(sprintf("  Edge list: %d directed edges\n", nrow(edge_dt)))
  
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Computing neighbor stats for '%s'...\n", var_name))
    t0 <- proc.time()
    
    stats <- compute_neighbor_stats_fast(cell_dt, edge_dt, var_name)
    
    # Determine the new column names
    max_col  <- paste0(var_name, "_max_neighbor")
    min_col  <- paste0(var_name, "_min_neighbor")
    mean_col <- paste0(var_name, "_mean_neighbor")
    
    # Remove old columns if they exist (in case of re-run)
    for (col in c(max_col, min_col, mean_col)) {
      if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
    }
    
    # Left join stats back to cell_dt
    cell_dt <- merge(cell_dt, stats, by.x = c("id", "year"),
                     by.y = c("from_id", "year"), all.x = TRUE)
    
    elapsed <- (proc.time() - t0)[3]
    cat(sprintf("  Done in %.1f seconds\n", elapsed))
  }
  
  cat("All neighbor features computed.\n")
  
  # Convert back to data.frame if the downstream RF predict expects one
  as.data.frame(cell_dt)
}

# --------------------------------------------------------------------------
# Usage (drop-in replacement for the original loop)
# --------------------------------------------------------------------------
# cell_data <- optimize_neighbor_features(cell_data, id_order, rook_neighbors_unique)
#
# # Then proceed with prediction using the pre-trained RF model as before:
# # predictions <- predict(rf_model, newdata = cell_data)
```

---

## 4. Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M R-level `lapply` iterations with `paste` + named-vector lookup | Single vectorized edge-list construction (~1.37M edges), built once in seconds |
| **Stats computation** | 6.46M `lapply` iterations Ã— 5 variables = 32.3M R function calls | 5 `data.table` grouped aggregations on ~38.5M-row join â€” compiled C internals |
| **Estimated runtime** | 86+ hours | **5â€“15 minutes** |
| **Memory** | ~6.46M-element list of integer vectors (high overhead) | ~38.5M-row data.table per variable (peak), freed after merge |
| **Numerical results** | max, min, mean of non-NA rook-neighbor values | **Identical** â€” same operations, same NA handling |
| **RF model** | Pre-trained, not modified | **Preserved** â€” only the feature-engineering step changes |

### Why not raster focal?

While `terra::focal()` with a rook kernel (3Ã—3, center=FALSE, sides only) would be blazingly fast for a single regular raster, this pipeline has complications that make it fragile:
- The grid may not be perfectly regular (the `nb` object suggests arbitrary topology).
- Panel structure (28 years) would require reshaping into 28 raster layers per variable and back.
- Any cell-ID-to-raster-cell misalignment would silently corrupt results.

The sparse `data.table` join approach is nearly as fast, works with any `nb` topology, and is trivially verifiable against the original output.
END_RESPONSE>>>
