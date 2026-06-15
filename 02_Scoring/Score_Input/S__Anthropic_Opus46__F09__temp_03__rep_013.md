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
run_id: Anthropic_F09_T03_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F09_Anthropic_temp_03_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(NÂ²)-like in practice
This function iterates over **all ~6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs `paste()`-based string keys for every neighbor Ã— that row's year (expensive string allocation).
- Performs named-vector lookups via `idx_lookup[neighbor_keys]` â€” which is a **hash-table lookup on character keys repeated ~6.46M Ã— ~4 neighbors â‰ˆ 25.8 million times**.

The real killer is that the neighbor topology is **purely spatial and year-invariant**, yet the function rebuilds the mapping for every cell-year combination. With 344,208 cells Ã— 28 years, it does 28Ã— redundant work on the spatial graph and creates ~6.46 million character-key lookups.

### 2. `compute_neighbor_stats` uses per-row `lapply`
For each of the 5 variables, it loops over 6.46 million entries in R-level `lapply`, extracting and summarizing small vectors. That's ~32.3 million R-level function calls total, with no vectorization.

### Summary of bottlenecks
| Step | Calls | Cost driver |
|---|---|---|
| `build_neighbor_lookup` | 6.46M | `paste()` + named character vector lookup per row |
| `compute_neighbor_stats` | 6.46M Ã— 5 vars | R-level `lapply` with per-row subsetting |

---

## Optimization Strategy

**Core insight:** The neighbor graph is spatial-only and time-invariant. Build it once as a simple integer-indexed adjacency table (cell index â†’ neighbor cell indices), then use a vectorized **join-based** approach to compute neighbor statistics per year.

### Steps:
1. **Build a static spatial adjacency `data.table`** with columns `(cell_idx, neighbor_cell_idx)` from the `nb` object â€” done once, ~1.37M rows.
2. **Add year via cross-join**: For each year, join cell attributes onto the neighbor table by `(neighbor_cell_idx, year)` â€” this is a keyed `data.table` equi-join, fully vectorized in C.
3. **Group-by aggregation**: Group by `(cell_idx, year)` and compute `max`, `min`, `mean` of each neighbor variable â€” one vectorized `data.table` operation per variable.
4. **Join results back** onto the main dataset.

This replaces ~32M R-level function calls with a handful of vectorized `data.table` joins and grouped aggregations.

**Expected speedup:** From ~86 hours to **~2â€“5 minutes** on a 16 GB laptop.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Convert main data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# cell_data must have columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order is the vector of cell IDs in the same order as rook_neighbors_unique
# rook_neighbors_unique is an nb object (list of integer index vectors)

setDT(cell_data)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build the static spatial adjacency table ONCE
#         This is ~1.37M rows, year-invariant.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_adjacency_table <- function(id_order, nb_obj) {
  # nb_obj[[i]] contains integer indices into id_order for neighbors of cell i
  # id_order[i] is the actual cell ID for spatial index i
  n <- length(nb_obj)
  
  from_idx <- rep(seq_len(n), lengths(nb_obj))
  to_idx   <- unlist(nb_obj, use.names = FALSE)
  
  # Remove the 0-neighbor sentinel that spdep uses (nb with no neighbors = 0L)
  valid <- to_idx != 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]
  
  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

adj_table <- build_adjacency_table(id_order, rook_neighbors_unique)
# Result: ~1,373,394 rows with columns (id, neighbor_id)

cat("Adjacency table rows:", nrow(adj_table), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Compute neighbor stats for all variables via vectorized joins
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Create a slim lookup of cell attributes by (id, year)
# Only keep the columns we need for neighbor stats
attr_cols <- c("id", "year", neighbor_source_vars)
cell_attrs <- cell_data[, ..attr_cols]
setkey(cell_attrs, id, year)

# Get unique years
years <- sort(unique(cell_data$year))

compute_all_neighbor_features <- function(adj_table, cell_attrs,
                                          neighbor_source_vars, years) {
  # For each year, we:
  #   1. Cross the adjacency table with that year
  #   2. Join neighbor attributes
  #   3. Aggregate by (id, year)
  
  # Pre-allocate list for results
  year_results <- vector("list", length(years))
  
  # Rename neighbor attribute columns to avoid collision
  neighbor_var_names <- paste0("n_", neighbor_source_vars)
  
  for (yi in seq_along(years)) {
    yr <- years[yi]
    
    # Adjacency table for this year: every spatial edge gets this year
    # ~1.37M rows per year
    adj_yr <- adj_table[, .(id, neighbor_id, year = yr)]
    
    # Join neighbor cell attributes onto adj_yr
    # Key: (neighbor_id, year) matched to cell_attrs (id, year)
    adj_yr <- merge(
      adj_yr,
      cell_attrs,
      by.x = c("neighbor_id", "year"),
      by.y = c("id", "year"),
      all.x = TRUE,
      sort = FALSE
    )
    
    # Aggregate: for each (id, year), compute max/min/mean of each variable
    # Build aggregation expressions dynamically
    agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
      list(
        bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
        bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
        bquote(mean(.(as.name(v)), na.rm = TRUE))
      )
    }), recursive = FALSE)
    
    agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
      paste0("neighbor_", c("max_", "min_", "mean_"), v)
    }))
    
    names(agg_exprs) <- agg_names
    
    # Execute aggregation
    agg_result <- adj_yr[,
      lapply(agg_exprs, eval, envir = .SD),
      by = .(id, year),
      .SDcols = neighbor_source_vars
    ]
    
    year_results[[yi]] <- agg_result
    
    if (yi %% 5 == 0 || yi == length(years)) {
      cat(sprintf("  Processed year %d (%d/%d)\n", yr, yi, length(years)))
    }
  }
  
  rbindlist(year_results)
}

# --- Actually, a cleaner and faster approach avoids the year loop entirely ---
# We can do one big merge. With ~1.37M edges Ã— 28 years = ~38.5M rows,
# this fits comfortably in 16 GB RAM.

compute_all_neighbor_features_vectorized <- function(adj_table, cell_attrs,
                                                      neighbor_source_vars) {
  cat("Building full edge-year table via cross join with years...\n")
  
  # Cross join adjacency with all years: ~38.5M rows
  all_years <- data.table(year = sort(unique(cell_attrs$year)))
  adj_full <- adj_table[, CJ_id := .I]  # just need the cross
  adj_full <- CJ(edge_id = seq_len(nrow(adj_table)),
                 year = all_years$year)
  adj_full[, `:=`(
    id          = adj_table$id[edge_id],
    neighbor_id = adj_table$neighbor_id[edge_id]
  )]
  adj_full[, edge_id := NULL]
  
  cat(sprintf("Edge-year table: %s rows\n", format(nrow(adj_full), big.mark = ",")))
  
  # Join neighbor attributes
  cat("Joining neighbor attributes...\n")
  setkey(cell_attrs, id, year)
  setkey(adj_full, neighbor_id, year)
  
  adj_full <- cell_attrs[adj_full, on = .(id = neighbor_id, year = year)]
  
  # Now adj_full has columns: id (= neighbor_id from original), year,
  # ntl, ec, pop_density, def, usd_est_n2, i.id (= focal cell id)
  # Fix column names â€” the merge flips id references
  setnames(adj_full, "i.id", "focal_id")
  # 'id' column now = neighbor_id, the source vars come from the neighbor
  
  # Aggregate by (focal_id, year)
  cat("Aggregating neighbor statistics...\n")
  
  # Build aggregation call
  agg_list <- list()
  for (v in neighbor_source_vars) {
    agg_list[[paste0("neighbor_max_", v)]]  <-
      substitute(as.numeric(max(VAR, na.rm = TRUE)), list(VAR = as.name(v)))
    agg_list[[paste0("neighbor_min_", v)]]  <-
      substitute(as.numeric(min(VAR, na.rm = TRUE)), list(VAR = as.name(v)))
    agg_list[[paste0("neighbor_mean_", v)]] <-
      substitute(mean(VAR, na.rm = TRUE), list(VAR = as.name(v)))
  }
  
  agg_call <- as.call(c(as.name("list"), agg_list))
  
  result <- adj_full[, eval(agg_call), by = .(focal_id, year)]
  setnames(result, "focal_id", "id")
  
  # Replace Inf/-Inf from max/min of all-NA groups with NA
  inf_cols <- grep("neighbor_max_|neighbor_min_", names(result), value = TRUE)
  for (col in inf_cols) {
    result[is.infinite(get(col)), (col) := NA_real_]
  }
  
  cat("Done.\n")
  result
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Execute and merge back
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cat("Computing neighbor features (vectorized)...\n")
t0 <- proc.time()

neighbor_features <- compute_all_neighbor_features_vectorized(
  adj_table, cell_attrs, neighbor_source_vars
)

elapsed <- (proc.time() - t0)["elapsed"]
cat(sprintf("Neighbor feature computation: %.1f seconds\n", elapsed))

# Remove any old neighbor columns from cell_data to avoid duplication
old_neighbor_cols <- grep("^neighbor_", names(cell_data), value = TRUE)
if (length(old_neighbor_cols) > 0) {
  cell_data[, (old_neighbor_cols) := NULL]
}

# Merge neighbor features onto cell_data
setkey(cell_data, id, year)
setkey(neighbor_features, id, year)
cell_data <- neighbor_features[cell_data, on = .(id, year)]

cat(sprintf("Final dataset: %s rows Ã— %d columns\n",
            format(nrow(cell_data), big.mark = ","), ncol(cell_data)))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Predict with the existing trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The trained model object (e.g., `rf_model`) is preserved as-is.
# Prediction proceeds exactly as before:
#
#   cell_data$rf_prediction <- predict(rf_model, newdata = cell_data)
#
# No retraining. The numerical estimand is identical because the
# neighbor max, min, and mean values are computed with the same
# arithmetic (max, min, mean of non-NA neighbor values), just
# via vectorized joins instead of row-level R loops.
```

---

## Why This Preserves the Original Numerical Estimand

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor graph | `rook_neighbors_unique` (nb object) | Same nb object â†’ same adjacency pairs |
| Neighbor max | `max(neighbor_vals)` after removing NAs | `max(var, na.rm = TRUE)` in grouped `data.table` |
| Neighbor min | `min(neighbor_vals)` after removing NAs | `min(var, na.rm = TRUE)` in grouped `data.table` |
| Neighbor mean | `mean(neighbor_vals)` after removing NAs | `mean(var, na.rm = TRUE)` in grouped `data.table` |
| RF model | Pre-trained, used for `predict()` only | Unchanged â€” no retraining |

The join-and-aggregate approach computes **mathematically identical** values; only the computational mechanism changes from R-level loops to C-level vectorized operations.

## Expected Performance

| Step | Original | Optimized |
|---|---|---|
| Build neighbor lookup | ~hours (6.46M string ops) | ~1 second (integer adjacency table) |
| Compute neighbor stats | ~80+ hours (32.3M R calls) | ~1â€“3 minutes (vectorized join + group-by) |
| **Total** | **~86+ hours** | **~2â€“5 minutes** |

Peak memory for the 38.5M-row edge table with 5 numeric columns is approximately 38.5M Ã— 7 columns Ã— 8 bytes â‰ˆ **2.2 GB**, well within the 16 GB laptop constraint.
END_RESPONSE>>>
