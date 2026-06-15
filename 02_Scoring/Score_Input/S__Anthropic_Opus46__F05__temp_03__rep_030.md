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
run_id: Anthropic_F05_T03_R030
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 30
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F05_Anthropic_temp_03_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations over 6.46M Rows

`build_neighbor_lookup` calls `paste()` and performs named-vector lookups **inside a per-row `lapply`** over ~6.46 million rows. Each iteration:

1. **Constructs string keys** (`paste(neighbor_cell_ids, data$year[i], sep = "_")`) â€” repeated string allocation and concatenation.
2. **Looks up those keys in `idx_lookup`** â€” a named character vector of length 6.46M, meaning each lookup is an O(N) hash probe on a massive names vector.

This means the inner loop performs **~6.46M Ã— avg_neighbors Ã— (string paste + hash lookup)** operations. With ~4 rook neighbors on average, that's ~25.8 million string constructions and hash lookups against a 6.46M-entry named vector.

### But the deeper issue is architectural:

The **neighbor relationship is purely spatial** (rook contiguity between grid cells) and **does not vary across years**. The current code re-discovers the same spatial neighbor structure for every year by encoding `(id, year)` pairs as string keys. This is entirely unnecessary.

Furthermore, `compute_neighbor_stats` is already vectorized over the lookup â€” but the lookup itself was built with the most expensive possible method.

### Summary of Redundancies

| Layer | Waste |
|---|---|
| **String keys** | `paste(id, year)` for 6.46M rows + per-row `paste` for neighbors |
| **Named vector lookup** | O(1)-amortized hash but on 6.46M-length names â€” huge constant factor |
| **Year redundancy** | Neighbor topology is year-invariant; year is encoded and decoded for nothing |
| **Row-level lapply in R** | 6.46M R-level function calls with no vectorization |

## Optimization Strategy

**Key insight**: Since neighbors are purely spatial and year-invariant, we can:

1. **Build the neighbor lookup once using integer indexing only** â€” map each cell `id` to its position in `id_order`, then for each cell, store neighbor positions as integer vectors. No strings ever.
2. **Map (cell_position, year) â†’ row index** using an integer matrix instead of a named character vector. With 344,208 cells and 28 years, a `matrix[cell_index, year_index]` of row numbers is only ~9.6M integers (~38 MB).
3. **Vectorize the neighbor-stats computation** using `data.table` or direct vectorized operations, avoiding per-row `lapply` entirely.

This reduces the entire pipeline from **~86 hours to minutes**.

## Working R Code

```r
# =============================================================================
# OPTIMIZED FEATURE CONSTRUCTION
# Preserves the original numerical estimand (max, min, mean of neighbor values)
# Preserves the trained Random Forest model (no retraining needed)
# =============================================================================

library(data.table)

build_neighbor_lookup_fast <- function(data_dt, id_order, rook_neighbors_unique) {
 # -------------------------------------------------------------------------
 # Step 1: Create integer mapping from cell id -> position in id_order
 # -------------------------------------------------------------------------
 n_cells <- length(id_order)
 id_to_pos <- integer(max(id_order))
 id_to_pos[id_order] <- seq_len(n_cells)
 # If ids are not contiguous integers, use a hash:
 # id_to_pos <- setNames(seq_len(n_cells), as.character(id_order))

 # -------------------------------------------------------------------------
 # Step 2: Build neighbor position lists (integer only, year-invariant)
 #   rook_neighbors_unique is an nb object: list of integer vectors
 #   where each element indexes into id_order
 # -------------------------------------------------------------------------
 # nb objects already store neighbor indices into the original ordering,
 # so neighbors[[k]] gives positions in id_order for cell id_order[k].
 # We just need to keep them as-is (they're already integer vectors).
 neighbor_positions <- rook_neighbors_unique  # list of length n_cells

 # Remove the 0-neighbor sentinel used by spdep (0L means no neighbors)
 neighbor_positions <- lapply(neighbor_positions, function(x) {
   x[x != 0L]
 })

 # -------------------------------------------------------------------------
 # Step 3: Build (cell_position, year) -> row_index matrix
 #   Dimensions: n_cells x n_years
 # -------------------------------------------------------------------------
 years <- sort(unique(data_dt$year))
 n_years <- length(years)
 year_to_col <- setNames(seq_len(n_years), as.character(years))

 # Pre-allocate matrix
 row_matrix <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)

 # Fill: for each row in data, find its cell position and year column
 cell_positions <- id_to_pos[data_dt$id]
 year_columns   <- year_to_col[as.character(data_dt$year)]
 row_indices    <- seq_len(nrow(data_dt))

 # Vectorized assignment
 row_matrix[cbind(cell_positions, year_columns)] <- row_indices

 # -------------------------------------------------------------------------
 # Step 4: Build the full neighbor lookup (row -> neighbor rows) vectorized
 #   For each row, get its cell position, get neighbor positions,
 #   then look up rows for the same year.
 # -------------------------------------------------------------------------
 # Instead of building a 6.46M-length list here (expensive),
 # we return the components and compute stats directly.
 # -------------------------------------------------------------------------

 list(
   neighbor_positions = neighbor_positions,
   row_matrix         = row_matrix,
   id_to_pos          = id_to_pos,
   year_to_col        = year_to_col,
   cell_positions     = cell_positions,
   year_columns       = year_columns
 )
}


compute_all_neighbor_stats_fast <- function(data_dt, lookup, var_names) {
 # -------------------------------------------------------------------------
 # Vectorized neighbor statistics computation
 # Strategy: iterate over cells (344K) not cell-years (6.46M).
 # For each cell, gather all neighbor cells, then for each year,
 # compute stats across neighbors using matrix operations.
 # -------------------------------------------------------------------------

 neighbor_positions <- lookup$neighbor_positions
 row_matrix         <- lookup$row_matrix
 n_cells            <- nrow(row_matrix)
 n_years            <- ncol(row_matrix)
 n_rows             <- nrow(data_dt)

 # Pre-extract variable columns as matrices for fast access
 var_list <- lapply(var_names, function(v) data_dt[[v]])
 names(var_list) <- var_names

 # Pre-allocate result columns
 result_cols <- list()
 for (var_name in var_names) {
   result_cols[[paste0(var_name, "_neighbor_max")]]  <- rep(NA_real_, n_rows)
   result_cols[[paste0(var_name, "_neighbor_min")]]  <- rep(NA_real_, n_rows)
   result_cols[[paste0(var_name, "_neighbor_mean")]] <- rep(NA_real_, n_rows)
 }

 # -------------------------------------------------------------------------
 # Main loop: iterate over cells (344K iterations, not 6.46M)
 # For each cell, get its neighbors, then process all years at once
 # -------------------------------------------------------------------------
 cat("Computing neighbor stats for", n_cells, "cells x", n_years, "years\n")
 report_interval <- 50000L

 for (ci in seq_len(n_cells)) {
   if (ci %% report_interval == 0L) {
     cat(sprintf("  Cell %d / %d (%.1f%%)\n", ci, n_cells, 100 * ci / n_cells))
   }

   nb_pos <- neighbor_positions[[ci]]
   if (length(nb_pos) == 0L) next

   # Row indices for this cell across all years: integer vector of length n_years
   this_cell_rows <- row_matrix[ci, ]  # NA where cell-year doesn't exist

   # Row indices for all neighbors across all years: matrix (n_neighbors x n_years)
   nb_row_mat <- row_matrix[nb_pos, , drop = FALSE]

   for (var_name in var_names) {
     vals <- var_list[[var_name]]

     # Extract neighbor values: matrix (n_neighbors x n_years)
     # nb_row_mat contains row indices; NA means missing cell-year
     nb_vals <- matrix(vals[nb_row_mat], nrow = length(nb_pos), ncol = n_years)
     # vals[NA] returns NA, which is correct

     # Compute column-wise (year-wise) stats
     # Using colMeans, etc. but need to handle NAs
     n_valid <- colSums(!is.na(nb_vals))

     col_max  <- apply(nb_vals, 2, function(x) {
       xv <- x[!is.na(x)]; if (length(xv) == 0L) NA_real_ else max(xv)
     })
     col_min  <- apply(nb_vals, 2, function(x) {
       xv <- x[!is.na(x)]; if (length(xv) == 0L) NA_real_ else min(xv)
     })
     col_mean <- colMeans(nb_vals, na.rm = TRUE)
     col_mean[n_valid == 0L] <- NA_real_

     # Assign to result vectors (only where this cell has a row for that year)
     valid_years <- which(!is.na(this_cell_rows))
     target_rows <- this_cell_rows[valid_years]

     result_cols[[paste0(var_name, "_neighbor_max")]][target_rows]  <- col_max[valid_years]
     result_cols[[paste0(var_name, "_neighbor_min")]][target_rows]  <- col_min[valid_years]
     result_cols[[paste0(var_name, "_neighbor_mean")]][target_rows] <- col_mean[valid_years]
   }
 }

 # Bind results to data
 for (col_name in names(result_cols)) {
   data_dt[[col_name]] <- result_cols[[col_name]]
 }

 data_dt
}


# =============================================================================
# EVEN FASTER: Fully vectorized using data.table edge-list approach
# This avoids the 344K cell loop entirely.
# =============================================================================

compute_all_neighbor_stats_vectorized <- function(cell_data, id_order,
                                                  rook_neighbors_unique,
                                                  var_names) {
 # -------------------------------------------------------------------------
 # Step 1: Build edge list from nb object (spatial only, year-invariant)
 # -------------------------------------------------------------------------
 cat("Building edge list...\n")
 edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
   nb <- rook_neighbors_unique[[i]]
   nb <- nb[nb != 0L]
   if (length(nb) == 0L) return(NULL)
   data.table(focal_pos = i, neighbor_pos = nb)
 }))

 # Map positions back to cell ids
 edges[, focal_id    := id_order[focal_pos]]
 edges[, neighbor_id := id_order[neighbor_pos]]
 edges[, c("focal_pos", "neighbor_pos") := NULL]

 cat(sprintf("Edge list: %d directed edges\n", nrow(edges)))

 # -------------------------------------------------------------------------
 # Step 2: Convert cell_data to data.table if not already
 # -------------------------------------------------------------------------
 dt <- as.data.table(cell_data)

 # -------------------------------------------------------------------------
 # Step 3: For each variable, join edges with data to get neighbor values,
 #          then aggregate by (focal_id, year)
 # -------------------------------------------------------------------------
 # Create a slim table for joining: (id, year, var1, var2, ...)
 join_cols <- c("id", "year", var_names)
 dt_slim <- dt[, ..join_cols]

 # Expand edges by year via join:
 #   For each (focal_id, neighbor_id) edge and each year,
 #   look up the neighbor's value.
 # This is: edges Ã— years, but done efficiently via keyed join.

 cat("Joining edges with data...\n")

 # Key the slim data for fast join
 setkey(dt_slim, id, year)

 # Cross join edges with unique years
 years_dt <- data.table(year = sort(unique(dt$year)))
 edge_years <- edges[, CJ_dt := TRUE][
   years_dt[, CJ_dt := TRUE],
   on = "CJ_dt",
   allow.cartesian = TRUE
 ]
 edge_years[, CJ_dt := NULL]

 # Now edge_years has (focal_id, neighbor_id, year) â€” ~38.5M rows
 cat(sprintf("Edge-year combinations: %d rows\n", nrow(edge_years)))

 # Join to get neighbor values
 setkey(edge_years, neighbor_id, year)
 edge_years <- dt_slim[edge_years, on = c("id" = "neighbor_id", "year")]

 # Now edge_years has columns: id (=neighbor_id), year, ntl, ec, ..., focal_id
 # Rename for clarity
 setnames(edge_years, "id", "neighbor_id")

 # -------------------------------------------------------------------------
 # Step 4: Aggregate by (focal_id, year) to get max, min, mean
 # -------------------------------------------------------------------------
 cat("Aggregating neighbor stats...\n")

 agg_exprs <- list()
 for (v in var_names) {
   agg_exprs[[paste0(v, "_neighbor_max")]]  <- parse(text = sprintf("max(%s, na.rm=TRUE)", v))[[1]]
   agg_exprs[[paste0(v, "_neighbor_min")]]  <- parse(text = sprintf("min(%s, na.rm=TRUE)", v))[[1]]
   agg_exprs[[paste0(v, "_neighbor_mean")]] <- parse(text = sprintf("mean(%s, na.rm=TRUE)", v))[[1]]
 }

 # Build the aggregation call
 agg_list <- lapply(agg_exprs, eval, envir = baseenv())

 # More straightforward approach:
 stats <- edge_years[, {
   res <- list()
   for (v in var_names) {
     vals <- get(v)
     vals <- vals[!is.na(vals)]
     if (length(vals) == 0L) {
       res[[paste0(v, "_neighbor_max")]]  <- NA_real_
       res[[paste0(v, "_neighbor_min")]]  <- NA_real_
       res[[paste0(v, "_neighbor_mean")]] <- NA_real_
     } else {
       res[[paste0(v, "_neighbor_max")]]  <- max(vals)
       res[[paste0(v, "_neighbor_min")]]  <- min(vals)
       res[[paste0(v, "_neighbor_mean")]] <- mean(vals)
     }
   }
   res
 }, by = .(focal_id, year)]

 # -------------------------------------------------------------------------
 # Step 5: Join aggregated stats back to the main data
 # -------------------------------------------------------------------------
 cat("Joining stats back to main data...\n")
 setkey(dt, id, year)
 setkey(stats, focal_id, year)
 dt <- stats[dt, on = c("focal_id" = "id", "year")]
 setnames(dt, "focal_id", "id")

 # Handle -Inf/Inf from max/min of empty sets (shouldn't happen after our check, but safety)
 new_cols <- names(stats)[!names(stats) %in% c("focal_id", "year")]
 for (col in new_cols) {
   dt[is.infinite(get(col)), (col) := NA_real_]
 }

 as.data.frame(dt)
}


# =============================================================================
# RECOMMENDED: Hybrid approach â€” cell-loop with matrix ops (memory-safe for 16GB)
# The fully vectorized version creates a ~38.5M row edge-year table which
# at ~5 numeric columns is ~1.5 GB â€” feasible but tight on 16 GB.
# The cell-loop version uses negligible extra memory.
# =============================================================================

# --- USAGE ---

# Convert to data.table for speed (optional, works with data.frame too)
library(data.table)
cell_data <- as.data.table(cell_data)

# Build the fast integer-only lookup
cat("Building fast neighbor lookup...\n")
system.time({
 fast_lookup <- build_neighbor_lookup_fast(
   cell_data, id_order, rook_neighbors_unique
 )
})

# Compute all neighbor stats in one pass
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor statistics...\n")
system.time({
 cell_data <- compute_all_neighbor_stats_fast(
   cell_data, fast_lookup, neighbor_source_vars
 )
})

# The trained Random Forest model is untouched â€” the new columns have
# identical names and identical numerical values as the original code
# would produce (max, min, mean of non-NA neighbor values per cell-year).

# Prediction proceeds as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Performance Comparison

| Metric | Original | Optimized (cell-loop) | Speedup |
|---|---|---|---|
| **Outer loop iterations** | 6.46M (row-level) | 344K (cell-level) | ~19Ã— fewer iterations |
| **String operations** | ~32.3M `paste()` calls + 6.46M key constructions | **Zero** | âˆž |
| **Lookup method** | Named character vector (6.46M entries) | Integer matrix indexing | Orders of magnitude faster |
| **Per-iteration work** | String paste + hash lookup per neighbor | Direct `matrix[i,]` slice + vectorized column ops | ~50-100Ã— faster per iteration |
| **Total variables** | 5 separate passes through `build_neighbor_lookup` | Single pass, all 5 vars together | 5Ã— fewer passes |
| **Estimated runtime** | ~86+ hours | **~5â€“15 minutes** | ~350â€“1000Ã— |
| **Extra memory** | ~500 MB (named vector overhead) | ~38 MB (integer matrix) | ~13Ã— less |

## Why This Preserves the Estimand

The optimized code computes **exactly the same three quantities** for each `(cell, year, variable)` triple:

- `max(neighbor_vals[!is.na(neighbor_vals)])`
- `min(neighbor_vals[!is.na(neighbor_vals)])`
- `mean(neighbor_vals[!is.na(neighbor_vals)])`

The only change is **how we find the neighbor rows** â€” replacing string-key hashing with integer matrix indexing. The neighbor topology, the year matching, and the aggregation functions are identical. The resulting columns have the same names, so the trained Random Forest model can be applied directly without retraining.
END_RESPONSE>>>
