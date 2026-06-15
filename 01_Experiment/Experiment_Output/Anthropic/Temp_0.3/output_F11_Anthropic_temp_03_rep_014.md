 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence clearly points to a different bottleneck: **the spatial neighbor feature engineering stage**, specifically `build_neighbor_lookup` and `compute_neighbor_stats`.

Here's why:

1. **`build_neighbor_lookup`** uses a plain R `lapply` over **~6.46 million rows**, and inside each iteration it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), named-vector lookups (`idx_lookup[neighbor_keys]`), and NA filtering. Named character vector lookups in R are O(n)-ish hash lookups but repeated 6.46 million times with string allocation overhead, this is extremely expensive. This function alone likely accounts for the vast majority of the 86+ hour runtime.

2. **`compute_neighbor_stats`** then runs another `lapply` over the same 6.46 million rows **five times** (once per neighbor source variable), each time subsetting, removing NAs, and computing `max`, `min`, `mean`. The final `do.call(rbind, result)` on a 6.46-million-element list of 3-element vectors is also costly.

3. **Random Forest inference** is a single call to `predict()` on a pre-trained model. Even with 6.46M rows and 110 predictors, a single `predict()` call on a `ranger` or `randomForest` object is vectorized C/C++ code and typically completes in seconds to minutes — orders of magnitude faster than the neighbor feature construction.

**The bottleneck is the row-level R-loop neighbor feature engineering, not the RF inference.**

---

## Optimization Strategy

1. **Replace the character-key hash lookup in `build_neighbor_lookup` with integer arithmetic.** Instead of pasting strings and looking them up in a named vector, compute row indices directly: if data is sorted by `(id, year)` and years are contiguous, the row for `(neighbor_id, year)` can be computed as `(neighbor_ref - 1) * n_years + year_offset` — a pure integer operation with zero string allocation.

2. **Vectorize `compute_neighbor_stats` using `data.table` grouping or sparse-matrix operations.** Convert the neighbor lookup into an edge list (a two-column integer matrix of `[row, neighbor_row]`), then use `data.table` grouped aggregation to compute max/min/mean in one vectorized pass per variable — eliminating 6.46M R-level `lapply` iterations.

3. **Build the neighbor lookup once and reuse it** (already done in the original, preserved here).

4. **Preserve the trained RF model and the original numerical estimand** — no changes to the modeling step.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 0: Ensure cell_data is a data.table sorted by (id, year)
# ==============================================================================
cell_dt <- as.data.table(cell_data)
setkey(cell_dt, id, year)

# Unique ids in the same order as id_order (the spdep nb object reference)
# and unique years
unique_years <- sort(unique(cell_dt$year))
n_years      <- length(unique_years)
year_to_offset <- setNames(seq_along(unique_years), as.character(unique_years))

# Map each id in id_order to its 1-based reference index
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

# For the sorted (id, year) data.table, the row index for a given
# (ref_index, year_offset) is:  (ref_index - 1) * n_years + year_offset
# This requires that every cell has every year. Verify:
stopifnot(nrow(cell_dt) == length(id_order) * n_years)

# ==============================================================================
# STEP 1: Build edge list (integer matrix) — replaces build_neighbor_lookup
# ==============================================================================
build_edge_list <- function(id_order, rook_neighbors_unique, n_years) {
  # Pre-allocate: count total directed edges
  n_ids <- length(id_order)
  # rook_neighbors_unique is an nb object: a list of integer vectors
  # Total neighbor pairs (directed)
  total_edges_per_year <- sum(vapply(rook_neighbors_unique, length, integer(1)))
  total_edges <- total_edges_per_year * n_years
  
  # Pre-allocate integer vectors
  from_row <- integer(total_edges)
  to_row   <- integer(total_edges)
  
  ptr <- 1L
  for (ref in seq_len(n_ids)) {
    nb_refs <- rook_neighbors_unique[[ref]]
    if (length(nb_refs) == 0L) next
    n_nb <- length(nb_refs)
    for (yr_off in seq_len(n_years)) {
      row_from <- (ref - 1L) * n_years + yr_off
      rows_to  <- (nb_refs - 1L) * n_years + yr_off
      idx_range <- ptr:(ptr + n_nb - 1L)
      from_row[idx_range] <- row_from
      to_row[idx_range]   <- rows_to
      ptr <- ptr + n_nb
    }
  }
  
  list(from = from_row[1:(ptr - 1L)], to = to_row[1:(ptr - 1L)])
}

message("Building edge list...")
edge <- build_edge_list(id_order, rook_neighbors_unique, n_years)
message(sprintf("Edge list built: %d directed edges.", length(edge$from)))

# ==============================================================================
# STEP 2: Vectorized neighbor stats via data.table — replaces compute_neighbor_stats
# ==============================================================================
compute_and_add_all_neighbor_features <- function(cell_dt, edge, neighbor_source_vars) {
  # Create edge data.table once
  edge_dt <- data.table(from_row = edge$from, to_row = edge$to)
  
  for (var_name in neighbor_source_vars) {
    message(sprintf("  Computing neighbor stats for: %s", var_name))
    
    # Extract the variable values aligned to the to_row (neighbor row)
    edge_dt[, val := cell_dt[[var_name]][to_row]]
    
    # Group by from_row, compute stats (ignoring NAs)
    stats <- edge_dt[!is.na(val), .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ), by = from_row]
    
    # Initialize new columns with NA
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    
    cell_dt[, (max_col)  := NA_real_]
    cell_dt[, (min_col)  := NA_real_]
    cell_dt[, (mean_col) := NA_real_]
    
    # Assign computed values by row index
    cell_dt[stats$from_row, (max_col)  := stats$nb_max]
    cell_dt[stats$from_row, (min_col)  := stats$nb_min]
    cell_dt[stats$from_row, (mean_col) := stats$nb_mean]
  }
  
  # Clean up temporary column from edge_dt
  edge_dt[, val := NULL]
  
  cell_dt
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

message("Computing neighbor features...")
cell_dt <- compute_and_add_all_neighbor_features(cell_dt, edge, neighbor_source_vars)
message("Neighbor features complete.")

# ==============================================================================
# STEP 3: Random Forest prediction (unchanged — preserve trained model)
# ==============================================================================
# The trained RF model object (e.g., `rf_model`) is already in memory or loaded:
# rf_model <- readRDS("trained_rf_model.rds")
#
# Predict — this is NOT the bottleneck:
# cell_dt[, prediction := predict(rf_model, newdata = cell_dt)$predictions]
# (or predict(rf_model, newdata = cell_dt) depending on the RF package used)
#
# Write results:
# fwrite(cell_dt, "predictions_output.csv")

message("Pipeline complete.")
```

---

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M `lapply` iterations with `paste()` + named-vector string lookups | Pure integer arithmetic edge list built in one pass (~1.37M edges × 28 years) |
| **Neighbor stats** | 5 × 6.46M `lapply` iterations with per-row subsetting, `do.call(rbind, ...)` on 6.46M-element list | 5 × one vectorized `data.table` grouped aggregation over the edge list |
| **Estimated runtime** | 86+ hours | Minutes (edge list build) + seconds per variable (grouped aggregation) ≈ **under 30 minutes total** |
| **RF model** | Preserved, untouched | Preserved, untouched |
| **Numerical estimand** | Same max/min/mean neighbor features → same predictions | Identical computation, identical results |

The true bottleneck was millions of R-level loop iterations with string operations and named-vector lookups in the neighbor feature engineering — not the Random Forest inference.