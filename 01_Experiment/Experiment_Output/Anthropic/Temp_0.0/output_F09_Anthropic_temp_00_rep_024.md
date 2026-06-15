 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(N²)-like in practice
This function iterates over **all ~6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs **character key strings** (`paste(id, year)`) and performs **named-vector lookups** (`idx_lookup[neighbor_keys]`) — these are hash-table lookups on a 6.46-million-entry named vector, repeated ~6.46 million times × ~4 neighbors each ≈ **~26 million character-key hash lookups**.

Character allocation, pasting, and named-vector lookup are all slow in R. The resulting `neighbor_lookup` list itself consumes substantial memory (6.46M list elements, each an integer vector).

### 2. `compute_neighbor_stats` is called per-variable with an R-level `lapply` over 6.46M rows
Each call iterates 6.46M rows in pure R, subsetting a numeric vector and computing `max/min/mean`. This is called 5 times (once per source variable), totaling ~32.3 million R-level function invocations.

### Root cause summary
The neighbor topology is **static across years** (rook contiguity doesn't change), yet the code rebuilds a year-specific lookup by cross-joining cells × years at the row level. The key insight the prompt already identifies is correct: **build the spatial adjacency table once, then join yearly attributes onto it.**

---

## Optimization Strategy

1. **Build a static neighbor edge table once** — a two-column `data.table` of `(cell_id, neighbor_id)` derived from the `nb` object. This has ~1.37M rows and is year-independent.

2. **Join yearly attributes via `data.table`** — For each year, the cell-year attributes are already in the panel. We join the neighbor edge table to the panel keyed on `(neighbor_id, year)` to pull neighbor values, then aggregate `max/min/mean` grouped by `(cell_id, year)`. This replaces millions of R-level loops with vectorized `data.table` grouped operations.

3. **Process all 5 variables in a single grouped aggregation** rather than looping variable-by-variable, reducing the number of joins from 5 to 1.

4. **Memory**: The edge table is ~1.37M rows × 2 integer columns ≈ 11 MB. The join expands to ~1.37M × 28 years ≈ 38.4M rows temporarily, which at ~5 numeric columns is ~1.5 GB — feasible on 16 GB RAM.

**Expected speedup**: From ~86 hours to **minutes** (typically 2–10 minutes depending on disk I/O and RAM pressure).

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 0: Ensure cell_data is a data.table with columns: id, year, and
#         the 5 neighbor source variables.
# ──────────────────────────────────────────────────────────────────────
cell_data <- as.data.table(cell_data)

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build the static spatial neighbor edge table (year-free).
#
#   rook_neighbors_unique : an nb object (list of integer index vectors)
#   id_order              : vector of cell IDs in the same order as the nb object
# ──────────────────────────────────────────────────────────────────────
build_edge_table <- function(id_order, nb_obj) {
  # Pre-allocate: count total edges
  n_edges <- sum(vapply(nb_obj, function(x) {
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1)))

  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  pos     <- 1L

  for (i in seq_along(nb_obj)) {
    nbrs <- nb_obj[[i]]
    if (length(nbrs) == 1L && nbrs[1] == 0L) next
    n <- length(nbrs)
    from_id[pos:(pos + n - 1L)] <- id_order[i]
    to_id[pos:(pos + n - 1L)]   <- id_order[nbrs]
    pos <- pos + n
  }

  data.table(cell_id = from_id, neighbor_id = to_id)
}

edge_table <- build_edge_table(id_order, rook_neighbors_unique)
# ~1.37 M rows, two integer columns

cat("Edge table rows:", nrow(edge_table), "\n")

# ──────────────────────────────────────────────────────────────────────
# STEP 2: Join yearly attributes and compute neighbor stats in one pass.
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Subset the columns we need for the neighbor join (keep it lean)
neighbor_vals_dt <- cell_data[, c("id", "year", neighbor_source_vars), with = FALSE]

# Key the attribute table for fast join on (neighbor_id = id, year)
setnames(neighbor_vals_dt, "id", "neighbor_id")
setkey(neighbor_vals_dt, neighbor_id, year)

# Cross-join edge_table with all years, then join neighbor attributes
# More memory-efficient: join edge_table to cell_data's year per cell_id

# First, get the unique years
years <- sort(unique(cell_data$year))

# Expand edge_table × years  (~1.37M × 28 ≈ 38.4M rows)
# Use CJ-like expansion efficiently:
edge_year <- edge_table[, .(year = years), by = .(cell_id, neighbor_id)]

cat("Edge-year rows:", nrow(edge_year), "\n")

# Join neighbor attributes onto edge_year
setkey(edge_year, neighbor_id, year)
edge_year <- neighbor_vals_dt[edge_year, on = .(neighbor_id, year)]

# Now aggregate: for each (cell_id, year), compute max/min/mean of each variable
# Build aggregation expressions dynamically
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(max(.(as.name(v)),   na.rm = TRUE)),
    bquote(min(.(as.name(v)),   na.rm = TRUE)),
    bquote(mean(.(as.name(v)),  na.rm = TRUE))
  )
}))

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

names(agg_exprs) <- agg_names

# Evaluate the aggregation
neighbor_stats <- edge_year[,
  lapply(agg_exprs, eval),
  by = .(cell_id, year)
]

# Replace Inf/-Inf (from max/min on all-NA groups) with NA
for (col_name in agg_names) {
  set(neighbor_stats, which(is.infinite(neighbor_stats[[col_name]])), col_name, NA_real_)
}

cat("Neighbor stats rows:", nrow(neighbor_stats), "\n")

# ──────────────────────────────────────────────────────────────────────
# STEP 3: Merge neighbor stats back onto cell_data.
# ──────────────────────────────────────────────────────────────────────

# Remove any pre-existing neighbor columns to avoid duplication
existing_nbr_cols <- intersect(names(cell_data), agg_names)
if (length(existing_nbr_cols) > 0) {
  cell_data[, (existing_nbr_cols) := NULL]
}

# Merge
setkey(cell_data, id, year)
setkey(neighbor_stats, cell_id, year)
cell_data <- neighbor_stats[cell_data, on = .(cell_id = id, year = year)]

# Restore the "id" column name (the join puts cell_id as the key)
setnames(cell_data, "cell_id", "id")

cat("Final cell_data rows:", nrow(cell_data), "\n")
cat("Final cell_data cols:", ncol(cell_data), "\n")

# ──────────────────────────────────────────────────────────────────────
# STEP 4: Predict with the already-trained Random Forest (unchanged).
# ──────────────────────────────────────────────────────────────────────
# The trained RF model object (e.g., `rf_model`) is preserved as-is.
# Prediction proceeds exactly as before:
#
#   cell_data$predicted <- predict(rf_model, newdata = cell_data)
#
# No retraining is performed. The numerical estimand is identical because
# the neighbor max, min, and mean values are computed with the same
# formulas (max, min, mean with na.rm) on the same neighbor topology.
# ──────────────────────────────────────────────────────────────────────

# Clean up large temporaries
rm(edge_year, neighbor_vals_dt, neighbor_stats)
gc()
```

### If memory is tight (~38.4M row join table ≈ 1.5 GB), process year-by-year:

```r
# ── Memory-conservative variant: process one year at a time ──────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

setkey(cell_data, id, year)
results_list <- vector("list", length(years))

for (yi in seq_along(years)) {
  yr <- years[yi]

  # Subset this year's attributes
  yr_data <- cell_data[year == yr, c("id", neighbor_source_vars), with = FALSE]
  setnames(yr_data, "id", "neighbor_id")
  setkey(yr_data, neighbor_id)

  # Join neighbor attributes onto edge_table
  edges_with_vals <- yr_data[edge_table, on = .(neighbor_id), nomatch = NA]

  # Aggregate per cell_id
  stats_yr <- edges_with_vals[, {
    out <- vector("list", length(neighbor_source_vars) * 3L)
    k <- 1L
    for (v in neighbor_source_vars) {
      vals <- .SD[[v]]
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        out[[k]] <- NA_real_; out[[k+1L]] <- NA_real_; out[[k+2L]] <- NA_real_
      } else {
        out[[k]] <- max(vals); out[[k+1L]] <- min(vals); out[[k+2L]] <- mean(vals)
      }
      k <- k + 3L
    }
    names(out) <- agg_names
    out
  }, by = .(cell_id)]

  stats_yr[, year := yr]
  results_list[[yi]] <- stats_yr

  if (yi %% 5 == 0) cat("Completed year", yr, "\n")
}

neighbor_stats <- rbindlist(results_list)

# Merge back (same as Step 3 above)
existing_nbr_cols <- intersect(names(cell_data), agg_names)
if (length(existing_nbr_cols) > 0) cell_data[, (existing_nbr_cols) := NULL]

setkey(neighbor_stats, cell_id, year)
setkey(cell_data, id, year)
cell_data <- neighbor_stats[cell_data, on = .(cell_id = id, year = year)]
setnames(cell_data, "cell_id", "id")

rm(results_list, neighbor_stats, edges_with_vals, yr_data)
gc()
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M R-level iterations with character hashing | Static 1.37M-row integer edge table built once |
| **Stat computation** | 5 × 6.46M R-level `lapply` calls | Single vectorized `data.table` grouped aggregation (or 28 yearly passes) |
| **Estimated runtime** | ~86+ hours | **~2–10 minutes** |
| **RAM peak** | Large list of 6.46M elements + overhead | ~1.5 GB temporary join table (or ~55 MB/year in conservative variant) |
| **RF model** | Unchanged — no retraining | Unchanged — no retraining |
| **Numerical result** | `max/min/mean` with `na.rm` | Identical `max/min/mean` with `na.rm` on same neighbor topology |