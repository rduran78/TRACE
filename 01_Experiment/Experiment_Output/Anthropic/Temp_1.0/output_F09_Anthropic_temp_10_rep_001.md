 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (estimated 86+ hours) because of three compounding problems:

### 1. Row-level `lapply` over 6.46 million rows in `build_neighbor_lookup`
For every single cell-year row, the function performs string pasting (`paste(id, year, sep="_")`), character-based hash lookups into `idx_lookup`, and subsetting. With ~6.46 million iterations, the overhead of R's interpreted `lapply` loop, repeated `paste` calls, and named-vector lookups (which are O(n) hash probes on a 6.46M-entry character vector) is enormous.

### 2. Row-level `lapply` over 6.46 million rows in `compute_neighbor_stats`
For each row, it subsets a numeric vector by variable-length index vectors, removes NAs, and computes `max/min/mean`. This is called **5 times** (once per neighbor source variable), giving ~32.3 million interpreted R iterations total.

### 3. The neighbor topology is **year-invariant** but is being rebuilt per cell-year
The rook-neighbor structure is purely spatial — cell A's neighbors don't change from year to year. Yet the current code builds a lookup that maps each of the 6.46M cell-year rows to its neighbor cell-year rows. This inflates the problem by a factor of 28 (the number of years). The topology only needs to be expressed once over the 344,208 cells.

---

## Optimization Strategy

**Core insight:** Separate the *time-invariant spatial topology* from the *time-varying attributes*, then use vectorized joins instead of row-level loops.

### Step-by-step plan:

1. **Build a cell-level edge table once** — a two-column `data.table` of `(cell_id, neighbor_cell_id)` from `rook_neighbors_unique`. This has ~1.37M rows and never changes.

2. **For each variable, join yearly attributes onto the edge table** — by joining `cell_data[, .(cell_id, year, value)]` onto the edge table by `neighbor_cell_id` and `year`, every neighbor's attribute value is attached in one vectorized merge.

3. **Aggregate with `data.table` grouping** — group by `(cell_id, year)` and compute `max`, `min`, `mean` in one pass. This replaces millions of `lapply` iterations with a single vectorized `data.table` aggregation.

4. **Join the aggregated stats back** to `cell_data`.

**Complexity reduction:**
- The edge table has ~1.37M rows. After joining with 28 years, the working table is ~1.37M × 28 ≈ 38.4M rows — large but manageable in RAM and extremely fast to aggregate with `data.table`.
- No per-row R-level loops. Everything is vectorized C-level code inside `data.table`.

**Expected speedup:** From 86+ hours to **minutes** (typically 2–10 minutes total depending on I/O).

**Preservation guarantees:**
- The trained Random Forest model is not retouched.
- The numerical results (neighbor max, min, mean) are identical to the originals — same rook topology, same aggregation functions.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 1: Build the time-invariant cell-level edge table (once)
# ==============================================================================
# rook_neighbors_unique : an spdep nb object (list of integer index vectors)
# id_order              : vector of cell IDs in the same order as the nb object
#
# This produces a data.table with columns: cell_id, neighbor_id
# Approximately 1,373,394 rows (directed neighbor pairs)

build_edge_table <- function(id_order, neighbors_nb) {
  # Pre-allocate: count total edges
  n_edges <- sum(vapply(neighbors_nb, function(x) {
    # spdep nb objects use 0L to denote "no neighbors"
    sum(x != 0L)
  }, integer(1)))

  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)

  pos <- 1L
  for (i in seq_along(neighbors_nb)) {
    nb <- neighbors_nb[[i]]
    nb <- nb[nb != 0L]
    if (length(nb) > 0L) {
      n        <- length(nb)
      from_id[pos:(pos + n - 1L)] <- id_order[i]
      to_id[pos:(pos + n - 1L)]   <- id_order[nb]
      pos      <- pos + n
    }
  }

  data.table(cell_id = from_id, neighbor_id = to_id)
}

edge_table <- build_edge_table(id_order, rook_neighbors_unique)

cat(sprintf(
  "Edge table built: %s directed neighbor pairs across %s cells.\n",
  format(nrow(edge_table), big.mark = ","),
  format(length(id_order), big.mark = ",")
))

# ==============================================================================
# STEP 2: Convert cell_data to data.table (if not already)
# ==============================================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure key columns exist
stopifnot(all(c("id", "year") %in% names(cell_data)))

# ==============================================================================
# STEP 3: For each neighbor source variable, compute neighbor stats via join
# ==============================================================================
# This function:
#   - Extracts (id, year, var_value) from cell_data
#   - Joins onto edge_table so each edge row gets the neighbor's value for that year
#   - Aggregates max/min/mean by (cell_id, year)
#   - Joins the result back onto cell_data
#
# Naming convention preserved: <var>_neighbor_max, <var>_neighbor_min, <var>_neighbor_mean

compute_and_add_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {

  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")

  # Remove old columns if they exist (to allow re-runs)
  for (col in c(col_max, col_min, col_mean)) {
    if (col %in% names(cell_dt)) {
      cell_dt[, (col) := NULL]
    }
  }

  # Extract the attribute slice: neighbor cell values keyed by (id, year)
  attr_dt <- cell_dt[, .(neighbor_id = id, year, val = get(var_name))]
  setkey(attr_dt, neighbor_id, year)

  # Join neighbor attribute onto edge table × year
  # edge_dt has (cell_id, neighbor_id); we need to cross with years
  # More efficient: join edge_dt to attr_dt directly
  # For each (cell_id --neighbor_id) edge, pull the neighbor's value for each year
  # Result: one row per (cell_id, neighbor_id, year) with the neighbor's value

  work <- merge(
    edge_dt,
    attr_dt,
    by = "neighbor_id",
    allow.cartesian = TRUE  # each neighbor_id has 28 year-rows
  )
  # work columns: neighbor_id, cell_id, year, val

  # Aggregate by (cell_id, year)
  stats <- work[
    !is.na(val),
    .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ),
    keyby = .(cell_id, year)
  ]

  setnames(stats,
           c("nb_max", "nb_min", "nb_mean"),
           c(col_max, col_min, col_mean))

  # Join back onto cell_data
  cell_dt <- merge(cell_dt, stats,
                   by.x = c("id", "year"),
                   by.y = c("cell_id", "year"),
                   all.x = TRUE)

  return(cell_dt)
}

# ==============================================================================
# STEP 4: Outer loop — compute for all 5 neighbor source variables
# ==============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
for (var_name in neighbor_source_vars) {
  t0 <- proc.time()
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, edge_table, var_name)
  elapsed <- (proc.time() - t0)[["elapsed"]]
  cat(sprintf("  %-15s done in %.1f seconds\n", var_name, elapsed))
}
cat("All neighbor features computed.\n")

# ==============================================================================
# STEP 5: Predict with the already-trained Random Forest (unchanged)
# ==============================================================================
# The trained model object (e.g., `rf_model`) and its expected feature names
# are exactly preserved. No retraining occurs.
#
# Example (adjust to your actual model object and prediction call):
#
# predictions <- predict(rf_model, newdata = cell_data)
# cell_data[, predicted := predictions]
```

---

## Memory Note (16 GB laptop)

The largest intermediate object is the `work` table inside the merge: ~1.37M edges × 28 years = ~38.4M rows × 4 columns ≈ **1.2 GB** at most. This is created and discarded once per variable, so peak memory stays well under 16 GB. If memory is tight, the loop already processes one variable at a time, so only one `work` table exists at any moment. You can add an explicit `rm(work); gc()` inside the function if needed.

---

## Summary

| Aspect | Before | After |
|---|---|---|
| Neighbor lookup granularity | Per cell-year (6.46M R-loop iterations) | Per cell (344K rows, vectorized) |
| Stats computation | 5 × 6.46M `lapply` iterations | 5 × one `data.table` grouped aggregation |
| Key data structure | Named character vector (6.46M entries) | Keyed `data.table` join |
| Estimated wall time | 86+ hours | ~2–10 minutes |
| Trained RF model | Preserved (unchanged) | Preserved (unchanged) |
| Numerical results | Original estimand | Identical (same topology, same functions) |