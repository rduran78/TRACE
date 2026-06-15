 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It is the two spatial neighbor feature construction functions, and the root causes are clear:

### 1. `build_neighbor_lookup` — O(n) `lapply` over 6.46 million rows with per-row string operations

For every one of the ~6.46 million rows, the function:
- Converts an integer ID to character and looks it up in a named vector (`id_to_ref`).
- Retrieves neighbor cell IDs from the `nb` object.
- Calls `paste()` to construct `"id_year"` key strings for every neighbor of that row.
- Performs named-vector lookup (`idx_lookup[neighbor_keys]`) — which is an **O(k)** hash lookup per neighbor key, but the string construction and allocation for ~6.46M × ~4 neighbors (average rook degree ≈ 4) means roughly **25+ million small string allocations and hash probes**.
- Wraps everything in an R-level `lapply`, so there is no vectorization; each iteration pays full R interpreter overhead.

**Estimated cost:** With ~6.46M iterations, each doing string paste + hash lookup for ~4 neighbors, this alone can take many hours in pure R.

### 2. `compute_neighbor_stats` — Another O(n) `lapply` with per-row subsetting

For each of the 6.46M rows, it:
- Subsets a numeric vector by index (`vals[idx]`).
- Removes NAs.
- Computes `max`, `min`, `mean`.

This is called **5 times** (once per source variable), so it executes ~32.3 million R-level loop iterations total. The per-row function-call overhead in R's `lapply` is the killer here.

### 3. `do.call(rbind, result)` on a 6.46M-element list of 3-vectors

This is a well-known R anti-pattern. Binding millions of small vectors into a matrix via `do.call(rbind, ...)` is extremely slow due to repeated memory allocation.

### Summary of bottleneck contributions (estimated):

| Component | Estimated share |
|---|---|
| `build_neighbor_lookup` (string ops + hash) | ~40% |
| `compute_neighbor_stats` (R-level loop × 5 vars) | ~50% |
| `do.call(rbind, ...)` × 5 | ~10% |

---

## Optimization Strategy

The core idea: **replace all row-level R loops and string operations with vectorized operations on integer indices using `data.table`.**

Specifically:

1. **Replace `build_neighbor_lookup`** with a single `data.table` join. Instead of building a list-of-integer-vectors (one per row), build a **flat edge table** `(row_i, neighbor_row_j)` using vectorized integer operations — no `paste`, no named-vector hash lookups.

2. **Replace `compute_neighbor_stats`** with a grouped `data.table` aggregation on the flat edge table: group by `row_i`, compute `max`, `min`, `mean` of the neighbor values in one vectorized pass. This replaces 6.46M R-level `lapply` iterations with a single C-level `data.table` grouped operation.

3. **Eliminate `do.call(rbind, ...)`** entirely — `data.table` returns the result as a table directly.

4. **Process all 5 variables in one pass** over the edge table if desired, or in 5 fast grouped aggregations (each taking seconds, not hours).

**Expected speedup:** From ~86+ hours to **~2–10 minutes** on the same laptop.

**Numerical equivalence:** The operations are identical — `max`, `min`, `mean` of the same neighbor values. The trained Random Forest model is untouched.

---

## Working R Code

```r
library(data.table)

#' Build a flat edge table mapping each cell-year row to its neighbor cell-year rows.
#' Replaces build_neighbor_lookup entirely — no lapply, no paste, no string hashing.
#'
#' @param cell_dt     data.table with columns: id, year (and all predictor columns).
#'                    Must have a column .row_idx = seq_len(nrow(.)).
#' @param id_order    integer vector of cell IDs in the order matching the nb object.
#' @param neighbors   spdep nb object (list of integer index vectors into id_order).
#' @return data.table with columns: row_i (focal row), row_j (neighbor row).
build_neighbor_edge_table <- function(cell_dt, id_order, neighbors) {

  n_cells <- length(id_order)

  # --- Step 1: Build flat cell-level edge list (vectorized) ---
  # neighbors[[k]] gives integer indices into id_order for cell id_order[k].
  # We need: from_cell_id -> to_cell_id

  n_neighbors <- vapply(neighbors, length, integer(1))
  from_cell_idx <- rep(seq_len(n_cells), times = n_neighbors)
  to_cell_idx   <- unlist(neighbors, use.names = FALSE)

  # Convert positional indices to actual cell IDs
  edge_cells <- data.table(
    from_id = id_order[from_cell_idx],
    to_id   = id_order[to_cell_idx]
  )
  rm(from_cell_idx, to_cell_idx, n_neighbors)

  # --- Step 2: Get the unique years present in the data ---
  years <- sort(unique(cell_dt$year))

  # --- Step 3: Cross-join edges × years, then join to row indices ---
  # Expand edges to all years (each spatial edge exists in every year)
  edge_years <- edge_cells[, .(year = years), by = .(from_id, to_id)]
  rm(edge_cells)

  # Build a lookup: (id, year) -> row index in cell_dt
  # cell_dt must already have .row_idx column
  id_year_lookup <- cell_dt[, .(id, year, .row_idx)]
  setkey(id_year_lookup, id, year)

  # Map from_id,year -> row_i
  setnames(id_year_lookup, ".row_idx", "row_i")
  setkey(edge_years, from_id, year)
  edge_years <- id_year_lookup[edge_years, on = .(id = from_id, year), nomatch = 0L]
  # Now edge_years has columns: id, year, row_i, to_id

  # Map to_id,year -> row_j
  setnames(id_year_lookup, "row_i", "row_j")
  edge_years <- id_year_lookup[edge_years, on = .(id = to_id, year), nomatch = 0L]
  # Now edge_years has columns: id, year, row_j, i.id, i.year, row_i, to_id

  # Keep only what we need
  result <- edge_years[, .(row_i, row_j)]
  setkey(result, row_i)

  rm(id_year_lookup, edge_years)
  gc()

  return(result)
}


#' Compute neighbor max, min, mean for a variable using the flat edge table.
#' Replaces compute_neighbor_stats — one vectorized data.table grouped aggregation.
#'
#' @param cell_dt    data.table with .row_idx and the variable column.
#' @param edge_dt    data.table with columns row_i, row_j (from build_neighbor_edge_table).
#' @param var_name   character: name of the variable to aggregate.
#' @return data.table with columns: .row_idx, <var>_max, <var>_min, <var>_mean
compute_neighbor_stats_fast <- function(cell_dt, edge_dt, var_name) {

  # Attach neighbor values to edge table
  vals <- cell_dt[[var_name]]
  agg <- edge_dt[, .(neighbor_val = vals[row_j]), by = row_i]

  # Remove NAs before aggregation
  agg <- agg[!is.na(neighbor_val)]

  # Grouped aggregation — single pass in C
  stats <- agg[, .(
    nb_max  = max(neighbor_val),
    nb_min  = min(neighbor_val),
    nb_mean = mean(neighbor_val)
  ), by = row_i]

  # Rename columns to match original naming convention
  setnames(stats, c("nb_max", "nb_min", "nb_mean"),
           paste0(var_name, c("_max", "_min", "_mean")))

  return(stats)
}


#' Main driver: build features for all neighbor source variables.
#' Drop-in replacement for the original outer loop.
#'
#' @param cell_data           data.frame or data.table — the full panel dataset.
#' @param id_order            integer vector of cell IDs matching the nb object.
#' @param rook_neighbors_unique  spdep nb object.
#' @param neighbor_source_vars   character vector of variable names.
#' @return data.table with all original columns plus neighbor feature columns.
add_all_neighbor_features <- function(cell_data, id_order,
                                      rook_neighbors_unique,
                                      neighbor_source_vars) {

  cell_dt <- as.data.table(cell_data)
  cell_dt[, .row_idx := .I]

  message("Building neighbor edge table (vectorized)...")
  t0 <- proc.time()
  edge_dt <- build_neighbor_edge_table(cell_dt, id_order, rook_neighbors_unique)
  message(sprintf("  Edge table: %d edges, built in %.1f seconds.",
                  nrow(edge_dt), (proc.time() - t0)[3]))

  # Compute and join neighbor stats for each variable
  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for '%s'...", var_name))
    t1 <- proc.time()

    stats <- compute_neighbor_stats_fast(cell_dt, edge_dt, var_name)

    # Left-join back to cell_dt on .row_idx = row_i
    # Rows with no valid neighbors will get NA (correct behavior, matches original)
    feat_cols <- paste0(var_name, c("_max", "_min", "_mean"))
    cell_dt <- stats[cell_dt, on = .(row_i = .row_idx)]
    setnames(cell_dt, "row_i", ".row_idx")

    elapsed <- (proc.time() - t1)[3]
    message(sprintf("  Done in %.1f seconds.", elapsed))
  }

  # Clean up helper column
  cell_dt[, .row_idx := NULL]

  rm(edge_dt)
  gc()

  return(cell_dt)
}


# ============================================================
# USAGE — drop-in replacement for the original outer loop
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is unchanged.
# Proceed directly to prediction:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Works and What It Preserves

| Concern | Status |
|---|---|
| **Numerical equivalence** | Identical — same `max`, `min`, `mean` over the same neighbor sets. NAs handled identically (removed before aggregation; rows with zero valid neighbors get `NA`). |
| **Trained RF model** | Untouched — only the feature construction pipeline is changed. Column names and semantics are preserved. |
| **Memory on 16 GB laptop** | The flat edge table has ~1.37M spatial edges × 28 years ≈ 38.5M rows × 2 integer columns ≈ **~0.6 GB**. The aggregation temporary is similar. Total peak overhead ≈ ~2–3 GB, well within 16 GB. |
| **Expected runtime** | Edge table construction: ~30–60 seconds. Each variable's grouped aggregation: ~10–30 seconds. **Total: ~2–5 minutes** vs. 86+ hours. |

### Key optimizations summarized:

1. **Eliminated all `paste()`/string hashing** — replaced with integer key joins in `data.table`.
2. **Eliminated 6.46M × 5 = 32.3M R-level `lapply` iterations** — replaced with 5 `data.table` grouped aggregations (executed in C).
3. **Eliminated `do.call(rbind, ...)`** — `data.table` returns results directly as a table.
4. **Edge table is built once and reused** for all 5 variables.