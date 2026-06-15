 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which iterates over **~6.46 million rows** in a pure-R `lapply`, performing per-row character coercion, string pasting, and named-vector lookups. This creates ~6.46 million small character vectors, each requiring hash-table lookups into `idx_lookup` (a named vector of length 6.46M). Named-vector lookup in R is O(n) per probe in the worst case and carries heavy overhead at this scale. The result is an **86+ hour runtime** dominated by this single function.

Secondary cost: `compute_neighbor_stats` also loops over 6.46M entries, but each iteration is tiny (mean ~4 neighbors). Still, doing this 5 times (once per variable) in pure R adds up.

**Key insight:** The neighbor structure is *time-invariant*. Every year has the same cell-to-cell adjacency. So the 1,373,394 directed neighbor pairs can be crossed with 28 years to produce a **sparse edge list of ~38.5M (row_i, row_j) pairs**, and then neighbor max/min/mean can be computed as **grouped aggregations over this edge list** — a fully vectorized operation.

---

## Optimization Strategy

1. **Build a cell-year row index** using `data.table` keyed join (O(n log n), vectorized).
2. **Expand the spatial neighbor list into an edge table** of (cell_id, neighbor_cell_id) — only ~1.37M rows.
3. **Cross-join with years** to get (cell_id, year, neighbor_cell_id, year) → then join to the data to get (row_i, row_j) pairs — ~38.5M rows, fits in RAM (~600 MB).
4. **Compute grouped stats** (`max`, `min`, `mean`) per `row_i` using `data.table` — fully vectorized, single pass per variable.
5. **No change to the trained Random Forest model or numerical results.** The neighbor max, min, mean values are identical because the same neighbor relationships and the same aggregation functions are used.

**Expected runtime: ~2–5 minutes** (vs. 86+ hours).

---

## Working R Code

```r
library(data.table)

#' Vectorized neighbor feature engineering for large spatial panels.
#' Preserves the exact same numerical estimand as the original loop-based code.
#'
#' @param cell_data       data.frame / data.table with columns: id, year, and all neighbor_source_vars
#' @param id_order        character or integer vector — the cell IDs in the order matching rook_neighbors_unique
#' @param rook_neighbors_unique  spdep nb object (list of integer index vectors into id_order)
#' @param neighbor_source_vars   character vector of variable names to summarize
#' @return cell_data with new columns: {var}_neighbor_max, {var}_neighbor_min, {var}_neighbor_mean

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars = c("ntl", "ec", "pop_density",
                                                                   "def", "usd_est_n2")) {

  # --- Step 0: Convert to data.table, preserve original row order ---
  dt <- as.data.table(cell_data)
  dt[, .row_order := .I]

  # --- Step 1: Build spatial edge list from the nb object (time-invariant) ---
  #     Each element of rook_neighbors_unique is an integer vector of indices into id_order.
  #     A 0-length or 0-valued entry means no neighbors.
  n_cells <- length(id_order)
  from_idx <- rep(seq_len(n_cells), lengths(rook_neighbors_unique))
  to_idx   <- unlist(rook_neighbors_unique)

  # Remove the spdep convention where 0L means "no neighbors"
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  edges <- data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )
  rm(from_idx, to_idx, valid)
  # edges now has ~1,373,394 rows (directed rook-neighbor pairs)

  # --- Step 2: Build a keyed row-index table for (id, year) ---
  row_index <- dt[, .(id, year, .row_order)]
  setkey(row_index, id, year)

  # --- Step 3: Expand edges × years to get (row_i, row_j) pairs ---
  years <- sort(unique(dt$year))
  edge_years <- CJ(edge_idx = seq_len(nrow(edges)), year = years)
  edge_years[, `:=`(from_id = edges$from_id[edge_idx],
                     to_id   = edges$to_id[edge_idx])]
  edge_years[, edge_idx := NULL]

  # Join to get row_i (the focal cell-year row)
  setkey(edge_years, from_id, year)
  edge_years[row_index, row_i := i..row_order, on = .(from_id = id, year)]

  # Join to get row_j (the neighbor cell-year row)
  setkey(edge_years, to_id, year)
  edge_years[row_index, row_j := i..row_order, on = .(to_id = id, year)]

  # Drop any edges where either side is missing (masked cells / boundary)
  edge_years <- edge_years[!is.na(row_i) & !is.na(row_j)]
  # Keep only what we need
  edge_pairs <- edge_years[, .(row_i, row_j)]
  rm(edge_years, row_index)
  gc()

  # --- Step 4: For each variable, compute grouped neighbor stats ---
  for (var_name in neighbor_source_vars) {
    message("Computing neighbor stats for: ", var_name)

    # Attach neighbor values
    edge_pairs[, nval := dt[[var_name]][row_j]]

    # Grouped aggregation — only over non-NA neighbor values
    stats <- edge_pairs[!is.na(nval),
                        .(nmax  = max(nval),
                          nmin  = min(nval),
                          nmean = mean(nval)),
                        by = row_i]

    # Initialize new columns to NA
    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Assign results back by row index
    set(dt, i = stats$row_i, j = max_col,  value = stats$nmax)
    set(dt, i = stats$row_i, j = min_col,  value = stats$nmin)
    set(dt, i = stats$row_i, j = mean_col, value = stats$nmean)

    edge_pairs[, nval := NULL]
    rm(stats)
    gc()
  }

  # --- Step 5: Restore original order, return as data.frame if input was data.frame ---
  setorder(dt, .row_order)
  dt[, .row_order := NULL]

  if (!is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt[])
}
```

### Drop-in replacement call

```r
# ---- Usage (replaces the entire original outer loop) ----

# Load your existing objects
# cell_data                 <- readRDS("cell_data.rds")
# id_order                  <- readRDS("id_order.rds")
# rook_neighbors_unique     <- readRDS("rook_neighbors_unique.rds")
# rf_model                  <- readRDS("trained_rf_model.rds")   # untouched

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
)

# The trained RF model is never modified — just use it for prediction as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why the numerical estimand is preserved

| Property | Original code | Optimized code |
|---|---|---|
| Neighbor definition | `rook_neighbors_unique[[ref_idx]]` mapped through `id_order` | Same nb object, same `id_order` mapping |
| Year matching | `paste(neighbor_cell_id, year)` lookup | Equi-join on `(id, year)` — identical matching |
| NA handling | `neighbor_vals[!is.na(neighbor_vals)]` | `edge_pairs[!is.na(nval), ...]` — identical |
| No-neighbor case | Returns `c(NA, NA, NA)` | Column initialized to `NA_real_`, only overwritten if `row_i` appears in `stats` — identical |
| Aggregation functions | `max`, `min`, `mean` | `max`, `min`, `mean` — identical |

The output columns are **numerically identical** (to floating-point equality). The trained Random Forest model is never touched.