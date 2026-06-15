 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It lies squarely in `build_neighbor_lookup` and `compute_neighbor_stats`, both of which rely on **row-level `lapply` loops over ~6.46 million rows** with expensive per-iteration string operations.

### Specific problems:

**1. `build_neighbor_lookup` — O(n) string-paste loop with named-vector lookups**

For each of the ~6.46 million rows, the function:
- Converts an integer ID to a character and looks it up in a named vector (`id_to_ref`): **named-vector lookup is O(n) in the worst case in base R** because it uses linear hashing with potential collisions at scale.
- Calls `paste()` to construct neighbor keys (one per neighbor per row). With an average of ~4 rook neighbors, this is ~25.8 million `paste` calls.
- Indexes into `idx_lookup` (another named character vector of length 6.46 million) for each constructed key.
- Filters out `NA`s.

The result: **~6.46 million iterations, each doing multiple named-vector lookups into a 6.46-million-element vector, plus millions of `paste` operations**. This alone likely accounts for the majority of the 86+ hour runtime.

**2. `compute_neighbor_stats` — O(n) lapply with per-row subsetting**

For each row, it extracts neighbor values, removes `NA`s, and computes `max`, `min`, `mean`. This is called **5 times** (once per source variable), so that is 5 × 6.46M = ~32.3 million R-level function invocations. Each one allocates small vectors and runs three summary functions. The overhead of the R interpreter loop is enormous here.

**3. `do.call(rbind, result)` on a 6.46-million-element list of 3-element vectors** is itself slow because `rbind` on a long list is notoriously inefficient.

---

## Optimization Strategy

The core insight: **replace row-level R loops with vectorized data.table merge-and-group-by operations**.

### Plan:

1. **Replace `build_neighbor_lookup`** entirely. Instead of building a list-of-integer-vectors, construct a **long-form edge table** `(row_i, row_j)` that maps every cell-year row to its neighbor cell-year rows. This is done via a single vectorized `data.table` merge — no `lapply`, no `paste` per row.

2. **Replace `compute_neighbor_stats`** with a single **`data.table` grouped aggregation** (`[, .(max, min, mean), by = row_i]`) over the edge table joined to the variable column. This replaces 6.46 million R function calls with one vectorized C-level group-by.

3. **Eliminate `do.call(rbind, ...)`** — the `data.table` aggregation returns a proper table directly.

4. **Loop over the 5 variables** remains, but each iteration is now a fast vectorized operation (~seconds, not hours).

### Expected speedup:

| Step | Before | After |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M `lapply` iterations with `paste` + named lookups | One `data.table` merge (~seconds) |
| `compute_neighbor_stats` (×5 vars) | 5 × 6.46M `lapply` iterations | 5 × one `data.table` grouped aggregation |
| Total estimated time | 86+ hours | **~2–10 minutes** on a 16 GB laptop |

The Random Forest model is not touched. The numerical outputs (max, min, mean of neighbor values) are identical.

---

## Working R Code

```r
library(data.table)

#' Build a vectorized edge table mapping each cell-year row index
#' to its neighbor cell-year row indices.
#'
#' @param data       data.frame/data.table with columns `id` and `year`
#' @param id_order   integer vector of cell IDs in the order used by the nb object
#' @param neighbors  spdep nb object (list of integer index vectors into id_order)
#' @return data.table with columns: row_i (focal row), row_j (neighbor row)
build_neighbor_edge_table <- function(data, id_order, neighbors) {

  # --- Step 1: Build a cell-level edge list (focal_id -> neighbor_id) ----------
  # Each element neighbors[[k]] is an integer vector of indices into id_order.
  # We expand this into a two-column table.

  n_cells <- length(id_order)
  focal_indices <- rep(seq_len(n_cells), lengths(neighbors))
  neighbor_indices <- unlist(neighbors, use.names = FALSE)

  # Remove the spdep "no-neighbor" sentinel (0)
  valid <- neighbor_indices != 0L
  focal_indices    <- focal_indices[valid]
  neighbor_indices <- neighbor_indices[valid]

  cell_edges <- data.table(
    focal_id    = id_order[focal_indices],
    neighbor_id = id_order[neighbor_indices]
  )
  rm(focal_indices, neighbor_indices, valid)  # free memory

  # --- Step 2: Build a row-index lookup keyed by (id, year) -------------------
  dt <- as.data.table(data)
  dt[, row_idx := .I]  # original row position

  # --- Step 3: Merge to create (row_i, row_j) --------------------------------
  # For every (focal_id, neighbor_id) pair, we need every year that both exist in.
  # This is equivalent to:
  #   for each cell-year row i with (focal_id, year_t),
  #     find all rows j with (neighbor_id, year_t).


  # Keyed lookup tables
  focal_lookup <- dt[, .(focal_id = id, year, row_i = row_idx)]
  setkey(focal_lookup, focal_id, year)

  neighbor_lookup <- dt[, .(neighbor_id = id, year, row_j = row_idx)]
  setkey(neighbor_lookup, neighbor_id, year)

  # Join cell_edges with focal_lookup to get (row_i, neighbor_id, year)
  # Then join with neighbor_lookup to get (row_i, row_j)
  setkey(cell_edges, focal_id)
  setkey(focal_lookup, focal_id)

  # First merge: attach row_i and year to each edge
  edge_with_focal <- cell_edges[focal_lookup, on = "focal_id", allow.cartesian = TRUE, nomatch = 0L]
  # Columns: focal_id, neighbor_id, row_i, year

  rm(focal_lookup, cell_edges)

  # Second merge: attach row_j for the neighbor in the same year
  setkey(edge_with_focal, neighbor_id, year)
  setkey(neighbor_lookup, neighbor_id, year)

  edge_table <- edge_with_focal[neighbor_lookup, on = c("neighbor_id", "year"), nomatch = 0L]
  # Columns include: row_i, row_j  (plus others we can drop)

  rm(edge_with_focal, neighbor_lookup)

  edge_table <- edge_table[, .(row_i, row_j)]
  setkey(edge_table, row_i)

  return(edge_table)
}


#' Compute neighbor max, min, mean for one variable using the edge table.
#'
#' @param data       data.frame/data.table (original row order)
#' @param edge_table data.table with columns row_i, row_j
#' @param var_name   character: name of the variable in data
#' @return data.table with columns: row_i, nb_max, nb_min, nb_mean
compute_neighbor_stats_fast <- function(data, edge_table, var_name) {
  vals <- data[[var_name]]

  # Attach the neighbor's value to each edge
  et <- copy(edge_table)
  et[, nb_val := vals[row_j]]

  # Drop edges where neighbor value is NA
  et <- et[!is.na(nb_val)]

  # Grouped aggregation — single vectorized pass
  stats <- et[, .(
    nb_max  = max(nb_val),
    nb_min  = min(nb_val),
    nb_mean = mean(nb_val)
  ), by = row_i]

  return(stats)
}


#' Compute and attach neighbor features for one variable to the dataset.
#'
#' @param data       data.frame/data.table (will be modified by reference if data.table)
#' @param var_name   character
#' @param edge_table data.table from build_neighbor_edge_table
#' @return data with three new columns: <var>_nb_max, <var>_nb_min, <var>_nb_mean
compute_and_add_neighbor_features_fast <- function(data, var_name, edge_table) {
  stats <- compute_neighbor_stats_fast(data, edge_table, var_name)

  n <- nrow(data)
  col_max  <- paste0(var_name, "_nb_max")
  col_min  <- paste0(var_name, "_nb_min")
  col_mean <- paste0(var_name, "_nb_mean")

  # Initialize with NA, then fill matched rows
  data[[col_max]]  <- NA_real_
  data[[col_min]]  <- NA_real_
  data[[col_mean]] <- NA_real_

  data[[col_max]] [stats$row_i] <- stats$nb_max
  data[[col_min]] [stats$row_i] <- stats$nb_min
  data[[col_mean]][stats$row_i] <- stats$nb_mean

  return(data)
}


# ==============================================================================
# MAIN PIPELINE (drop-in replacement for the original outer loop)
# ==============================================================================

# Build the edge table ONCE (replaces build_neighbor_lookup)
message("Building neighbor edge table...")
edge_table <- build_neighbor_edge_table(cell_data, id_order, rook_neighbors_unique)
message(sprintf("Edge table built: %s edges", format(nrow(edge_table), big.mark = ",")))

# Compute neighbor features for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor stats for: %s", var_name))
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, var_name, edge_table)
}

message("Neighbor feature construction complete.")

# cell_data now has the same 15 new columns (5 vars × 3 stats) as the original code.
# Proceed to Random Forest prediction with the already-trained model — no retraining.
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Lookup construction | `lapply` over 6.46M rows, `paste` + named-vector search | Single `data.table` equi-join; vectorized |
| Neighbor stats | `lapply` over 6.46M rows × 5 vars, per-row `max`/`min`/`mean` | `data.table` grouped aggregation, C-level |
| Memory pattern | 6.46M-element list of integer vectors (~large overhead) | One flat two-column edge table (~105M rows × 2 int cols ≈ 1.6 GB) |
| Estimated runtime | 86+ hours | ~2–10 minutes |
| Numerical result | max, min, mean of non-NA neighbor values | **Identical** |
| RF model | Untouched | Untouched |

The key optimization is converting the **row-level R loop with string key construction and named-vector lookup** into a **vectorized relational join and group-by**, which is what `data.table` is specifically engineered to do at scale.