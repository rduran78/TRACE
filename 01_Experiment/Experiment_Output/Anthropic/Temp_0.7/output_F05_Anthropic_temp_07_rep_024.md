 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's why:

### The Core Problem: Per-Row String Operations at Scale

Inside `build_neighbor_lookup`, for each of the ~6.46 million rows, the code:

1. **Creates `idx_lookup`** — this line `setNames(seq_len(nrow(data)), paste(data$id, data$year, sep = "_"))` is defined *outside* the `lapply` but still constructs a named character vector of 6.46M string keys once. That part is O(N) and acceptable in isolation.

2. **Per-row `paste()` + named-vector lookup** — for each row `i`, it builds `neighbor_keys` via `paste()` and then does character-based lookup into `idx_lookup`. With ~4 neighbors per cell on average, this is ~6.46M × 4 = ~25.8M `paste()` calls and named-vector hash lookups. Named-vector lookup in R is hash-based but has significant per-call overhead due to string hashing and R's general interpreted overhead.

### The Broader Algorithmic Issue

The entire `build_neighbor_lookup` function is architecturally suboptimal because it solves a **join problem** with **row-by-row string manipulation**. The neighbor lookup is fundamentally: *"for each (cell, year), find the row indices of (neighbor_cell, same_year)"*. This is a **merge/join** operation that can be vectorized completely.

**Estimated cost of current approach:**
- `build_neighbor_lookup`: ~6.46M iterations of `lapply`, each doing `paste()` + character indexing → **hours**
- `compute_neighbor_stats`: runs 5 times over the lookup, each iterating 6.46M entries → significant but secondary
- Total: **86+ hours** as reported

### The Fix: Vectorized Join via `data.table`

We can replace the entire per-row string-key construction with:
1. An **integer-keyed join** — map `(id, year)` → row index using `data.table` keyed joins.
2. **Expand the neighbor list once** into a flat edge table `(source_row, neighbor_id)`, then join to get `(source_row, neighbor_row)`.
3. **Compute all neighbor stats vectorially** using `data.table` grouped aggregation.

This eliminates all per-row `paste()` calls and replaces the `lapply` with vectorized operations.

---

## Optimization Strategy

| Step | Current | Proposed |
|------|---------|----------|
| Map (id,year)→row | 6.46M-entry named char vector | `data.table` integer-keyed join |
| Expand neighbors per row | `lapply` over 6.46M rows, `paste` per neighbor | Vectorized expansion of `nb` object into flat `data.table` |
| Look up neighbor rows | Character hash per neighbor key | Keyed `data.table` equi-join on `(id, year)` integer columns |
| Compute stats | `lapply` over 6.46M rows per variable | `data.table` grouped `max/min/mean` per variable |

**Expected runtime:** Minutes instead of days. The join is O(E) where E ≈ total directed neighbor-year pairs (~1.37M neighbors × 28 years ≈ 38.5M edges), and grouped aggregation is similarly O(E).

---

## Working R Code

```r
library(data.table)

#' Build a flat edge table: for every row in cell_data, list the row indices
#' of its rook neighbors in the same year.
#'
#' @param cell_data       data.frame/data.table with columns `id` and `year`
#' @param id_order        integer vector of cell IDs in the order matching the nb object
#' @param rook_neighbors  spdep nb object (list of integer index vectors into id_order)
#' @return data.table with columns: src_row, nbr_row
build_neighbor_edge_table <- function(cell_data, id_order, rook_neighbors) {

  dt <- as.data.table(cell_data)
  dt[, src_row := .I]

  # --- Step 1: Expand the spatial nb object into a flat (src_id, nbr_id) table ---
  # Each element of rook_neighbors is an integer vector of indices into id_order.
  n_cells <- length(id_order)
  stopifnot(length(rook_neighbors) == n_cells)

  # Pre-compute lengths for allocation
  lens <- lengths(rook_neighbors)
  total_edges <- sum(lens)

  src_id_vec <- rep(id_order, times = lens)
  nbr_id_vec <- id_order[unlist(rook_neighbors, use.names = FALSE)]

  edges_spatial <- data.table(
    src_id = src_id_vec,
    nbr_id = nbr_id_vec
  )

  # --- Step 2: Join to get (src_row, nbr_row) for every (cell, year) ---
  # Build a lookup from (id, year) -> row index
  row_lookup <- dt[, .(id, year, src_row)]
  setkey(row_lookup, id, year)

  # For every row in dt, get its spatial neighbors via join on src_id
  # First, create (src_row, src_id, year, nbr_id) by joining dt rows to spatial edges
  src_info <- dt[, .(src_row, src_id = id, year)]
  setkey(edges_spatial, src_id)
  setkey(src_info, src_id)

  # This is a many-to-many join: each row has multiple neighbors,
  # and each src_id appears in multiple years.
  # Use edges_spatial as the left table keyed on src_id, join with src_info.
  # More efficient: join src_info to edges_spatial on src_id
  expanded <- edges_spatial[src_info,
    .(src_row, nbr_id, year),
    on = .(src_id),
    allow.cartesian = TRUE,
    nomatch = NULL
  ]

  # Now look up the row index of each (nbr_id, year)
  # Create a keyed lookup: (id, year) -> row index
  nbr_lookup <- dt[, .(nbr_row = src_row, id, year)]
  setkey(nbr_lookup, id, year)

  # Join to resolve nbr_id + year -> nbr_row
  expanded[, id := nbr_id]
  result <- nbr_lookup[expanded, .(src_row, nbr_row), on = .(id, year), nomatch = NA]

  # Drop edges where the neighbor cell-year doesn't exist in the data

  result <- result[!is.na(nbr_row)]

  return(result)
}


#' Compute neighbor max, min, mean for a variable using the edge table.
#'
#' @param cell_data  data.frame/data.table with the source variable
#' @param var_name   character: column name to aggregate
#' @param edge_dt    data.table with columns src_row, nbr_row
#' @return data.table with nrow(cell_data) rows and columns:
#'         nb_max_{var}, nb_min_{var}, nb_mean_{var}
compute_neighbor_stats_fast <- function(cell_data, var_name, edge_dt) {

  dt <- as.data.table(cell_data)
  n <- nrow(dt)

  # Attach the neighbor's value to each edge
  vals <- dt[[var_name]]
  edge_work <- edge_dt[, .(src_row, nbr_val = vals[nbr_row])]

  # Drop NAs in the variable
  edge_work <- edge_work[!is.na(nbr_val)]

  # Grouped aggregation
  agg <- edge_work[, .(
    nb_max  = max(nbr_val),
    nb_min  = min(nbr_val),
    nb_mean = mean(nbr_val)
  ), keyby = src_row]

  # Allocate full-length result with NAs for rows with no valid neighbors
  out <- data.table(
    src_row = seq_len(n),
    nb_max  = NA_real_,
    nb_min  = NA_real_,
    nb_mean = NA_real_
  )
  out[agg, on = .(src_row), `:=`(
    nb_max  = i.nb_max,
    nb_min  = i.nb_min,
    nb_mean = i.nb_mean
  )]

  # Rename columns to match original naming convention
  max_col  <- paste0("nb_max_", var_name)
  min_col  <- paste0("nb_min_", var_name)
  mean_col <- paste0("nb_mean_", var_name)
  setnames(out, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))

  out[, src_row := NULL]
  return(out)
}


#' Main driver: replaces the entire outer loop.
#'
#' @param cell_data              data.frame with columns id, year, and all source vars
#' @param id_order               integer vector of cell IDs matching the nb object order
#' @param rook_neighbors_unique  spdep nb object
#' @param neighbor_source_vars   character vector of variable names
#' @return cell_data with neighbor feature columns appended
add_all_neighbor_features <- function(cell_data,
                                      id_order,
                                      rook_neighbors_unique,
                                      neighbor_source_vars = c("ntl", "ec", "pop_density",
                                                                "def", "usd_est_n2")) {

  message("Building neighbor edge table...")
  t0 <- proc.time()
  edge_dt <- build_neighbor_edge_table(cell_data, id_order, rook_neighbors_unique)
  message(sprintf("  Edge table: %s edges built in %.1f seconds.",
                  format(nrow(edge_dt), big.mark = ","),
                  (proc.time() - t0)[3]))

  cell_data <- as.data.table(cell_data)

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for '%s'...", var_name))
    t1 <- proc.time()
    stats_dt <- compute_neighbor_stats_fast(cell_data, var_name, edge_dt)
    cell_data <- cbind(cell_data, stats_dt)
    message(sprintf("  Done in %.1f seconds.", (proc.time() - t1)[3]))
  }

  return(cell_data)
}


# ============================================================
# USAGE — drop-in replacement for the original outer loop
# ============================================================
# cell_data <- add_all_neighbor_features(
#   cell_data,
#   id_order,
#   rook_neighbors_unique,
#   neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# )
#
# # The trained Random Forest model is untouched — just use predict() as before:
# # preds <- predict(rf_model, newdata = cell_data)
```

---

## Complexity and Memory Analysis

| Metric | Original | Optimized |
|--------|----------|-----------|
| **Neighbor expansion** | O(N) `lapply`, each with `paste` + char hash | O(E) vectorized `data.table` join |
| **Stats computation** | O(N) `lapply` × 5 vars | O(E) grouped aggregation × 5 vars |
| **Total string ops** | ~25.8M `paste()` + 25.8M char lookups | **Zero** |
| **Edge table memory** | N/A (implicit) | ~38.5M rows × 2 int cols ≈ **0.6 GB** |
| **Expected runtime** | 86+ hours | **~5–15 minutes** |

### Why This Preserves Correctness

- **Same estimand**: For each `(cell, year)` row, we compute `max`, `min`, `mean` over the exact same set of neighbor values — those neighbors identified by `rook_neighbors_unique` that exist in the same year. The `nomatch = NA` + `!is.na(nbr_row)` filtering exactly replicates the original `result[!is.na(result)]` logic.
- **Same column names**: Output columns follow the same `nb_max_{var}`, `nb_min_{var}`, `nb_mean_{var}` naming pattern (adjust the prefix if your `compute_and_add_neighbor_features` used a different convention).
- **Trained RF model untouched**: No model retraining is needed — the feature columns are numerically identical, so `predict(rf_model, newdata = cell_data)` produces the same predictions.

### Key Insight

The fundamental algorithmic shift is from **row-iteration with string-key lookups** (O(N) interpreted R loop with per-element string overhead) to a **single vectorized relational join** (O(E) in compiled C code inside `data.table`). The string-key pattern was indeed a symptom of solving a join problem imperatively rather than declaratively.