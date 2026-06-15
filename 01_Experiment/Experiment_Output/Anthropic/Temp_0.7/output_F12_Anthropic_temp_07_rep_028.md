 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The colleague is **partially right but misses the deeper bottleneck**. Let me audit each claim:

**Claim 1: `do.call(rbind, result)` is a bottleneck.**
This is a minor concern. `do.call(rbind, ...)` on a list of ~6.46 million 3-element numeric vectors is not free, but it's a single matrix assembly call — not the dominant cost. This can be improved but is not the main problem.

**Claim 2: "Repeated list binding inside `compute_neighbor_stats()`".**
There is **no repeated list binding** inside `compute_neighbor_stats()`. It uses `lapply` to build the list in one pass and then `rbind`s once. This claim is factually incorrect against the code.

### The True Bottleneck: `build_neighbor_lookup()`

The real bottleneck is **`build_neighbor_lookup()`**, which runs a `lapply` over **~6.46 million rows**, and for each row:

1. Calls `as.character()` on a scalar to look up `id_to_ref`.
2. Indexes into the `neighbors` list to get neighbor cell IDs.
3. Calls `paste()` to construct character keys for every neighbor of every row.
4. Performs **named character vector lookups** (`idx_lookup[neighbor_keys]`) — this is a **hash lookup on ~6.46 million keys repeated for every row's neighbors**.

With ~1,373,394 directed neighbor relationships spread across 344,208 cells and 28 years, each cell has ~4 neighbors on average (rook contiguity). That means for each of the 6.46M rows, we `paste` ~4 keys and do ~4 named vector lookups. That's **~25.8 million `paste` + hash-lookup operations**, all inside an R-level loop with per-element overhead. The `paste()` calls alone generate enormous garbage-collection pressure.

Furthermore, `build_neighbor_lookup()` produces a **list of 6.46 million integer vectors** — a huge memory structure that must then be traversed again 5 times (once per variable) by `compute_neighbor_stats()`.

**Summary:** The deep bottleneck is `build_neighbor_lookup()` with its per-row string construction and named-vector lookups across 6.46M rows. The secondary cost is iterating that 6.46M-element lookup list 5 times in `compute_neighbor_stats()`. The `do.call(rbind, ...)` is a distant third.

---

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup()` entirely** — eliminate the per-row `lapply`. Instead, build a flat edge list (a two-column matrix: `from_row → to_row`) using fully vectorized operations. This replaces 6.46M R-level iterations with a handful of vectorized calls.

2. **Vectorize `compute_neighbor_stats()` using the edge list** — use `data.table` grouped aggregation on the edge list to compute max/min/mean of neighbor values in one vectorized pass per variable. This eliminates 6.46M R-level function calls per variable.

3. **Preserve the trained Random Forest model** — we only change feature engineering; the model object and all downstream predictions are untouched.

4. **Preserve the original numerical estimand** — the same max, min, mean statistics over the same rook neighbors are computed; only the computational method changes.

---

## Working R Code

```r
library(data.table)

# =============================================================================
# OPTIMIZED: build_neighbor_edge_list()
# Replaces build_neighbor_lookup().
# Returns a two-column integer matrix: (from_row, to_row)
# Fully vectorized — no per-row lapply.
# =============================================================================
build_neighbor_edge_list <- function(data, id_order, neighbors) {
  # data must have columns: id, year
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer vectors of neighbor indices)

  n_cells <- length(id_order)
  stopifnot(n_cells == length(neighbors))

  # --- Step 1: Build flat cell-level edge list (cell_from_idx -> cell_to_idx)
  #     where indices refer to positions in id_order.
  from_cell <- rep(seq_len(n_cells), lengths(neighbors))
  to_cell   <- unlist(neighbors, use.names = FALSE)
  # Now from_cell[k] is a neighbor of to_cell[k] in id_order-space

  # --- Step 2: Map cell indices to cell IDs
  from_cell_id <- id_order[from_cell]
  to_cell_id   <- id_order[to_cell]

  # --- Step 3: Build a data.table of (id, year) with row indices
  dt <- data.table(
    id       = data$id,
    year     = data$year,
    row_idx  = seq_len(nrow(data))
  )
  setkey(dt, id, year)

  # --- Step 4: Get unique years
  years <- sort(unique(dt$year))

  # --- Step 5: For each year, cross the cell-level edges with that year
  #     to get row-level edges. Vectorized via data.table joins.
  #     We build a cell-edge data.table once, then join per year.
  cell_edges <- data.table(from_id = from_cell_id, to_id = to_cell_id)

  edge_list <- rbindlist(lapply(years, function(yr) {
    # Rows in this year, keyed by id
    yr_rows <- dt[year == yr, .(id, row_idx)]
    setkey(yr_rows, id)

    # Join from_id -> from_row
    merged <- cell_edges[yr_rows, on = .(from_id = id), nomatch = 0L,
                         .(from_row = i.row_idx, to_id)]

    # Join to_id -> to_row
    setkey(yr_rows, id)
    merged <- merged[yr_rows, on = .(to_id = id), nomatch = 0L,
                     .(from_row, to_row = i.row_idx)]

    merged
  }), use.names = TRUE)

  # Return as a two-column integer matrix for maximum downstream speed
  as.matrix(edge_list[, .(from_row, to_row)])
}


# =============================================================================
# OPTIMIZED: compute_neighbor_stats_vectorized()
# Replaces compute_neighbor_stats().
# Uses data.table grouped aggregation — no per-row lapply.
# =============================================================================
compute_neighbor_stats_vectorized <- function(data, edge_matrix, var_name) {
  n <- nrow(data)
  vals <- data[[var_name]]

  # Build data.table of edges with neighbor values
  edge_dt <- data.table(
    from_row = edge_matrix[, 1],
    val      = vals[edge_matrix[, 2]]  # value at the neighbor row
  )

  # Remove edges where the neighbor value is NA
  edge_dt <- edge_dt[!is.na(val)]

  # Grouped aggregation: max, min, mean per from_row
  stats <- edge_dt[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = from_row]

  # Initialize output columns with NA
  out_max  <- rep(NA_real_, n)
  out_min  <- rep(NA_real_, n)
  out_mean <- rep(NA_real_, n)

  # Fill in computed values
  out_max[stats$from_row]  <- stats$nb_max
  out_min[stats$from_row]  <- stats$nb_min
  out_mean[stats$from_row] <- stats$nb_mean

  # Return as a 3-column matrix (same shape as original)
  cbind(out_max, out_min, out_mean)
}


# =============================================================================
# OPTIMIZED: compute_and_add_neighbor_features_vectorized()
# Drop-in replacement for compute_and_add_neighbor_features().
# Preserves original column naming convention.
# =============================================================================
compute_and_add_neighbor_features_vectorized <- function(data, var_name,
                                                          edge_matrix) {
  stats_mat <- compute_neighbor_stats_vectorized(data, edge_matrix, var_name)

  data[[paste0(var_name, "_neighbor_max")]]  <- stats_mat[, 1]
  data[[paste0(var_name, "_neighbor_min")]]  <- stats_mat[, 2]
  data[[paste0(var_name, "_neighbor_mean")]] <- stats_mat[, 3]

  data
}


# =============================================================================
# OPTIMIZED OUTER PIPELINE
# Replaces the original outer loop.
# The trained Random Forest model is NOT touched.
# =============================================================================

# --- Build the edge list ONCE (replaces build_neighbor_lookup) ---
message("Building vectorized neighbor edge list...")
edge_matrix <- build_neighbor_edge_list(cell_data, id_order, rook_neighbors_unique)
message(sprintf("Edge list built: %d row-level edges.", nrow(edge_matrix)))

# --- Compute neighbor features for all 5 variables ---
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor stats for: %s", var_name))
  cell_data <- compute_and_add_neighbor_features_vectorized(
    cell_data, var_name, edge_matrix
  )
}

message("All neighbor features computed. Ready for prediction with existing RF model.")
```

---

## Expected Speedup

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M R-level iterations with `paste` + hash lookup | Vectorized `data.table` joins; ~28 iterations (one per year) |
| `compute_neighbor_stats` (×5 vars) | ~6.46M R-level iterations per variable | Single `data.table` grouped aggregation per variable |
| `do.call(rbind, ...)` | Binding 6.46M list elements | Eliminated entirely (pre-allocated vectors) |
| **Estimated total time** | **86+ hours** | **Minutes** (typically 5–15 min on 16 GB laptop) |

The key insight: the original code performed **~38.8 million R-interpreter-level function invocations** (6.46M × 6 calls: lookup + 5 stats), each with string construction and hash lookups. The optimized version reduces this to **~33 vectorized operations** (28 year-joins + 5 grouped aggregations), letting R's internal C routines and `data.table`'s optimized grouping handle the heavy lifting.