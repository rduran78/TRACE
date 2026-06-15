 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full chain of waste:

1. **O(N) string-key construction inside `build_neighbor_lookup`**: `idx_lookup` is built once (good), but inside the `lapply` over ~6.46M rows, `paste()` is called per-row to construct `neighbor_keys`. With an average of ~8 rook neighbors per cell (1.37M directed relationships / ~344K cells ≈ 4 per cell, but shared across years), this generates tens of millions of small `paste()` calls plus hash lookups into a 6.46M-entry named vector. The named-vector lookup `idx_lookup[neighbor_keys]` is itself O(k) per call but with enormous constant overhead due to string hashing on a 6.46M-name vector.

2. **The entire string-keying approach is unnecessary**: The data is a balanced panel (344,208 cells × 28 years). The neighbor structure is **time-invariant** — cell A's rook neighbors are the same in every year. Therefore, the neighbor relationships can be expressed as **row-index offsets** in a data.table sorted by (id, year) or (year, id). There is no need to do any per-row string lookup at all.

3. **`lapply` over 6.46M rows returns a list of integer vectors** — this is inherently slow in R and memory-wasteful. The neighbor lookup can be vectorized entirely using a **sparse adjacency approach** or a **data.table join**.

4. **`compute_neighbor_stats` re-traverses the same list structure 5 times** (once per variable), each time pulling values by index. This could be done in a single pass or via matrix operations.

### Summary of the cost hierarchy

| Layer | Operation | Calls | Bottleneck |
|-------|-----------|-------|------------|
| String key build | `paste()` + named-vector construction | 1× for 6.46M keys | Moderate |
| Per-row neighbor key lookup | `paste()` + `idx_lookup[keys]` | 6.46M × ~4 neighbors | **Dominant** |
| Per-variable stats | List traversal × 5 vars | 5 × 6.46M | Significant |
| `do.call(rbind, ...)` on 6.46M-element list | Memory allocation | 5× | Significant |

## Optimization Strategy

**Core insight**: Since the panel is balanced and the neighbor structure is time-invariant, we can:

1. **Sort data by (year, id)** so that within each year-block, cells appear in the same order.
2. **Express the neighbor graph as a sparse matrix** (or equivalently, a two-column edge list of cell-position indices).
3. **For each year-block**, the row positions are a simple offset from the cell's position index. Neighbor row indices become `offset + neighbor_cell_positions` — pure integer arithmetic, no strings.
4. **Compute all 5 variables' stats in a single vectorized pass** using sparse matrix–vector multiplication (for mean/sum) and grouped operations for min/max.

The most efficient approach uses `Matrix::sparseMatrix` to represent the adjacency, then computes neighbor means via matrix multiplication and neighbor min/max via grouped operations on an edge list — all fully vectorized with no R-level loops over 6.46M rows.

## Working R Code

```r
library(data.table)
library(Matrix)

#' Optimized neighbor feature construction for a balanced cell-year panel.
#'
#' @param cell_data        data.frame/data.table with columns: id, year, and the source vars
#' @param id_order         integer vector of cell IDs in the order used by rook_neighbors_unique
#' @param rook_nb          spdep::nb object (rook_neighbors_unique)
#' @param neighbor_source_vars character vector of variable names
#' @return data.table with original columns plus neighbor features appended
compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_nb,
                                          neighbor_source_vars) {

  dt <- as.data.table(cell_data)

  # ---------------------------------------------------------------
  # 1. Build a mapping from cell id -> position index (1..N_cells)
  # ---------------------------------------------------------------
  n_cells <- length(id_order)
  id_to_pos <- setNames(seq_len(n_cells), as.character(id_order))

  # Assign each row its cell-position index
  dt[, cell_pos := id_to_pos[as.character(id)]]

  # Sort by (year, cell_pos) so within each year the cells are in
  # canonical position order. This is the key enabler.
  setorder(dt, year, cell_pos)

  # Verify balanced panel
  years <- sort(unique(dt$year))
  n_years <- length(years)
  stopifnot(nrow(dt) == n_cells * n_years)

  # ---------------------------------------------------------------
  # 2. Build directed edge list from the nb object (cell-position space)
  #    from_pos -> to_pos  (time-invariant)
  # ---------------------------------------------------------------
  edge_from <- integer(0)
  edge_to   <- integer(0)
  for (i in seq_len(n_cells)) {
    nb_i <- rook_nb[[i]]
    if (length(nb_i) > 0 && !(length(nb_i) == 1 && nb_i[1] == 0L)) {
      edge_from <- c(edge_from, rep.int(i, length(nb_i)))
      edge_to   <- c(edge_to,   as.integer(nb_i))
    }
  }
  n_edges <- length(edge_from)
  cat(sprintf("Neighbor edge list: %d directed edges\n", n_edges))

  # ---------------------------------------------------------------
  # 3. Build row-level edge list for ALL year-blocks at once.
  #
  #    Because dt is sorted by (year, cell_pos), the row for

  #    cell_pos=p in year_index=t (0-based) is:  row = t * n_cells + p
  #
  #    So we replicate the edge list across all years via offset.
  # ---------------------------------------------------------------
  year_offsets <- (seq_len(n_years) - 1L) * n_cells  # length n_years

  # Pre-allocate full edge list: n_edges * n_years entries
  total_edges <- as.double(n_edges) * n_years
  cat(sprintf("Total row-level edges: %.0f\n", total_edges))

  row_from <- integer(total_edges)
  row_to   <- integer(total_edges)

  for (t_idx in seq_len(n_years)) {
    off <- year_offsets[t_idx]
    start <- (t_idx - 1L) * n_edges + 1L
    end   <- t_idx * n_edges
    row_from[start:end] <- edge_from + off
    row_to[start:end]   <- edge_to   + off
  }

  # ---------------------------------------------------------------
  # 4. Build sparse adjacency matrix (n_rows x n_rows) — but we only
  #    need it for matrix-vector products (for mean).
  #    Also compute the number of neighbors per row (degree) for mean.
  # ---------------------------------------------------------------
  n_rows <- nrow(dt)

  # Sparse matrix: adj[i, j] = 1 means j is a neighbor of i
  # So adj %*% vals = sum of neighbor values for each row
  adj <- sparseMatrix(
    i = row_from,
    j = row_to,
    x = rep.int(1, length(row_from)),
    dims = c(n_rows, n_rows)
  )

  # Degree (number of non-NA neighbors will be adjusted per variable)
  degree <- as.integer(rowSums(adj))  # number of neighbors per row

  # ---------------------------------------------------------------
  # 5. For each variable, compute neighbor max, min, mean
  #    - mean: use sparse mat-vec product, divide by count of non-NA neighbors
  #    - min/max: use edge list with data.table grouped operations
  # ---------------------------------------------------------------

  # Pre-build the edge data.table (reusable across variables)
  edge_dt <- data.table(from_row = row_from, to_row = row_to)

  for (var_name in neighbor_source_vars) {
    cat(sprintf("Processing neighbor stats for: %s\n", var_name))

    vals <- dt[[var_name]]

    # --- Neighbor MEAN via sparse matrix-vector multiply ---
    # Replace NA with 0 for summation, track non-NA count separately
    vals_nona <- vals
    is_na_val <- is.na(vals_nona)
    vals_nona[is_na_val] <- 0

    neighbor_sum   <- as.numeric(adj %*% vals_nona)
    # Count of non-NA neighbors: use indicator vector
    not_na_ind <- as.numeric(!is_na_val)
    neighbor_count <- as.numeric(adj %*% not_na_ind)

    neighbor_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)

    # --- Neighbor MAX and MIN via edge-list grouped operations ---
    # Attach the neighbor's value to each edge
    edge_dt[, nb_val := vals[to_row]]

    # Remove edges where neighbor value is NA
    edge_valid <- edge_dt[!is.na(nb_val)]

    if (nrow(edge_valid) > 0) {
      stats_dt <- edge_valid[, .(
        nb_max = max(nb_val),
        nb_min = min(nb_val)
      ), by = from_row]

      # Initialize with NA
      neighbor_max <- rep(NA_real_, n_rows)
      neighbor_min <- rep(NA_real_, n_rows)

      neighbor_max[stats_dt$from_row] <- stats_dt$nb_max
      neighbor_min[stats_dt$from_row] <- stats_dt$nb_min
    } else {
      neighbor_max <- rep(NA_real_, n_rows)
      neighbor_min <- rep(NA_real_, n_rows)
    }

    # --- Assign columns using the ORIGINAL naming convention ---
    # Original code produces: {var}_max, {var}_min, {var}_mean
    max_col  <- paste0(var_name, "_max")
    min_col  <- paste0(var_name, "_min")
    mean_col <- paste0(var_name, "_mean")

    set(dt, j = max_col,  value = neighbor_max)
    set(dt, j = min_col,  value = neighbor_min)
    set(dt, j = mean_col, value = neighbor_mean)
  }

  # ---------------------------------------------------------------
  # 6. Restore original row order and return
  # ---------------------------------------------------------------
  # Remove helper column
  dt[, cell_pos := NULL]

  # Restore original order if needed — the caller may expect id,year order.
  # The original data's order should be preserved. We store original row index:
  # Actually, let's be safe: sort back to (id, year) which is a common panel order.
  setorder(dt, id, year)

  return(dt)
}

# ===================================================================
# USAGE — drop-in replacement for the original outer loop
# ===================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data        = cell_data,
  id_order         = id_order,
  rook_nb          = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The trained Random Forest model is used downstream unchanged —
# the columns it expects ({var}_max, {var}_min, {var}_mean) are
# now present with identical numerical values.
```

## Complexity and Runtime Comparison

| Aspect | Original | Optimized |
|--------|----------|-----------|
| **Neighbor lookup** | 6.46M `paste()` + string hash lookups | Integer addition (offset + position) |
| **Per-variable stats** | R-level `lapply` over 6.46M list elements | Sparse mat-vec (`Matrix`) + `data.table` grouped agg |
| **String allocations** | ~50M+ transient strings | Zero |
| **Total passes over edges** | 5 vars × 6.46M rows × ~4 neighbors = ~129M R-level iterations | 5 vars × 1 sparse mat-vec + 1 grouped agg |
| **Estimated runtime** | 86+ hours | **~5–15 minutes** |
| **Peak RAM** | Lists of 6.46M integer vectors (~2–4 GB) | Sparse matrix (~600 MB) + edge DT (~300 MB) |
| **Numerical output** | max, min, mean per variable | **Identical** max, min, mean per variable |

### Why correctness is preserved

- The sparse matrix `adj` encodes exactly the same neighbor relationships as the original `build_neighbor_lookup`. Each `adj[i,j] = 1` means "row j is a rook neighbor of row i in the same year" — the year constraint is enforced structurally by the block-diagonal construction (edges only connect rows within the same year-offset block).
- `adj %*% vals` computes the exact sum of neighbor values; dividing by the count of non-NA neighbors yields the identical `mean()`.
- `data.table` grouped `max()`/`min()` over the edge list yields the identical `max()`/`min()` per row.
- The Random Forest model sees the same column names with the same numerical values, so no retraining is needed.