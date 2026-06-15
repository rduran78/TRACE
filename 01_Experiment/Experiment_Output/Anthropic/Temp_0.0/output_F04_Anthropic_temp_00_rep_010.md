 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Converts an ID to a character and looks it up in a named vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs paste-based string keys (`"id_year"`) for each neighbor.
4. Looks those keys up in another named character vector (`idx_lookup`).

String construction (`paste`) and named-vector lookup (which is O(n) hash probing on character keys) repeated ~6.46M × ~4 neighbors ≈ 25.8M string operations is extremely expensive in base R. The secondary bottleneck is `compute_neighbor_stats`, which uses an `lapply` over 6.46M elements returning 3-element vectors, then `do.call(rbind, ...)` on a 6.46M-element list — a notoriously slow pattern.

**Root causes, ranked by impact:**

1. **Per-row string key construction and lookup in `build_neighbor_lookup`**: O(N × avg_neighbors) string pastes and hash lookups on character vectors.
2. **`do.call(rbind, list_of_vectors)`** in `compute_neighbor_stats`: incrementally builds a huge matrix from millions of tiny vectors.
3. **Repeated R-level iteration** (`lapply`) over 6.46M rows without vectorization.

## Optimization Strategy

1. **Replace string-key lookups with integer-indexed sparse matrix representation.** Build a sparse adjacency structure once using `data.table` integer joins instead of character paste/match. For each row `i`, we need the set of rows `j` that share the same year and whose cell ID is a rook neighbor. This is a merge/join problem, not a per-row string problem.

2. **Use `data.table` for the join** to construct a two-column edge list `(row_i, row_j)` representing "row j is a spatial neighbor of row i in the same year." This replaces the entire `build_neighbor_lookup` function.

3. **Vectorize `compute_neighbor_stats`** using grouped aggregation on the edge list via `data.table`, eliminating the `lapply` + `do.call(rbind, ...)` pattern entirely.

4. **Process all 5 variables in a single grouped aggregation** instead of looping over variables.

This reduces the complexity from ~6.46M R-level iterations with string operations to a single vectorized `data.table` merge + grouped aggregation.

## Optimized Working R Code

```r
library(data.table)

#' Build a data.table edge list: for every row in cell_data, find all rows
#' that are (a) rook neighbors and (b) in the same year.
#' Returns a data.table with columns: row_i, row_j
build_neighbor_edgelist <- function(cell_data_dt, id_order, rook_neighbors_unique) {

  # --- Step 1: Build a cell-level neighbor edge list (integer IDs) ---
  # id_order is the vector of cell IDs in the order matching the nb object.
  # rook_neighbors_unique[[k]] gives integer indices into id_order for
  # neighbors of id_order[k].

  # Expand nb object into a two-column data.table of (from_id, to_id)
  from_idx <- rep(
    seq_along(rook_neighbors_unique),
    lengths(rook_neighbors_unique)
  )
  to_idx <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Remove the 0-neighbor sentinel if spdep uses 0L for "no neighbors"
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  cell_edges <- data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )
  # cell_edges now has ~1,373,394 rows (directed rook-neighbor pairs)

  # --- Step 2: Join with cell_data to expand to row-level edges ---
  # We need: for each (from_id, year) row, find all (to_id, year) rows.

  # Create a minimal lookup: cell id + year -> row index
  # Ensure cell_data_dt has a row_idx column
  cell_data_dt[, row_idx := .I]

  # Keyed lookup tables
  from_lookup <- cell_data_dt[, .(row_i = row_idx, from_id = id, year)]
  to_lookup   <- cell_data_dt[, .(row_j = row_idx, to_id = id, year)]

  # Merge cell_edges with from_lookup on from_id, then with to_lookup on

  # (to_id, year). This is the key vectorized operation.
  # First join: attach row indices and years for the "from" side
  setkey(cell_edges, from_id)
  setkey(from_lookup, from_id)
  edges_with_from <- cell_edges[from_lookup,
    on = "from_id",
    allow.cartesian = TRUE,
    nomatch = 0L
  ]
  # edges_with_from has columns: from_id, to_id, row_i, year

  # Second join: attach row indices for the "to" side, matching on (to_id, year)
  setkey(edges_with_from, to_id, year)
  setkey(to_lookup, to_id, year)
  full_edges <- edges_with_from[to_lookup,
    on = c("to_id", "year"),
    nomatch = 0L
  ]
  # full_edges has columns: from_id, to_id, row_i, year, row_j

  full_edges[, .(row_i, row_j)]
}


#' Compute neighbor max, min, mean for multiple variables at once,
#' using the precomputed edge list. Returns the original data.table
#' with new columns appended.
compute_all_neighbor_features <- function(cell_data_dt, edge_dt, neighbor_source_vars) {

  n <- nrow(cell_data_dt)

  # Attach neighbor variable values via the edge list
  # edge_dt$row_j indexes into cell_data_dt for the neighbor row
  # We pull all source variable values for the neighbor rows at once.

  neighbor_vals <- cell_data_dt[edge_dt$row_j, ..neighbor_source_vars]
  neighbor_vals[, row_i := edge_dt$row_i]

  # Grouped aggregation: for each row_i, compute max/min/mean of each variable
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  # Build the aggregation call programmatically
  # Using a simpler and more robust approach: melt + dcast or direct computation
  stats <- neighbor_vals[,
    {
      out <- vector("list", length(neighbor_source_vars) * 3L)
      k <- 1L
      for (v in neighbor_source_vars) {
        vals <- .SD[[v]]
        vals <- vals[!is.na(vals)]
        if (length(vals) == 0L) {
          out[[k]]     <- NA_real_
          out[[k + 1]] <- NA_real_
          out[[k + 2]] <- NA_real_
        } else {
          out[[k]]     <- max(vals)
          out[[k + 1]] <- min(vals)
          out[[k + 2]] <- mean(vals)
        }
        k <- k + 3L
      }
      names(out) <- agg_names
      out
    },
    by = row_i,
    .SDcols = neighbor_source_vars
  ]

  # Rows with no neighbors at all won't appear in stats.
  # Create a full-index frame and left-join.
  all_rows <- data.table(row_i = seq_len(n))
  stats <- stats[all_rows, on = "row_i"]

  # Replace Inf/-Inf from max/min of empty sets with NA (safety)
  for (col_name in agg_names) {
    vals <- stats[[col_name]]
    set(stats, which(is.infinite(vals)), col_name, NA_real_)
  }

  # Bind new columns to cell_data_dt
  for (col_name in agg_names) {
    set(cell_data_dt, j = col_name, value = stats[[col_name]])
  }

  cell_data_dt
}


# ============================================================
# MAIN EXECUTION
# ============================================================

# Convert to data.table if not already
cell_data_dt <- as.data.table(cell_data)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Step 1: Build edge list (replaces build_neighbor_lookup)
# Expected size: ~1.37M cell-edges × 28 years ≈ ~38.5M row-pairs
message("Building neighbor edge list...")
edge_dt <- build_neighbor_edgelist(cell_data_dt, id_order, rook_neighbors_unique)
message(sprintf("Edge list built: %s row-pairs", format(nrow(edge_dt), big.mark = ",")))

# Step 2: Compute all neighbor features at once (replaces the for loop)
message("Computing neighbor features...")
cell_data_dt <- compute_all_neighbor_features(cell_data_dt, edge_dt, neighbor_source_vars)
message("Neighbor features complete.")

# Convert back to data.frame if downstream code expects it
cell_data <- as.data.frame(cell_data_dt)

# The trained Random Forest model is unchanged and can be applied directly:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Expected Performance Improvement

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `paste` + character hash lookups | One `data.table` merge on integer keys (~38.5M rows, vectorized C) |
| `compute_neighbor_stats` | 6.46M R-level `lapply` iterations × 5 variables | Single grouped `data.table` aggregation |
| `do.call(rbind, ...)` | Binds 6.46M 3-element vectors × 5 calls | Eliminated entirely |
| **Estimated wall time** | **86+ hours** | **~5–15 minutes** (dominated by the ~38.5M-row join) |
| **RAM peak** | Moderate but slow | ~38.5M-row edge table ≈ ~600 MB; fits in 16 GB |
| **Numerical result** | max/min/mean of non-NA rook-neighbor values | Identical — same aggregation logic preserved |
| **RF model** | Unchanged | Unchanged — no retraining |

The key insight is that the original code solves a **batch equi-join** problem (match on neighbor-ID + year) using a **per-row string-key loop**, which is the worst-case pattern for R performance. Replacing it with a single vectorized `data.table` join on integer keys eliminates millions of R interpreter overhead cycles and leverages optimized C-level hash joins and grouped aggregation.