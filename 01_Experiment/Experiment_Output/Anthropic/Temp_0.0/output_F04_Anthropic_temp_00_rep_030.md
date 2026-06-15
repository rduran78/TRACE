 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Performs a character-based hash lookup (`id_to_ref`) — fine individually, but done 6.46M times.
2. Subsets the `neighbors` list to get neighbor cell IDs.
3. Constructs character paste keys (`paste(id, year, sep="_")`) for every neighbor of every row.
4. Looks those keys up in `idx_lookup` (a named character vector of length 6.46M — each lookup is O(n) in the worst case for R named vectors, though internally hashed).
5. Filters `NA`s.

The result is **~6.46 million R-level iterations**, each allocating multiple small character vectors and performing multiple hash lookups. This is catastrophically slow in interpreted R. `compute_neighbor_stats` is a secondary bottleneck (another 6.46M-iteration `lapply` per variable), but lighter per iteration.

**Root cause summary:** Row-level `lapply` with per-row string construction and lookup over millions of rows in pure R.

## Optimization Strategy

**Core idea:** Eliminate the row-level loop entirely. Replace it with a vectorized join using `data.table`. Instead of building a per-row list of neighbor indices, build a **long-format edge table** (source_row → neighbor_row) once, then compute all neighbor statistics via grouped aggregation — all in compiled C code under the hood.

### Steps

1. **Vectorized neighbor lookup construction:** Expand the `nb` object into a long `data.table` of `(cell_id, neighbor_cell_id)` pairs. Join with the panel data on `(neighbor_cell_id, year)` to get `(source_row, neighbor_row)` pairs. No per-row `lapply`.

2. **Vectorized neighbor stats:** For each variable, index into the values vector using the neighbor_row column, then group-by source_row to compute `max`, `min`, `mean`. This is a single `data.table` grouped aggregation — extremely fast.

3. **Memory:** The edge table will have ~6.46M rows × (avg ~4 rook neighbors) ≈ 25.8M rows × a few integer/double columns — well within 16 GB.

## Working R Code

```r
library(data.table)

# ── Step 1: Build vectorized edge table (replaces build_neighbor_lookup) ──────

build_neighbor_edges <- function(cell_data_dt, id_order, neighbors) {
  # Expand nb object into a long data.table of (cell_id, neighbor_cell_id)
  # neighbors is a list of integer index vectors into id_order
  n_cells <- length(id_order)
  from_idx <- rep(seq_len(n_cells), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  edges <- data.table(
    cell_id          = id_order[from_idx],
    neighbor_cell_id = id_order[to_idx]
  )

  # Create a row-index column in the panel data
  cell_data_dt[, .row_idx := .I]

  # For each edge × year combination, find the source row and neighbor row.
  # Source side: join edges with panel on cell_id to get (source_row, year, neighbor_cell_id)
  source_keys <- cell_data_dt[, .(cell_id = id, year, source_row = .row_idx)]
  setkey(source_keys, cell_id)
  setkey(edges, cell_id)

  # Merge: for every (cell_id, neighbor_cell_id) edge, replicate across all years
  # of that cell_id present in the panel
  edge_year <- merge(edges, source_keys, by = "cell_id", allow.cartesian = TRUE)
  # edge_year now has: cell_id, neighbor_cell_id, year, source_row

  # Neighbor side: find the row index of each (neighbor_cell_id, year)
  neighbor_keys <- cell_data_dt[, .(neighbor_cell_id = id, year, neighbor_row = .row_idx)]
  setkey(neighbor_keys, neighbor_cell_id, year)
  setkey(edge_year, neighbor_cell_id, year)

  # Inner join: drops edges where the neighbor doesn't exist in that year
  edge_full <- merge(edge_year, neighbor_keys, by = c("neighbor_cell_id", "year"))
  # edge_full has: neighbor_cell_id, year, cell_id, source_row, neighbor_row

  # We only need source_row and neighbor_row
  edge_full[, .(source_row, neighbor_row)]
}

# ── Step 2: Compute neighbor stats vectorized (replaces compute_neighbor_stats) ─

compute_neighbor_stats_vec <- function(cell_data_dt, edge_dt, var_name) {
  n <- nrow(cell_data_dt)
  vals <- cell_data_dt[[var_name]]

  # Attach neighbor values
  work <- edge_dt[, .(source_row, nval = vals[neighbor_row])]
  # Remove NA neighbor values
  work <- work[!is.na(nval)]

  # Grouped aggregation
  agg <- work[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = source_row]

  # Initialize output columns with NA
  out_max  <- rep(NA_real_, n)
  out_min  <- rep(NA_real_, n)
  out_mean <- rep(NA_real_, n)

  out_max[agg$source_row]  <- agg$nb_max
  out_min[agg$source_row]  <- agg$nb_min
  out_mean[agg$source_row] <- agg$nb_mean

  data.table(nb_max = out_max, nb_min = out_min, nb_mean = out_mean)
}

# ── Step 3: Main pipeline (replaces outer loop) ──────────────────────────────

run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {

  cell_data_dt <- as.data.table(cell_data)

  message("Building edge table...")
  edge_dt <- build_neighbor_edges(cell_data_dt, id_order, rook_neighbors_unique)
  setkey(edge_dt, source_row)
  message(sprintf("Edge table: %s rows", format(nrow(edge_dt), big.mark = ",")))

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))
    stats <- compute_neighbor_stats_vec(cell_data_dt, edge_dt, var_name)

    col_max  <- paste0(var_name, "_nb_max")
    col_min  <- paste0(var_name, "_nb_min")
    col_mean <- paste0(var_name, "_nb_mean")

    set(cell_data_dt, j = col_max,  value = stats$nb_max)
    set(cell_data_dt, j = col_min,  value = stats$nb_min)
    set(cell_data_dt, j = col_mean, value = stats$nb_mean)
  }

  # Clean up helper column
  cell_data_dt[, .row_idx := NULL]

  # Return as data.frame if original was data.frame
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    return(as.data.frame(cell_data_dt))
  }
  cell_data_dt
}

# ── Usage ─────────────────────────────────────────────────────────────────────
# cell_data <- run_neighbor_feature_pipeline(cell_data, id_order, rook_neighbors_unique)
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

## Why This Works and What It Preserves

| Concern | Status |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` over the identical neighbor sets produce identical values. The inner join replicates the original `!is.na` filter. |
| **Trained RF model untouched** | Only the feature-construction step is replaced; the model object and `predict()` call are unchanged. |
| **Column names** | Must match what the RF expects — adjust the `paste0` naming pattern to match your existing `compute_and_add_neighbor_features` output names. |
| **Memory** | Edge table ≈ 25.8M rows × 2 int columns ≈ 0.4 GB. Temporary `work` table per variable ≈ 0.6 GB peak. Well within 16 GB. |
| **Expected speedup** | The `merge` and grouped `data.table` operations run in compiled C. Expected wall time: **~2–10 minutes** total (vs. 86+ hours), roughly a **500–2500× speedup**. |

The key insight is that the original code's per-row `lapply` with string construction is replaced by two bulk equi-joins and five grouped aggregations — all operations `data.table` executes at near-C speed via radix-based indexing.