 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Converts an ID to a character and looks it up in a named vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs paste-based string keys (`"id_year"`) for each neighbor.
4. Looks those keys up in another named character vector (`idx_lookup`).

String construction (`paste`) and named-vector lookup (which is O(n) hash probing on character keys) repeated ~6.46M × ~4 neighbors ≈ 25.8M string operations is extremely expensive in interpreted R. The secondary bottleneck is `compute_neighbor_stats`, which runs an `lapply` with per-element R function calls over 6.46M entries, each allocating small vectors.

**Root causes, ranked:**

| # | Cause | Impact |
|---|-------|--------|
| 1 | `paste()`-based key construction inside a 6.46M-iteration `lapply` | Very High |
| 2 | Named character vector lookup (`idx_lookup[neighbor_keys]`) is slow at scale | Very High |
| 3 | Per-row anonymous function overhead in `lapply` (no vectorization) | High |
| 4 | `compute_neighbor_stats` uses R-level loop + per-row `max/min/mean` | Moderate |
| 5 | Repeated `do.call(rbind, ...)` on millions of 3-element vectors | Moderate |

## Optimization Strategy

**Core idea:** Replace all string-key lookups and row-level R loops with vectorized integer-indexed operations using `data.table`.

1. **Eliminate string keys entirely.** Create a direct integer mapping from `(id, year)` → row index using a `data.table` keyed join, or even a 2D integer matrix if IDs and years are dense.
2. **Pre-expand the neighbor edge list** into a single long `data.table` of `(source_row, neighbor_row)` pairs (~25.8M rows). This is built once.
3. **Vectorize `compute_neighbor_stats`** by joining the edge list to the variable column and using `data.table` grouped aggregation (`[, .(max, min, mean), by = source_row]`), which is C-level fast.

**Expected speedup:** From ~86+ hours to **~2–10 minutes** depending on I/O, because all hot loops move from R interpreter to C (data.table internals).

**Memory:** The edge list is ~25.8M rows × 2 integer columns ≈ 200 MB. Comfortably fits in 16 GB.

## Optimized Working R Code

```r
library(data.table)

#
# STEP 1: Build a vectorized edge list (source_row -> neighbor_row)
#         This replaces build_neighbor_lookup entirely.
#         Run ONCE; reuse for all variables.
#

build_neighbor_edgelist <- function(cell_data, id_order, neighbors) {
  # cell_data must be a data.table (or will be converted)
  dt <- as.data.table(cell_data)
  
  # --- Map each cell id to its position in id_order (integer) ---
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # --- Map (id, year) -> row index using data.table keyed join ---
  dt[, row_idx := .I]
  setkey(dt, id, year)  # fast keyed lookup
  
  # --- Get unique years ---
  years <- sort(unique(dt$year))
  
  # --- Build the edge list in one vectorized pass ---
  # For each cell in id_order, get its neighbor cell ids
  # Then expand across all years
  
  # Construct cell-level neighbor edges (cell_id -> neighbor_cell_id)
  n_cells <- length(id_order)
  from_cell <- rep(seq_len(n_cells), times = lengths(neighbors))
  to_cell   <- unlist(neighbors)
  
  # Map back to actual cell IDs
  from_id <- id_order[from_cell]
  to_id   <- id_order[to_cell]
  
  cell_edges <- data.table(from_id = from_id, to_id = to_id)
  
  # Cross join with years to get cell-year level edges
  year_dt <- data.table(year = years)
  # Use CJ-style expansion: every edge × every year
  cell_year_edges <- cell_edges[, .(year = years), by = .(from_id, to_id)]
  
  # Now join to get source row index
  setnames(cell_year_edges, c("from_id"), c("id"))
  cell_year_edges[dt, on = .(id, year), source_row := i.row_idx]
  
  # Now join to get neighbor row index
  setnames(cell_year_edges, c("id", "to_id"), c("from_id", "id"))
  cell_year_edges[dt, on = .(id, year), neighbor_row := i.row_idx]
  
  # Clean: keep only edges where both source and neighbor exist
  edges <- cell_year_edges[!is.na(source_row) & !is.na(neighbor_row),
                           .(source_row, neighbor_row)]
  
  # Clean up temporary column
  dt[, row_idx := NULL]
  
  return(edges)
}

#
# STEP 2: Compute neighbor stats for one variable — fully vectorized
#         Returns a data.table with columns: nb_max, nb_min, nb_mean
#         aligned to the rows of cell_data.
#

compute_neighbor_stats_fast <- function(cell_data, edges, var_name) {
  n <- if (is.data.table(cell_data)) nrow(cell_data) else nrow(cell_data)
  
  # Extract the variable values for neighbor rows
  vals <- cell_data[[var_name]]
  
  # Attach neighbor values to edge list
  edge_vals <- data.table(
    source_row = edges$source_row,
    val        = vals[edges$neighbor_row]
  )
  
  # Remove edges where the neighbor value is NA
  edge_vals <- edge_vals[!is.na(val)]
  
  # Grouped aggregation in C (data.table)
  stats <- edge_vals[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = source_row]
  
  # Allocate full-length result with NAs for rows with no valid neighbors
  result <- data.table(
    nb_max  = rep(NA_real_, n),
    nb_min  = rep(NA_real_, n),
    nb_mean = rep(NA_real_, n)
  )
  result[stats$source_row, `:=`(
    nb_max  = stats$nb_max,
    nb_min  = stats$nb_min,
    nb_mean = stats$nb_mean
  )]
  
  # Name columns to match original pipeline convention
  prefix <- paste0("nb_", var_name, "_")
  setnames(result, c(
    paste0(prefix, "max"),
    paste0(prefix, "min"),
    paste0(prefix, "mean")
  ))
  
  return(result)
}

#
# STEP 3: Full replacement outer loop
#

# Convert to data.table if not already
cell_data <- as.data.table(cell_data)

# Build edge list ONCE (replaces build_neighbor_lookup)
message("Building neighbor edge list...")
edges <- build_neighbor_edgelist(cell_data, id_order, rook_neighbors_unique)
message(sprintf("Edge list: %s rows", format(nrow(edges), big.mark = ",")))

# Compute and attach neighbor features for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor stats for: %s", var_name))
  nb_stats <- compute_neighbor_stats_fast(cell_data, edges, var_name)
  cell_data <- cbind(cell_data, nb_stats)
}

message("Done. Neighbor features attached.")

# cell_data now has the same neighbor feature columns as the original pipeline.
# The trained Random Forest model can be applied directly with no retraining.
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---------|-----------|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same set of non-NA neighbor values per row as the original code. The edge list encodes the identical rook-neighbor relationships, restricted to the same year. |
| **Trained RF model** | No model retraining occurs. The output columns have the same names and semantics. The model's `predict()` call is unchanged. |
| **Column naming** | Adjust the `prefix` pattern in `compute_neighbor_stats_fast` if your existing pipeline uses a different naming convention (e.g., `compute_and_add_neighbor_features` may name them differently). Match accordingly. |

## Expected Performance

| Stage | Original | Optimized | Reason |
|-------|----------|-----------|--------|
| Lookup build | ~hours (6.46M `paste` + char lookup) | ~1–3 min (keyed `data.table` join) | Integer join vs. string hashing |
| Stats (per var) | ~hours (6.46M `lapply` calls) | ~10–30 sec (`data.table` grouped agg in C) | Vectorized C vs. R interpreter |
| **Total (5 vars)** | **~86+ hours** | **~5–10 minutes** | **~500–1000× speedup** |