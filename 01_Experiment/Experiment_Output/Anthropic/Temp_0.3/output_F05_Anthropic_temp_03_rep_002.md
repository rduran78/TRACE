 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### Local Inefficiency
Inside `build_neighbor_lookup`, the `idx_lookup` named vector (6.46M entries) is built once, but then **6.46 million `paste()` + named-vector lookups** are performed inside the `lapply` — each call to `paste(neighbor_cell_ids, data$year[i], sep = "_")` constructs temporary character vectors and does hash-table lookups against a 6.46M-entry named character vector. String hashing at this scale is extremely expensive.

### Broader Algorithmic Problem
The real issue is that **the entire neighbor-lookup concept is being solved in string-key space when it should be solved in integer-index space**. The data has a regular panel structure (each cell appears once per year), so the mapping from `(cell_id, year)` → row index can be represented as a **dense integer matrix** (cell × year), turning every neighbor lookup into a direct integer-indexed matrix access — O(1) with no hashing.

Furthermore, `compute_neighbor_stats` is called **5 separate times**, each time iterating over 6.46M rows and chasing the same neighbor indices. These passes can be **fused into a single pass** or, better yet, **fully vectorized** using sparse-matrix multiplication or data.table joins.

### Cost Breakdown (Current)
| Step | Operations | Cost Driver |
|---|---|---|
| `paste()` for `idx_lookup` | 6.46M string concatenations | One-time, tolerable |
| `paste()` inside `lapply` | ~6.46M × ~4 neighbors = ~25.8M concatenations | Dominant cost |
| Named-vector lookup | ~25.8M hash lookups against 6.46M-entry table | Dominant cost |
| `compute_neighbor_stats` | 5 vars × 6.46M rows × index chasing | Repeated traversal |

**Estimated total: 86+ hours** — almost entirely from the string operations and R-level loop overhead.

---

## Optimization Strategy

1. **Replace string keys with a dense integer lookup matrix** `row_matrix[cell_index, year_index]` → row number. Lookup becomes a single integer matrix access.

2. **Pre-build all neighbor row-indices as a single integer operation** using vectorized construction — no `lapply` over 6.46M rows.

3. **Compute all 5 variables' neighbor stats in one vectorized pass** using a sparse adjacency matrix (or a single grouped data.table operation), eliminating per-row R-level loops entirely.

4. **Preserve exact numerical output**: max, min, mean of non-NA neighbor values per cell-year, per variable — identical column names and values.

---

## Working R Code

```r
library(data.table)
library(Matrix)

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                       neighbor_source_vars) {
  # ---------------------------------------------------------------
  # STEP 1: Build dense integer lookup matrix  (cell_index × year_index) → row

  # ---------------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]

  # Create integer mappings
  unique_ids   <- as.character(id_order)
  unique_years <- sort(unique(dt$year))

  id_to_int   <- setNames(seq_along(unique_ids), unique_ids)
  year_to_int <- setNames(seq_along(unique_years), as.character(unique_years))

  n_ids   <- length(unique_ids)
  n_years <- length(unique_years)

  # Dense lookup matrix: row_matrix[cell_int, year_int] = row index in dt
  # Initialize with NA
  row_matrix <- matrix(NA_integer_, nrow = n_ids, ncol = n_years)

  cell_ints <- id_to_int[as.character(dt$id)]
  year_ints <- year_to_int[as.character(dt$year)]
  row_matrix[cbind(cell_ints, year_ints)] <- dt$row_idx

  cat("Step 1 complete: dense lookup matrix built (",
      n_ids, "cells x", n_years, "years )\n")

  # ---------------------------------------------------------------
  # STEP 2: Build sparse directed neighbor adjacency in cell-index space

  # ---------------------------------------------------------------
  # rook_neighbors_unique is an nb object: list of length n_ids,
  # each element is an integer vector of neighbor indices into id_order
  # (with 0L meaning no neighbors per spdep convention)

  # Build edge list (from_cell_int, to_cell_int)
  from_list <- rep(seq_along(rook_neighbors_unique),
                   lengths(rook_neighbors_unique))
  to_list   <- unlist(rook_neighbors_unique)

  # Remove spdep's 0-encoded "no neighbor" entries
  valid <- to_list != 0L
  from_list <- from_list[valid]
  to_list   <- to_list[valid]

  n_edges <- length(from_list)
  cat("Step 2 complete:", n_edges, "directed neighbor edges\n")

  # ---------------------------------------------------------------
  # STEP 3: Expand edges to cell-year level (vectorized)
  # ---------------------------------------------------------------
  # For each year, every edge (i→j) in cell space becomes
  # (row_matrix[i,y] → row_matrix[j,y]) in row space.
  # We vectorize across all years at once.

  # Replicate edges for each year
  from_cell_expanded <- rep(from_list, times = n_years)
  to_cell_expanded   <- rep(to_list,   times = n_years)
  year_int_expanded  <- rep(seq_len(n_years), each = n_edges)

  # Map to row indices
  from_row <- row_matrix[cbind(from_cell_expanded, year_int_expanded)]
  to_row   <- row_matrix[cbind(to_cell_expanded,   year_int_expanded)]

  # Remove pairs where either cell-year doesn't exist in the data
  valid2 <- !is.na(from_row) & !is.na(to_row)
  from_row <- from_row[valid2]
  to_row   <- to_row[valid2]

  cat("Step 3 complete:", sum(valid2),
      "cell-year neighbor pairs constructed\n")

  # Clean up large temporaries
  rm(from_cell_expanded, to_cell_expanded, year_int_expanded, valid2)
  gc()

  # ---------------------------------------------------------------
  # STEP 4: Compute neighbor stats for each variable (vectorized)
  # ---------------------------------------------------------------
  # Strategy: use data.table grouping.
  # Build an edge table: for each "from_row", gather neighbor values from "to_row".
  # Group by from_row, compute max/min/mean.

  edge_dt <- data.table(from_row = from_row, to_row = to_row)
  rm(from_row, to_row)
  gc()

  n_rows <- nrow(dt)

  for (var_name in neighbor_source_vars) {
    cat("  Computing neighbor stats for:", var_name, "...")
    t0 <- proc.time()

    # Attach neighbor values
    vals <- dt[[var_name]]
    edge_dt[, nval := vals[to_row]]

    # Compute grouped stats (excluding NAs)
    stats <- edge_dt[!is.na(nval),
                     .(vmax  = max(nval),
                       vmin  = min(nval),
                       vmean = mean(nval)),
                     by = from_row]

    # Initialize result columns with NA
    col_max  <- paste0("neighbor_max_",  var_name)
    col_min  <- paste0("neighbor_min_",  var_name)
    col_mean <- paste0("neighbor_mean_", var_name)

    dt[, (col_max)  := NA_real_]
    dt[, (col_min)  := NA_real_]
    dt[, (col_mean) := NA_real_]

    # Fill in computed values
    dt[stats$from_row, (col_max)  := stats$vmax]
    dt[stats$from_row, (col_min)  := stats$vmin]
    dt[stats$from_row, (col_mean) := stats$vmean]

    elapsed <- (proc.time() - t0)[3]
    cat(" done in", round(elapsed, 1), "sec\n")
  }

  # Clean up the temporary column
  edge_dt[, nval := NULL]

  # ---------------------------------------------------------------
  # STEP 5: Return as data.frame (preserving compatibility)
  # ---------------------------------------------------------------
  dt[, row_idx := NULL]
  result <- as.data.frame(dt)

  cat("All neighbor features computed.\n")
  return(result)
}

# ===================================================================
# USAGE — drop-in replacement for the original outer loop
# ===================================================================
# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#
# cell_data <- optimize_neighbor_features(
#   cell_data,
#   id_order,
#   rook_neighbors_unique,
#   neighbor_source_vars
# )
#
# # The trained Random Forest model is unchanged — just call predict():
# # predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` are computed on exactly the same set of non-NA neighbor values per cell-year. The grouping key `from_row` is a 1:1 mapping to the original row. |
| **Column names** | Output columns are named `neighbor_max_{var}`, `neighbor_min_{var}`, `neighbor_mean_{var}` — adjust the prefix in the code if your original names differ. |
| **NA handling** | Rows with zero valid neighbors get `NA` for all three stats, matching the original `c(NA, NA, NA)` return. |
| **Trained RF model** | No model retraining. The feature matrix is numerically identical; just pass it to `predict()`. |

---

## Expected Performance

| Step | Estimated Time | Estimated RAM |
|---|---|---|
| Dense lookup matrix | ~2 sec | ~40 MB |
| Edge expansion (vectorized) | ~10 sec | ~600 MB peak |
| Per-variable grouped stats (×5) | ~15–30 sec each | ~500 MB working |
| **Total** | **~3–5 minutes** | **< 8 GB peak** |

This is a **~1,000× speedup** over the original 86+ hour estimate, achieved by eliminating all string operations and R-level row loops in favor of vectorized integer indexing and data.table grouped aggregation.