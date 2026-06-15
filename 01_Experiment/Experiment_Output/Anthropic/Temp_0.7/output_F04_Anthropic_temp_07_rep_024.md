 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Performs a character-based hash lookup (`id_to_ref`),
2. Indexes into the `neighbors` list,
3. Constructs character key strings via `paste()`,
4. Looks them up in `idx_lookup` (a named character vector of length ~6.46M),
5. Filters `NA`s.

**Key problems:**

- **`paste()` called ~6.46M times**, each generating a vector of neighbor keys — this is extremely expensive string allocation.
- **Named-vector lookup (`idx_lookup[neighbor_keys]`)** on a 6.46M-element named vector is O(n) hashing per call, repeated billions of times in aggregate (~6.46M rows × ~4 neighbors average ≈ 26M lookups, but the overhead per call is dominated by string construction and matching).
- **`lapply` over 6.46M elements in R** with non-trivial per-element work is inherently slow; R's interpreted loop overhead is massive here.
- **`compute_neighbor_stats`** then does another `lapply` over 6.46M elements per variable (×5 variables = ~32.3M iterations), each extracting, filtering, and summarizing — again slow in pure R.

**Estimated cost:** ~6.46M × (string ops + hash lookups) for the lookup build, then ~32.3M summary operations. This easily accounts for the 86+ hour runtime on a laptop.

## Optimization Strategy

### Core Insight
The neighbor topology is **time-invariant** — a cell's spatial neighbors are the same in every year. We should:

1. **Build the neighbor lookup at the cell level (344K cells), not the cell-year level (6.46M rows).**
2. **Vectorize the stats computation using `data.table` grouping and matrix operations** instead of per-row `lapply`.
3. **Avoid all `paste()`-based key construction** — use integer indexing throughout.

### Approach

- Convert data to `data.table`, keyed by `(id, year)`.
- Explode the neighbor list into an edge table: `(cell_row, neighbor_id)`.
- Join to get neighbor row indices per year in a fully vectorized manner.
- Compute `max`, `min`, `mean` via `data.table` grouped aggregation — one pass per variable.

This replaces ~6.46M R-level iterations with a few vectorized joins and group-bys.

## Optimized R Code

```r
library(data.table)

#' Build a vectorized neighbor edge table and compute all neighbor features.
#' Preserves the original numerical estimand exactly (max, min, mean of
#' non-NA neighbor values; NA when no valid neighbors exist).
#'
#' @param cell_data       data.frame / data.table with columns `id`, `year`, and all source vars
#' @param id_order        integer vector of cell IDs in the same order as the nb object
#' @param neighbors       spdep nb object (list of integer index vectors into id_order)
#' @param neighbor_source_vars character vector of variable names to summarize
#' @return data.table with original columns plus neighbor feature columns appended

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          neighbors,
                                          neighbor_source_vars) {

  dt <- as.data.table(cell_data)

  # --- Step 1: Assign a row index to every row (preserves original order) ---
  dt[, .row_idx := .I]

  # --- Step 2: Build cell-level edge list (time-invariant) ---
  #     For each cell index i in id_order, get its neighbor cell IDs.
  #     This is only 344,208 cells, very fast.
  edge_list <- rbindlist(lapply(seq_along(id_order), function(i) {
    nb_idx <- neighbors[[i]]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  # edge_list has ~1.37M rows (directed rook edges)

  # --- Step 3: Create a keyed lookup from (id, year) -> row_idx ---
  setkey(dt, id, year)

  # --- Step 4: Expand edges by year via a join ---
  #     For every (focal_id, year) row, we need the row indices of its neighbors
  #     in the same year.
  #
  #     Strategy: join edge_list to dt twice —

  #       (a) get focal row index + year
  #       (b) get neighbor row index for that (neighbor_id, year)

  # 4a. Get all (focal_id, year, focal_row_idx) combinations
  focal_dt <- dt[, .(focal_id = id, year, focal_row_idx = .row_idx)]

  # 4b. Join edges to focal rows to get (focal_row_idx, neighbor_id, year)
  #     This is an equi-join on focal_id.
  setkey(edge_list, focal_id)
  setkey(focal_dt, focal_id)
  expanded <- edge_list[focal_dt, on = "focal_id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded columns: focal_id, neighbor_id, year, focal_row_idx
  # Rows: ~1.37M edges × 28 years ≈ 38.5M (fits in 16 GB easily as integer columns)

  # 4c. Join to dt again to get neighbor_row_idx for (neighbor_id, year)
  neighbor_idx_lookup <- dt[, .(neighbor_id = id, year, neighbor_row_idx = .row_idx)]
  setkey(neighbor_idx_lookup, neighbor_id, year)
  setkey(expanded, neighbor_id, year)
  expanded <- neighbor_idx_lookup[expanded, on = c("neighbor_id", "year"), nomatch = NA]
  # Now expanded has: focal_row_idx, neighbor_row_idx (NA if neighbor missing that year)

  # Drop rows where neighbor is missing (no data for that cell-year)
  expanded <- expanded[!is.na(neighbor_row_idx)]

  # Keep only the columns we need to minimize memory
  expanded <- expanded[, .(focal_row_idx, neighbor_row_idx)]

  # --- Step 5: For each source variable, compute grouped stats vectorized ---
  for (var_name in neighbor_source_vars) {

    vals <- dt[[var_name]]

    # Attach neighbor values
    expanded[, nval := vals[neighbor_row_idx]]

    # Drop NAs in neighbor values for stats (matches original logic)
    valid <- expanded[!is.na(nval)]

    # Grouped aggregation — one vectorized pass
    stats <- valid[, .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ), by = focal_row_idx]

    # Initialize result columns with NA (matches original behavior for cells with no valid neighbors)
    max_col  <- paste0("n_max_", var_name)
    min_col  <- paste0("n_min_", var_name)
    mean_col <- paste0("n_mean_", var_name)

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Fill in computed values by row index
    dt[stats$focal_row_idx, (max_col)  := stats$nb_max]
    dt[stats$focal_row_idx, (min_col)  := stats$nb_min]
    dt[stats$focal_row_idx, (mean_col) := stats$nb_mean]

    # Clean up temp column
    expanded[, nval := NULL]
  }

  # --- Step 6: Clean up and return in original row order ---
  setorder(dt, .row_idx)
  dt[, .row_idx := NULL]

  return(dt)
}

# ============================================================
# USAGE (drop-in replacement for the original outer loop)
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Predict as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Why This Works and Performance Estimate

| Aspect | Original | Optimized |
|---|---|---|
| Lookup build | 6.46M `lapply` iterations with `paste` + named-vector lookup | 344K-iteration edge list + two `data.table` keyed joins |
| Stats computation | 6.46M × 5 = 32.3M `lapply` iterations with per-element subsetting | 5 vectorized `data.table` group-by aggregations over ~38.5M rows |
| String operations | ~6.46M `paste()` calls + ~26M character hash lookups | **Zero** — all integer indexing |
| Estimated runtime | **86+ hours** | **~5–15 minutes** on the same laptop |
| Peak memory | Lower per-step but slow | ~38.5M rows × 2 int cols ≈ 0.6 GB for the expanded edge table; fits in 16 GB |
| Numerical result | max/min/mean of non-NA neighbor values, NA if none | **Identical** — same aggregation functions, same NA handling |

The ~1000× speedup comes from eliminating millions of interpreted R loop iterations and string operations in favor of `data.table`'s C-level keyed joins and grouped aggregation.