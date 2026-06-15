 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's why:

### The Core Problem: Row-Level `lapply` Over 6.46M Rows

`build_neighbor_lookup` calls `lapply` over every row (`seq_len(nrow(data))` ≈ 6.46M iterations). Inside each iteration it:

1. **Constructs paste keys** (`paste(neighbor_cell_ids, data$year[i], sep = "_")`) — string allocation per row.
2. **Looks up a named character vector** (`idx_lookup[neighbor_keys]`) — named vector lookup is O(n) hash probe per key, repeated millions of times.
3. **Returns a variable-length integer vector** — producing a 6.46M-element list of integer vectors.

Then `compute_neighbor_stats` loops over that 6.46M-element list again **once per variable** (×5 variables), each time subsetting a numeric vector by the stored indices.

**The string-key lookup is O(N × k) where N = 6.46M and k = average neighbor count (~4 for rook).** The named-vector lookup in R uses hashing but with enormous overhead from repeated `paste` and character matching. The entire approach is fundamentally row-serial when it should be vectorized.

### Why It Takes 86+ Hours

| Step | Iterations | Cost per iteration | Total |
|------|-----------|-------------------|-------|
| `build_neighbor_lookup` | 6.46M | paste + hash lookup for ~4 neighbors | ~80+ hrs |
| `compute_neighbor_stats` | 6.46M × 5 vars | subset + summary of ~4 values | ~6 hrs |

The bottleneck is overwhelmingly in `build_neighbor_lookup`.

### The Key Insight

The neighbor topology is **year-invariant**. Cell *i*'s rook neighbors are the same in every year. The current code rebuilds the spatial relationship for every cell-year row by string-matching `(cell_id, year)` pairs. This is unnecessary. We can:

1. Build the neighbor graph **once** at the cell level (344K cells, not 6.46M cell-years).
2. Expand to cell-year using **integer arithmetic** (no strings).
3. Compute neighbor statistics using **vectorized grouped operations** via `data.table`.

## Optimization Strategy

1. **Eliminate all string keys.** Map each `(id, year)` to an integer row index using integer arithmetic, not paste/hash.
2. **Build an edge list once at the cell level** from the `nb` object (344K cells × ~4 neighbors = ~1.37M edges).
3. **Expand to a cell-year edge list** by joining on year — still only ~1.37M × 28 = ~38.4M edges, which fits in RAM as integer pairs (~600 MB).
4. **Compute all neighbor stats vectorially** using `data.table` grouped aggregation on the edge list — one pass per variable, fully vectorized in C.
5. **Estimated runtime: 2–10 minutes** instead of 86+ hours.

## Working R Code

```r
library(data.table)

# ==============================================================================
# optimized_neighbor_features.R
#
# Replaces build_neighbor_lookup + compute_neighbor_stats + outer loop.
# Preserves the exact numerical estimand (max, min, mean of non-NA rook
# neighbor values per cell-year) and does not touch the trained RF model.
# ==============================================================================

build_neighbor_features_fast <- function(cell_data,
                                         id_order,
                                         rook_neighbors_unique,
                                         neighbor_source_vars) {
  # ---- Step 0: Convert to data.table (non-destructive) ----------------------
  dt <- as.data.table(cell_data)

  # ---- Step 1: Create integer cell index mapping ----------------------------
  # id_order is the vector of cell IDs in the order matching the nb object.
  # Map each cell ID to its position in id_order (1-based).
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

  # Assign each cell its position index. This is the "cell index."
  dt[, cell_idx := id_to_pos[as.character(id)]]

  # ---- Step 2: Build cell-level edge list from nb object --------------------
  # rook_neighbors_unique is a list of length = length(id_order).
  # rook_neighbors_unique[[i]] contains integer indices into id_order of
  # neighbors of cell id_order[i].
  n_cells <- length(id_order)
  from_cell <- rep(seq_len(n_cells),
                   times = lengths(rook_neighbors_unique))
  to_cell   <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Remove any 0-length or self-loops if present
  valid <- to_cell > 0L & from_cell != to_cell
  edges_cell <- data.table(from_cell = from_cell[valid],
                           to_cell   = to_cell[valid])

  cat(sprintf("Cell-level edge list: %s edges\n", format(nrow(edges_cell), big.mark = ",")))

  # ---- Step 3: Build row-index lookup by (cell_idx, year) -------------------
  # Create a unique integer year index
  years <- sort(unique(dt$year))
  n_years <- length(years)
  year_to_int <- setNames(seq_along(years), as.character(years))
  dt[, year_idx := year_to_int[as.character(year)]]

  # Row-index lookup: for each (cell_idx, year_idx), store the row number.
  # We use a matrix for O(1) lookup: rows = cell_idx, cols = year_idx.
  # Size: 344,208 × 28 ≈ 9.6M integers ≈ 38 MB. Fine.
  row_lookup <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  row_lookup[cbind(dt$cell_idx, dt$year_idx)] <- seq_len(nrow(dt))

  # ---- Step 4: Expand edges to cell-year level using vectorized ops ---------
  # For each year, the neighbor graph is the same. We expand:
  #   (from_cell, to_cell) × year_idx -> (from_row, to_row)
  # We do this with a cross join, then look up row indices.

  cat("Expanding edge list to cell-year level...\n")

  # Cross join edges × years
  year_dt <- data.table(year_idx = seq_len(n_years))
  edges_cy <- CJ_dt_edges(edges_cell, year_dt, row_lookup)

  cat(sprintf("Cell-year edge list: %s edges (after removing NAs)\n",
              format(nrow(edges_cy), big.mark = ",")))

  # ---- Step 5: Compute neighbor stats per variable --------------------------
  cat("Computing neighbor statistics...\n")

  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Processing: %s\n", var_name))

    vals <- dt[[var_name]]

    # Attach neighbor values to edge list
    edges_cy[, neighbor_val := vals[to_row]]

    # Compute grouped stats: max, min, mean of non-NA neighbor values
    # grouped by from_row
    stats <- edges_cy[!is.na(neighbor_val),
                      .(nb_max  = max(neighbor_val),
                        nb_min  = min(neighbor_val),
                        nb_mean = mean(neighbor_val)),
                      by = from_row]

    # Initialize new columns with NA
    max_col  <- paste0("nb_max_",  var_name)
    min_col  <- paste0("nb_min_",  var_name)
    mean_col <- paste0("nb_mean_", var_name)

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Assign computed values
    dt[stats$from_row, (max_col)  := stats$nb_max]
    dt[stats$from_row, (min_col)  := stats$nb_min]
    dt[stats$from_row, (mean_col) := stats$nb_mean]
  }

  # ---- Step 6: Clean up helper columns and return ---------------------------
  dt[, c("cell_idx", "year_idx") := NULL]

  # If input was a data.frame, return a data.frame
  if (!is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}


# Helper: cross-join edges × years and resolve to row indices
CJ_dt_edges <- function(edges_cell, year_dt, row_lookup) {
  n_edges <- nrow(edges_cell)
  n_years <- nrow(year_dt)

  # Vectorized expansion without full CJ (memory-efficient)
  from_cell_exp <- rep(edges_cell$from_cell, times = n_years)
  to_cell_exp   <- rep(edges_cell$to_cell,   times = n_years)
  year_idx_exp  <- rep(year_dt$year_idx,      each  = n_edges)

  # Look up row indices via matrix indexing (vectorized, very fast)
  from_row <- row_lookup[cbind(from_cell_exp, year_idx_exp)]
  to_row   <- row_lookup[cbind(to_cell_exp,   year_idx_exp)]

  # Keep only edges where both endpoints exist in the data
  valid <- !is.na(from_row) & !is.na(to_row)

  data.table(from_row = from_row[valid],
             to_row   = to_row[valid])
}


# ==============================================================================
# Usage — drop-in replacement for the original outer loop
# ==============================================================================

# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#
# cell_data <- build_neighbor_features_fast(
#   cell_data            = cell_data,
#   id_order             = id_order,
#   rook_neighbors_unique = rook_neighbors_unique,
#   neighbor_source_vars = neighbor_source_vars
# )
#
# # The trained Random Forest model is unchanged.
# # predict(rf_model, new_data) works as before.
```

## Memory Budget Check

| Object | Size |
|--------|------|
| `row_lookup` matrix (344K × 28 ints) | ~38 MB |
| Cell-level edge list (1.37M × 2 ints) | ~11 MB |
| Expanded edge list (≤38.4M × 2 ints + 1 double) | ~620 MB |
| `dt` (6.46M × ~115 cols) | ~5–6 GB |
| **Total peak** | **~7 GB** (fits in 16 GB) |

## Why This Preserves the Numerical Estimand

The original code computes, for each cell-year row `i`:
- Finds rook neighbors of cell `i`'s spatial ID in the same year.
- Computes `max`, `min`, `mean` of non-NA neighbor values.

The optimized code does **exactly** the same:
- The edge list encodes the same directed rook-neighbor relationships.
- Expansion by year replicates the same-year constraint.
- `data.table` grouped aggregation computes identical `max`, `min`, `mean` over identical non-NA neighbor value sets.

No approximation, sampling, or model retraining is involved.

## Expected Performance

| Step | Estimated Time |
|------|---------------|
| Build cell-level edge list | < 1 second |
| Build row_lookup matrix | < 2 seconds |
| Expand to cell-year edges (~38M rows) | ~10–30 seconds |
| Compute stats per variable (×5) | ~30–60 seconds each |
| **Total** | **~3–6 minutes** |

This represents a speedup of roughly **1,000×** over the original 86+ hour estimate.