 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** as the dominant bottleneck — specifically `build_neighbor_lookup` and `compute_neighbor_stats`.

Here's why:

1. **`build_neighbor_lookup`** uses a single-threaded `lapply` over **~6.46 million rows**. For each row, it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) hash-table probing per call, and doing this 6.46 million times with string construction each time is extremely expensive.

2. **`compute_neighbor_stats`** then iterates over the 6.46 million-element neighbor lookup list, computing `max`, `min`, and `mean` for each entry. This is called **5 times** (once per neighbor source variable), yielding ~32.3 million list iterations.

3. The Random Forest step is a **single call** to `predict()` on a pre-trained model. Even with 6.46M rows and 110 predictors, modern `ranger` or `randomForest` predict calls are internally vectorized in C/C++ and typically complete in seconds to minutes — not hours.

**The 86+ hour runtime is dominated by the O(n) string-based lookups and R-level list iteration in the neighbor feature engineering, not by RF inference.**

---

## Optimization Strategy

1. **Eliminate string-keyed lookups entirely.** Replace the `paste(id, year, sep="_")` → named-vector lookup with a direct integer-indexed matrix. Pre-build a `(cell_index × year_index) → row_index` integer matrix for O(1) positional access.

2. **Vectorize neighbor stats computation.** Instead of `lapply` over 6.46M elements, unroll the neighbor lookup into a flat vector of `(source_row, neighbor_row)` pairs, extract values with vectorized indexing, then use `data.table` grouped aggregation (`max`, `min`, `mean`) in a single pass.

3. **Build the neighbor-to-row mapping once** and reuse it for all 5 variables, avoiding repeated iteration.

4. **Preserve the trained RF model and the original numerical estimand** — we only change the feature engineering, not the model or the prediction target.

---

## Working R Code

```r
library(data.table)

# ==============================================================
# OPTIMIZED NEIGHBOR FEATURE ENGINEERING
# ==============================================================

build_neighbor_edges <- function(data_dt, id_order, neighbors) {

# data_dt: a data.table with columns 'id', 'year', and a .ROW_IDX column
# id_order: integer vector of cell IDs in the order matching 'neighbors'
# neighbors: spdep nb object (list of integer index vectors into id_order)
#
# Returns a data.table with columns: source_row, neighbor_row
# representing all (row_i, row_j) pairs where j is a rook neighbor of i
# in the same year.

  # Step 1: Map cell id -> position in id_order (1-based)
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

  # Step 2: Map (cell_position, year) -> row index using an integer matrix
  #          Rows = cell positions (1..n_cells), Cols = year indices (1..n_years)
  years_unique <- sort(unique(data_dt$year))
  n_cells <- length(id_order)
  n_years <- length(years_unique)
  year_to_col <- setNames(seq_along(years_unique), as.character(years_unique))

  # Build the lookup matrix: cell_pos x year_col -> row index
  cell_pos_vec <- id_to_pos[as.character(data_dt$id)]
  year_col_vec <- year_to_col[as.character(data_dt$year)]
  row_idx_vec  <- data_dt$.ROW_IDX

  lookup_mat <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  lookup_mat[cbind(cell_pos_vec, year_col_vec)] <- row_idx_vec

  # Step 3: Build edge list (source_row, neighbor_row)
  # For each cell position, get its neighbor positions from the nb object.
  # Then for every year, pair source row with neighbor rows.

  # Pre-compute: for each cell_pos, which neighbor positions?
  # Flatten into (cell_pos, neighbor_pos) pairs at the cell level.
  cell_from <- rep(seq_along(neighbors), lengths(neighbors))
  cell_to   <- unlist(neighbors, use.names = FALSE)

  # Remove zero-length / self-referencing if any (spdep nb can have 0L entries)
  valid <- cell_to > 0L
  cell_from <- cell_from[valid]
  cell_to   <- cell_to[valid]

  n_edges_cell <- length(cell_from)

  # Now expand across all years: each cell-level edge becomes n_years row-level edges
  # Use vectorized matrix indexing
  source_rows   <- lookup_mat[cbind(
    rep(cell_from, times = n_years),
    rep(seq_len(n_years), each = n_edges_cell)
  )]
  neighbor_rows <- lookup_mat[cbind(
    rep(cell_to, times = n_years),
    rep(seq_len(n_years), each = n_edges_cell)
  )]

  # Remove pairs where either source or neighbor row is NA (missing cell-year)
  valid2 <- !is.na(source_rows) & !is.na(neighbor_rows)

  data.table(
    source_row   = source_rows[valid2],
    neighbor_row = neighbor_rows[valid2]
  )
}


compute_and_add_all_neighbor_features <- function(cell_data, id_order,
                                                   neighbors, var_names) {
# cell_data: data.frame or data.table with columns id, year, and all var_names
# id_order:  integer vector of cell IDs matching the nb object
# neighbors: spdep nb list
# var_names: character vector of variable names to compute neighbor stats for
#
# Returns cell_data (data.table) with new columns:
#   <var>_neighbor_max, <var>_neighbor_min, <var>_neighbor_mean
# for each var in var_names.

  dt <- as.data.table(cell_data)
  dt[, .ROW_IDX := .I]

  # Build the edge table once (the expensive part, but now vectorized)
  message("Building neighbor edge table...")
  edges <- build_neighbor_edges(dt, id_order, neighbors)
  message(sprintf("  Edge table: %s row-level neighbor pairs", format(nrow(edges), big.mark = ",")))

  # For each variable, compute grouped stats via data.table
  for (vn in var_names) {
    message(sprintf("Computing neighbor stats for: %s", vn))

    # Attach the neighbor's value to each edge
    edges[, val := dt[[vn]][neighbor_row]]

    # Remove NA values before aggregation
    edges_valid <- edges[!is.na(val)]

    # Grouped aggregation: one pass per variable
    agg <- edges_valid[, .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ), by = source_row]

    # Initialize new columns with NA
    max_col  <- paste0(vn, "_neighbor_max")
    min_col  <- paste0(vn, "_neighbor_min")
    mean_col <- paste0(vn, "_neighbor_mean")

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Fill in computed values by row index
    set(dt, i = agg$source_row, j = max_col,  value = agg$nb_max)
    set(dt, i = agg$source_row, j = min_col,  value = agg$nb_min)
    set(dt, i = agg$source_row, j = mean_col, value = agg$nb_mean)
  }

  # Clean up helper column
  dt[, .ROW_IDX := NULL]
  edges[, val := NULL]

  return(dt[])
}


# ==============================================================
# USAGE — drop-in replacement for the original outer loop
# ==============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_and_add_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# ==============================================================
# RANDOM FOREST PREDICTION (unchanged — not the bottleneck)
# ==============================================================
# The pre-trained model is loaded and used as-is.
# Example (assuming ranger):
#   rf_model   <- readRDS("trained_rf_model.rds")
#   predictions <- predict(rf_model, data = cell_data)$predictions
#   cell_data$predicted <- predictions
#   fwrite(cell_data, "final_predictions.csv")
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | `lapply` over 6.46M rows with `paste` + named-vector string lookup | Integer matrix `(cell_pos, year_col) → row_idx`, O(1) positional access |
| **Edge expansion** | Implicit per-row; repeated 6.46M times | Vectorized `cbind` indexing across all cells × years in one shot |
| **Neighbor stats** | `lapply` over 6.46M-element list, called 5× | `data.table` grouped aggregation (`by = source_row`), one pass per variable |
| **Estimated runtime** | 86+ hours | Minutes (vectorized C-backed operations in data.table) |
| **RF model** | Preserved ✓ | Preserved ✓ |
| **Numerical estimand** | Preserved ✓ (same max/min/mean logic) | Preserved ✓ |

The optimization reduces the runtime from **86+ hours to the order of minutes** by eliminating millions of R-level string operations and replacing scalar list iteration with vectorized integer indexing and `data.table` grouped aggregation.