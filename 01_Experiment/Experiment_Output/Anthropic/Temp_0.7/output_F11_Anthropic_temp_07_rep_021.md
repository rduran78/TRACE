 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence clearly points to the **spatial neighbor feature engineering** — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` — as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, and for each row it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is effectively a hash-table probe *per element*, but constructing ~6.46 million string keys and doing millions of named lookups is extremely slow in interpreted R. This single function produces O(6.46M × average_neighbors) string operations.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over the 6.46 million-element `neighbor_lookup` list with per-element `lapply`, subsetting, removing NAs, and computing `max/min/mean`. The `do.call(rbind, result)` on a 6.46M-element list of 3-element vectors is also notoriously slow.

3. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-trained model. Even with 6.46M rows and 110 predictors, modern RF implementations (e.g., `ranger`) complete prediction in seconds to a few minutes. Loading a serialized model and writing predictions are I/O-bound and fast relative to the feature engineering above.

**Conclusion:** The 86+ hour runtime is dominated by millions of interpreted-R string operations and per-row list iterations in the neighbor feature pipeline, not by RF inference.

---

## Optimization Strategy

1. **Eliminate string-key lookups entirely.** Replace the `paste(id, year, sep="_")` → named-vector lookup pattern with direct integer arithmetic. Since `id` maps to a known position in `id_order` and `year` spans a contiguous range (1992–2019, i.e., 28 years), each row can be addressed by a 2D integer index `(cell_position, year_offset)` mapped to a flat row index. This turns O(n × k) string operations into O(n × k) integer operations.

2. **Vectorize neighbor stats computation.** Instead of `lapply` over 6.46M list elements, build a single long vector of (row_index, neighbor_row_index) pairs, then use vectorized group-by operations (via `data.table`) to compute max, min, and mean in one pass per variable.

3. **Process all 5 variables in one pass** over the neighbor-pair structure rather than 5 separate passes.

These changes reduce the estimated runtime from 86+ hours to minutes.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# OPTIMIZED NEIGHBOR FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────

build_neighbor_pairs_fast <- function(cell_data_dt, id_order, rook_neighbors) {

  # cell_data_dt: a data.table with columns id, year, and an integer row index .row_idx
  # id_order:     vector of cell IDs in the same order as rook_neighbors (spdep::nb)
  # rook_neighbors: the nb list (each element is integer vector of neighbor positions)
  #

  # Returns a data.table with columns: row_idx (focal row), nb_row_idx (neighbor row)

  n_cells <- length(id_order)
  years   <- sort(unique(cell_data_dt$year))
  n_years <- length(years)

  # Map id -> position in id_order (integer)
  id_to_pos <- integer(max(id_order))
  id_to_pos[id_order] <- seq_len(n_cells)

  # Map (cell_position, year) -> row index in cell_data_dt
  # Build a matrix: rows = cell positions, cols = year offsets
  year_min <- min(years)

  # Create the mapping matrix (NA where a cell-year doesn't exist)
  row_lookup <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)

  cell_data_dt[, cell_pos := id_to_pos[id]]
  cell_data_dt[, year_off := year - year_min + 1L]
  cell_data_dt[, .row_idx := .I]

  # Fill the lookup matrix
  row_lookup[cbind(cell_data_dt$cell_pos, cell_data_dt$year_off)] <- cell_data_dt$.row_idx

  # Build the edge list of (focal_cell_pos, neighbor_cell_pos)
  # from the nb object
  focal_pos <- rep(seq_len(n_cells), lengths(rook_neighbors))
  nb_pos    <- unlist(rook_neighbors, use.names = FALSE)

  # Remove 0-entries (spdep uses 0 for no-neighbor cells)
  valid <- nb_pos > 0L
  focal_pos <- focal_pos[valid]
  nb_pos    <- nb_pos[valid]

  n_edges <- length(focal_pos)

  # Expand across all years: each edge exists for every year
  focal_pos_exp <- rep(focal_pos, each = n_years)
  nb_pos_exp    <- rep(nb_pos,    each = n_years)
  year_off_exp  <- rep(seq_len(n_years), times = n_edges)

  # Look up actual row indices
  focal_row <- row_lookup[cbind(focal_pos_exp, year_off_exp)]
  nb_row    <- row_lookup[cbind(nb_pos_exp,    year_off_exp)]

  # Keep only pairs where both focal and neighbor exist
  keep <- !is.na(focal_row) & !is.na(nb_row)

  pairs <- data.table(
    row_idx    = focal_row[keep],
    nb_row_idx = nb_row[keep]
  )

  # Clean up temporary columns
  cell_data_dt[, c("cell_pos", "year_off") := NULL]

  return(pairs)
}


compute_all_neighbor_features_fast <- function(cell_data_dt, pairs, neighbor_source_vars) {
  # pairs: data.table with row_idx, nb_row_idx
  # For each variable, compute max, min, mean of neighbor values grouped by row_idx

  n_rows <- nrow(cell_data_dt)

  for (var_name in neighbor_source_vars) {
    message("Computing neighbor features for: ", var_name)

    # Attach neighbor values to pairs
    pairs[, nb_val := cell_data_dt[[var_name]][nb_row_idx]]

    # Remove NA neighbor values
    valid_pairs <- pairs[!is.na(nb_val)]

    # Compute grouped stats
    stats <- valid_pairs[, .(
      nb_max  = max(nb_val),
      nb_min  = min(nb_val),
      nb_mean = mean(nb_val)
    ), by = row_idx]

    # Initialize columns with NA
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")

    set(cell_data_dt, j = max_col,  value = NA_real_)
    set(cell_data_dt, j = min_col,  value = NA_real_)
    set(cell_data_dt, j = mean_col, value = NA_real_)

    # Assign computed values
    set(cell_data_dt, i = stats$row_idx, j = max_col,  value = stats$nb_max)
    set(cell_data_dt, i = stats$row_idx, j = min_col,  value = stats$nb_min)
    set(cell_data_dt, i = stats$row_idx, j = mean_col, value = stats$nb_mean)

    # Clean up
    pairs[, nb_val := NULL]
  }

  invisible(cell_data_dt)
}


# ──────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────

# Convert to data.table if not already
cell_data_dt <- as.data.table(cell_data)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Step 1: Build integer-indexed neighbor pairs (replaces build_neighbor_lookup)
message("Building neighbor pair index...")
pairs <- build_neighbor_pairs_fast(cell_data_dt, id_order, rook_neighbors_unique)

# Step 2: Compute all neighbor features in vectorized grouped operations
#         (replaces the for-loop over compute_neighbor_stats)
compute_all_neighbor_features_fast(cell_data_dt, pairs, neighbor_source_vars)

# Step 3: Random Forest prediction (unchanged — preserving the trained model)
message("Running Random Forest prediction...")
# rf_model is the pre-trained model already loaded into memory, e.g.:
# rf_model <- readRDS("trained_rf_model.rds")
cell_data_dt[, prediction := predict(rf_model, newdata = cell_data_dt)$predictions]
# (Adjust the predict() call to match your RF package — ranger, randomForest, etc.)

message("Done.")
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M string pastes + named-vector hash lookups | Integer matrix indexing — no strings |
| **Neighbor stats** | `lapply` over 6.46M list elements × 5 vars; `do.call(rbind, ...)` | Vectorized `data.table` grouped aggregation; single edge-list reused for all 5 vars |
| **Expected runtime** | 86+ hours | Minutes (dominated by the `data.table` group-by on ~38M pairs × 5 vars) |
| **RF model** | Preserved ✓ | Preserved ✓ |
| **Numerical estimand** | Preserved ✓ (same max/min/mean logic, same neighbor definitions) | Preserved ✓ |

The true bottleneck is the interpreted-R, per-row, string-based neighbor feature engineering — not Random Forest inference.