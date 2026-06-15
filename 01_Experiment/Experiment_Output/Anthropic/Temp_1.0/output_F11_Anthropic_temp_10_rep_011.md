 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence clearly points to a different bottleneck: **the spatial neighbor feature engineering pipeline**, specifically `build_neighbor_lookup()` and `compute_neighbor_stats()`.

Here's why:

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, and for each row it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), named vector lookups (`idx_lookup[neighbor_keys]`), and NA filtering. Named character vector lookup in R is O(n) per lookup in the worst case, and with ~6.46M rows each touching multiple neighbors, this is catastrophically slow.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over ~6.46M elements with `lapply`, performing subsetting, NA removal, and computing `max`, `min`, `mean`. The `do.call(rbind, result)` on a 6.46M-element list of 3-vectors is also expensive.

3. **The outer loop** compounds this: 5 variables × 6.46M rows × multiple neighbor lookups per row = tens of millions of character-based hash lookups and small-vector statistics.

4. By contrast, **Random Forest `predict()`** on a pre-trained model with ~6.46M rows and ~110 predictors is a single vectorized call that typically completes in minutes (even on a laptop), not hours. Loading a serialized model and writing predictions are trivially fast operations.

**The 86+ hour runtime is dominated by the row-by-row `lapply` loops with string-based key lookups for neighbor feature construction**, not by model inference.

---

## Optimization Strategy

1. **Replace string-based key lookups with integer-indexed direct lookups.** Build a matrix mapping `(cell_index, year_index)` → row number using integer arithmetic, eliminating all `paste()` and named-vector lookups.

2. **Vectorize `compute_neighbor_stats()`** using `data.table` grouped operations or, even better, a pre-flattened edge list with vectorized aggregation — replacing the per-row `lapply` with a single grouped `data.table` summarization.

3. **Build the neighbor edge list once** as a two-column integer matrix (row_index → neighbor_row_index) and reuse it for all 5 variables.

These changes reduce the complexity from millions of interpreted R-level loop iterations with string operations to a handful of vectorized, memory-contiguous operations.

---

## Working R Code

```r
library(data.table)

# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE ENGINEERING
# =============================================================================

#' Build a fast integer-indexed lookup and a flat edge list.
#'
#' Instead of pasting strings and doing named-vector lookups for every row,
#' we create a direct (cell_index, year_index) -> row_number matrix and
#' then expand the neighbor graph into a flat edge list of row pairs.
#'
#' @param dt           data.table with columns `id` and `year`
#' @param id_order     integer vector of cell IDs in the order matching
#'                     the rook_neighbors_unique nb object
#' @param nb_list      spdep::nb object (list of integer neighbor index vectors)
#' @return A data.table with columns `from_row` and `to_row`
build_neighbor_edge_list <- function(dt, id_order, nb_list) {

  # --- Step 1: Map cell IDs to sequential integer indices ---
  # id_order[k] is the cell ID for the k-th element of the nb object.
  # We need the reverse: cell_id -> k
  n_cells <- length(id_order)
  id_to_cell_idx <- integer(max(id_order))
  id_to_cell_idx[id_order] <- seq_len(n_cells)
  # If IDs are very large/sparse, use a hash instead:
  # id_to_cell_idx <- new.env(hash = TRUE, size = n_cells)
  # for (k in seq_len(n_cells)) {
  #   id_to_cell_idx[[as.character(id_order[k])]] <- k
  # }

  # --- Step 2: Map years to sequential integer indices ---
  years_sorted <- sort(unique(dt$year))
  n_years <- length(years_sorted)
  year_to_year_idx <- integer(max(years_sorted) - min(years_sorted) + 1)
  year_offset <- min(years_sorted) - 1L
  year_to_year_idx[years_sorted - year_offset] <- seq_len(n_years)

  # --- Step 3: Build (cell_idx, year_idx) -> row_number matrix ---
  # This replaces the expensive paste/named-lookup approach.
  row_lookup <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)

  cell_indices <- id_to_cell_idx[dt$id]
  year_indices <- year_to_year_idx[dt$year - year_offset]
  row_numbers  <- seq_len(nrow(dt))

  row_lookup[cbind(cell_indices, year_indices)] <- row_numbers

  # --- Step 4: Expand the nb list into a flat edge list ---
  # For each cell, get its neighbors; then for each year, create
  # (from_row, to_row) pairs.
  #
  # First, build a flat cell-level edge list from the nb object.
  from_cell <- rep(seq_len(n_cells), times = lengths(nb_list))
  to_cell   <- unlist(nb_list, use.names = FALSE)

  # Remove 0-entries (spdep uses 0 to denote "no neighbors")
  valid <- to_cell > 0L
  from_cell <- from_cell[valid]
  to_cell   <- to_cell[valid]

  n_edges_cell <- length(from_cell)

  # Now cross with all years: each cell-level edge becomes n_years row-level edges.
  from_cell_rep <- rep(from_cell, each = n_years)
  to_cell_rep   <- rep(to_cell,   each = n_years)
  year_idx_rep  <- rep(seq_len(n_years), times = n_edges_cell)

  from_row <- row_lookup[cbind(from_cell_rep, year_idx_rep)]
  to_row   <- row_lookup[cbind(to_cell_rep,   year_idx_rep)]

  # Drop pairs where either side is missing from the data
  keep <- !is.na(from_row) & !is.na(to_row)

  data.table(from_row = from_row[keep], to_row = to_row[keep])
}


#' Compute neighbor max, min, mean for a variable using vectorized data.table ops.
#'
#' @param dt        data.table with the source variable
#' @param edge_dt   data.table with `from_row` and `to_row` columns
#' @param var_name  character: name of the variable to aggregate
#' @return A data.table with columns: nb_max_{var}, nb_min_{var}, nb_mean_{var}
#'         with nrow(dt) rows (NA for rows with no valid neighbors)
compute_neighbor_stats_fast <- function(dt, edge_dt, var_name) {

  n <- nrow(dt)

  # Pull the values for the neighbor (to_row) side
  vals <- dt[[var_name]][edge_dt$to_row]

  # Build a small working table
  work <- data.table(
    from_row = edge_dt$from_row,
    val      = vals
  )

  # Drop NAs in the variable itself
  work <- work[!is.na(val)]

  # Grouped aggregation — single pass, fully vectorized
  agg <- work[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), keyby = from_row]

  # Map back to full row set
  col_max  <- rep(NA_real_, n)
  col_min  <- rep(NA_real_, n)
  col_mean <- rep(NA_real_, n)

  col_max[agg$from_row]  <- agg$nb_max
  col_min[agg$from_row]  <- agg$nb_min
  col_mean[agg$from_row] <- agg$nb_mean

  max_name  <- paste0("nb_max_",  var_name)
  min_name  <- paste0("nb_min_",  var_name)
  mean_name <- paste0("nb_mean_", var_name)

  out <- data.table(col_max, col_min, col_mean)
  setnames(out, c(max_name, min_name, mean_name))
  out
}


# =============================================================================
# MAIN PIPELINE (drop-in replacement for the original outer loop)
# =============================================================================

# Convert to data.table if not already (non-destructive; does not alter contents)
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# --- Build the edge list ONCE (replaces build_neighbor_lookup) ---
message("Building neighbor edge list...")
edge_dt <- build_neighbor_edge_list(cell_data, id_order, rook_neighbors_unique)
message(sprintf("  Edge list: %s row-level directed edges", format(nrow(edge_dt), big.mark = ",")))

# --- Compute neighbor features for each variable ---
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor stats for: %s", var_name))
  stats_dt <- compute_neighbor_stats_fast(cell_data, edge_dt, var_name)
  # Bind new columns to cell_data in place
  cell_data[, names(stats_dt) := stats_dt]
}

message("Neighbor feature engineering complete.")

# =============================================================================
# RANDOM FOREST INFERENCE (unchanged — this was never the bottleneck)
# =============================================================================

# Load the pre-trained model (preserved as-is)
# rf_model <- readRDS("path/to/trained_rf_model.rds")

# Predict (single vectorized call; typically minutes, not hours)
# cell_data[, prediction := predict(rf_model, newdata = cell_data)]

# Write output
# fwrite(cell_data, "path/to/output.csv")
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup mechanism** | `paste()` + named character vector (~6.46M string ops) | Integer matrix indexing `row_lookup[cbind(cell, year)]` |
| **Neighbor iteration** | `lapply` over 6.46M rows, repeated 5× | Single flat edge list built once, reused 5× |
| **Aggregation** | Per-row `max/min/mean` in interpreted R loop | `data.table` grouped `[, .(max, min, mean), by=from_row]` — vectorized C-level |
| **`do.call(rbind, ...)`** | Binds 6.46M 3-element vectors (extremely slow) | Eliminated; direct indexed assignment into pre-allocated vectors |
| **Estimated runtime** | 86+ hours | **Minutes** (edge list build ~1–3 min; each variable's stats ~30–60 sec) |
| **RF model** | Preserved, untouched | Preserved, untouched |
| **Numerical results** | Same `max`, `min`, `mean` per row per variable | Identical numerical estimand preserved |

The true bottleneck is the **O(n × k) interpreted-loop neighbor feature construction with string-key lookups**, not Random Forest inference. The fix is vectorized integer indexing and `data.table` grouped aggregation.