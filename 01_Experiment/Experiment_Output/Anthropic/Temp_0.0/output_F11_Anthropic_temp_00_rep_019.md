 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` — as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, and for each row it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is hash-based but carries significant per-call overhead when done millions of times in an interpreted loop. With ~1.37 million neighbor relationships spread across 344,208 cells × 28 years, this function alone generates tens of millions of string operations.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46 million rows with an `lapply` loop that subsets, removes NAs, and computes `max/min/mean`. That's ~32.3 million R-level function invocations just for the stats.

3. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-trained model. Even with 6.46 million rows and 110 predictors, `predict.randomForest` (or `predict.ranger`) is implemented in C/C++ and is typically completed in seconds to a few minutes. Loading a serialized model (`readRDS`) is also fast. Writing predictions is trivial.

4. The **86+ hour runtime estimate** is entirely consistent with millions of iterations of interpreted R loops doing string manipulation and named-vector lookups — not with a single vectorized C-level predict call.

**Verdict:** The bottleneck is the row-level `lapply` loops in `build_neighbor_lookup()` and `compute_neighbor_stats()`. The optimization target is to vectorize these operations, eliminating per-row interpreted R overhead.

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()`** with a fully vectorized `data.table` join approach. Instead of building a per-row list of neighbor indices (6.46M-element list), we construct an **edge table** (a two-column data.table of `focal_row → neighbor_row` mappings) using vectorized operations. This avoids all per-row string pasting and named-vector lookups.

2. **Replace `compute_neighbor_stats()`** with a single grouped `data.table` aggregation over the edge table. For each variable, we join the neighbor values, then compute `max`, `min`, and `mean` grouped by the focal row — all in C-level `data.table` internals.

3. **Leave the Random Forest model and predict call untouched**, preserving the trained model and the original numerical estimand.

This reduces the complexity from ~6.46M × k interpreted R iterations to a handful of vectorized joins and group-by operations, bringing the expected runtime from 86+ hours down to **minutes**.

---

## Working R Code

```r
library(data.table)

# ============================================================
# OPTIMIZED PIPELINE — replaces build_neighbor_lookup() and
# compute_neighbor_stats() with vectorized data.table operations.
# The trained Random Forest model and original estimand are
# preserved exactly.
# ============================================================

#' Build a vectorized edge table mapping each focal cell-year row
#' to its neighbor cell-year rows.
#'
#' @param dt          A data.table with columns `id` and `year`
#'                    (and a `.row_idx` column will be added).
#' @param id_order    Integer vector of cell IDs in the order used
#'                    by the nb object.
#' @param nb_list     A precomputed spdep::nb object (list of
#'                    integer neighbor index vectors).
#' @return A data.table with columns `focal_row` and `neighbor_row`.
build_edge_table <- function(dt, id_order, nb_list) {
  n_cells <- length(id_order)

  # --- Step 1: Build cell-level directed edge list (vectorized) ---
  # Number of neighbors per cell

  n_neighbors <- lengths(nb_list)                        # integer vector, length n_cells
  focal_cell_idx    <- rep(seq_len(n_cells), n_neighbors)
  neighbor_cell_idx <- unlist(nb_list, use.names = FALSE)

  # Map cell indices back to actual cell IDs
  cell_edges <- data.table(
    focal_id    = id_order[focal_cell_idx],
    neighbor_id = id_order[neighbor_cell_idx]
  )

  # --- Step 2: Map cell-year combinations to row indices ---
  # Ensure dt has a row index
  dt[, .row_idx := .I]

  # Keyed lookup table: (id, year) -> row index
  id_year_key <- dt[, .(id, year, .row_idx)]
  setkey(id_year_key, id, year)

  # --- Step 3: Expand cell edges across all years (vectorized) ---
  years <- sort(unique(dt$year))

  # Cross join cell_edges × years
  # Use CJ-like expansion but more memory-friendly:
  edge_expanded <- cell_edges[, .(
    focal_id    = rep(focal_id,    length(years)),
    neighbor_id = rep(neighbor_id, length(years)),
    year        = rep(years, each = .N)
  )]

  # --- Step 4: Join to get focal_row and neighbor_row ---
  # Join focal side
  setkey(edge_expanded, focal_id, year)
  edge_expanded[id_year_key, focal_row := i..row_idx, on = .(focal_id = id, year = year)]

  # Join neighbor side
  setkey(edge_expanded, neighbor_id, year)
  edge_expanded[id_year_key, neighbor_row := i..row_idx, on = .(neighbor_id = id, year = year)]

  # Drop edges where either side is missing (boundary / missing year)
  edge_table <- edge_expanded[!is.na(focal_row) & !is.na(neighbor_row),
                              .(focal_row, neighbor_row)]

  setkey(edge_table, focal_row)
  return(edge_table)
}


#' Compute neighbor max, min, mean for a variable and attach
#' the three new columns to the data.table in place.
#'
#' @param dt         The main data.table (modified in place).
#' @param var_name   Character: name of the source variable.
#' @param edge_table A data.table with columns `focal_row`, `neighbor_row`.
compute_and_add_neighbor_features_fast <- function(dt, var_name, edge_table) {
  # Pull neighbor values via the edge table
  work <- edge_table[, .(focal_row, neighbor_row)]
  work[, val := dt[[var_name]][neighbor_row]]

  # Drop NAs in the variable (mirrors original logic)
  work <- work[!is.na(val)]

  # Grouped aggregation — all in C
  agg <- work[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), keyby = focal_row]

  # Prepare NA-filled result columns
  n <- nrow(dt)
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  set(dt, j = col_max,  value = rep(NA_real_, n))
  set(dt, j = col_min,  value = rep(NA_real_, n))
  set(dt, j = col_mean, value = rep(NA_real_, n))

  # Fill in computed values at the correct rows
  rows <- agg$focal_row
  set(dt, i = rows, j = col_max,  value = agg$nb_max)
  set(dt, i = rows, j = col_min,  value = agg$nb_min)
  set(dt, i = rows, j = col_mean, value = agg$nb_mean)

  invisible(dt)
}


# ============================================================
# MAIN EXECUTION
# ============================================================

# Convert to data.table if not already
cell_data <- as.data.table(cell_data)

# Build the edge table once (replaces build_neighbor_lookup)
message("Building vectorized edge table...")
edge_table <- build_edge_table(cell_data, id_order, rook_neighbors_unique)
message(sprintf("Edge table: %s rows", format(nrow(edge_table), big.mark = ",")))

# Compute neighbor features for all 5 variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor stats for: %s", var_name))
  compute_and_add_neighbor_features_fast(cell_data, var_name, edge_table)
}

message("Neighbor feature engineering complete.")

# ============================================================
# RANDOM FOREST INFERENCE — unchanged, preserving trained model
# and original numerical estimand exactly.
# ============================================================
# rf_model <- readRDS("path/to/trained_rf_model.rds")
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
# fwrite(cell_data, "path/to/output.csv")
```

---

## Why This Works and Complexity Summary

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `lapply` iterations, each with `paste()` + named lookup | One vectorized cross-join + two keyed `data.table` joins |
| `compute_neighbor_stats` (×5 vars) | 5 × 6.46M `lapply` iterations with per-row subsetting | 5 × one grouped `data.table` aggregation (C-level) |
| String operations | ~6.46M `paste()` calls + ~6.46M `as.character()` calls | Zero — integer joins only |
| Expected runtime | 86+ hours | **Minutes** (typically 5–15 min on 16 GB laptop) |
| RF inference | Fast (single `predict()` call) | **Unchanged** — not the bottleneck |
| Numerical results | Original estimand | **Preserved exactly** — same max/min/mean logic, same NA handling |

The edge table expansion (`n_edges × n_years`) will produce roughly 1.37M × 28 ≈ 38.5M rows, which at two integer columns is ~300 MB — well within the 16 GB RAM budget. The `data.table` grouped aggregation over 38.5M rows is a routine operation completed in seconds.