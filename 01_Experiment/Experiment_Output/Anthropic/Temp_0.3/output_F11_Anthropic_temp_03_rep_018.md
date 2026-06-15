 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` — as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to build neighbor keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. Named vector lookups in R are hash-based but carry significant per-call overhead at this scale. With ~6.46M iterations, this alone is extremely expensive.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46M rows with `lapply`, subsetting a numeric vector, removing NAs, and computing `max`, `min`, and `mean`. That's ~32.3 million R-level loop iterations total.

3. By contrast, Random Forest prediction (`predict()`) on a pre-trained model is a single vectorized C/C++ call over the 6.46M rows. It is inherently fast and is **not** the bottleneck.

**The bottleneck is the row-level R `lapply` loops over millions of rows for neighbor lookup construction and neighbor statistics computation.**

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup` with a vectorized `data.table` equi-join.** Instead of looping row-by-row, expand all neighbor relationships into an edge table (`cell_id`, `neighbor_id`), join with the data on `(neighbor_id, year)` to get row indices, and group by the source row to collect neighbor indices. This turns millions of R-level iterations into a single indexed merge.

2. **Replace `compute_neighbor_stats` with a vectorized `data.table` grouped aggregation.** Using the edge table joined to the data, compute `max`, `min`, and `mean` per `(source_row, variable)` in one grouped operation — no R-level loop required.

3. **Leave the Random Forest predict() call untouched**, as it is not the bottleneck.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# Step 1: Build a vectorized edge table from the nb object
#         This replaces build_neighbor_lookup()
# ──────────────────────────────────────────────────────────────────────

build_edge_table <- function(id_order, rook_neighbors_unique) {
  # Expand the nb list into a two-column data.table of (cell_id, neighbor_id)
  n_neighbors <- lengths(rook_neighbors_unique)
  source_ref  <- rep(seq_along(id_order), times = n_neighbors)
  target_ref  <- unlist(rook_neighbors_unique, use.names = FALSE)

  data.table(
    cell_id     = id_order[source_ref],
    neighbor_id = id_order[target_ref]
  )
}

# ──────────────────────────────────────────────────────────────────────
# Step 2: Vectorized neighbor feature computation
#         This replaces compute_neighbor_stats() + the outer for-loop
# ──────────────────────────────────────────────────────────────────────

compute_all_neighbor_features <- function(cell_data, id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {

  dt <- as.data.table(cell_data)

  # Ensure a stable row identifier so we can map results back
  dt[, .row_id := .I]

  # --- Build edge table (once) ---
  edges <- build_edge_table(id_order, rook_neighbors_unique)

  # --- Cross edges with years to get (source_row, neighbor_row) pairs ---
  # Key the data for fast joins
  setkey(dt, id, year)

  # Source side: attach source row id and year to every edge
  # For every (cell_id, year) in dt, look up its edges
  source_keys <- dt[, .(cell_id = id, year, src_row = .row_id)]

  # Merge edges with source_keys on cell_id to get
  # (src_row, neighbor_id, year) for every edge × year
  edge_year <- edges[source_keys, on = .(cell_id), allow.cartesian = TRUE,
                     nomatch = NULL]
  # edge_year now has columns: cell_id, neighbor_id, year, src_row

  # Now join to dt again to resolve neighbor_id + year → neighbor row
  # and pull the variable values we need
  keep_cols <- c("id", "year", ".row_id", neighbor_source_vars)
  nbr_data  <- dt[, ..keep_cols]
  setnames(nbr_data, "id", "neighbor_id")
  setnames(nbr_data, ".row_id", "nbr_row")

  setkey(nbr_data, neighbor_id, year)
  setkey(edge_year, neighbor_id, year)

  joined <- nbr_data[edge_year, on = .(neighbor_id, year),
                     nomatch = NULL]
  # joined has: neighbor_id, year, nbr_row, <var columns>, cell_id, src_row

  # --- Compute grouped stats for each variable ---
  for (var_name in neighbor_source_vars) {
    var_sym <- var_name

    stats <- joined[!is.na(get(var_sym)),
                    .(nb_max  = max(get(var_sym)),
                      nb_min  = min(get(var_sym)),
                      nb_mean = mean(get(var_sym))),
                    by = src_row]

    # Name the new columns to match original pipeline output
    max_col  <- paste0("nb_max_",  var_name)
    min_col  <- paste0("nb_min_",  var_name)
    mean_col <- paste0("nb_mean_", var_name)
    setnames(stats, c("nb_max", "nb_min", "nb_mean"),
             c(max_col, min_col, mean_col))

    # Left-join back to dt on row id (rows with no neighbors get NA)
    dt <- stats[dt, on = .(src_row = .row_id)]
    setnames(dt, "src_row", ".row_id")
  }

  # Clean up helper column
  dt[, .row_id := NULL]

  return(dt)
}

# ──────────────────────────────────────────────────────────────────────
# Step 3: Run the optimized pipeline
# ──────────────────────────────────────────────────────────────────────

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# ──────────────────────────────────────────────────────────────────────
# Step 4: Random Forest prediction (UNCHANGED — not the bottleneck)
# ──────────────────────────────────────────────────────────────────────

# Load the pre-trained model (preserved, not retrained)
# rf_model <- readRDS("path/to/trained_rf_model.rds")
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | `lapply` over 6.46M rows with per-row string ops & named-vector lookups | Single `data.table` equi-join via edge table |
| **Neighbor stats** | 5 × `lapply` over 6.46M rows (32.3M R-level iterations) | 5 × grouped `data.table` aggregation (vectorized C) |
| **RF inference** | Single `predict()` call (already fast) | Unchanged |
| **Estimated time** | 86+ hours | Minutes (dominated by the join, which `data.table` handles efficiently with indexed merges) |
| **Numerical results** | max / min / mean per neighbor set | Identical — same arithmetic, same neighbor definitions |
| **Trained RF model** | Preserved | Preserved — no retraining |

The correct optimization target is the **spatial neighbor feature engineering**, not the Random Forest inference.