 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` — as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to construct neighbor keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. Named vector lookups in R are hash-based but carry significant per-call overhead at this scale. With ~6.46M rows and an average of ~4 rook neighbors per cell, this generates roughly **25.8 million string-paste and hash-lookup operations** inside a sequential `lapply`.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46M rows with `lapply`, subsetting a numeric vector, removing NAs, and computing `max`, `min`, and `mean`. That's another **~32.3 million** R-level function-call iterations total.

3. **Combined**: The neighbor pipeline performs on the order of **~38+ million R-interpreted loop iterations** with per-element string operations and small-vector statistics. This is the classic R anti-pattern that produces the estimated 86+ hour runtime.

4. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-trained model over a matrix of ~6.46M × 110 features. The `ranger` or `randomForest` predict methods are implemented in C/C++ and are highly optimized. Even on a laptop, prediction on this scale typically completes in **seconds to a few minutes** — orders of magnitude faster than the neighbor feature loop.

**Verdict**: The bottleneck is the row-level R `lapply` loops performing string construction, named-vector lookups, and per-row summary statistics across millions of rows. The Random Forest inference is negligible by comparison.

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()`** with a vectorized `data.table` merge/join approach. Instead of looping row-by-row to paste keys and look up indices, we:
   - Expand the `nb` object into an edge list (cell_id → neighbor_id) once.
   - Join this edge list with the data on (neighbor_id, year) to get row indices of neighbors directly.
   - This replaces millions of `paste` + named-vector lookups with a single keyed `data.table` join.

2. **Replace `compute_neighbor_stats()`** with a **grouped `data.table` aggregation**. Once we have an edge table mapping each row to its neighbor rows, we can gather neighbor values and compute `max`, `min`, `mean` in one vectorized grouped operation per variable — no `lapply` over 6.46M elements.

3. **Process all 5 variables** in the same join pass or with minimal repeated joins.

This reduces the runtime from ~86+ hours to an estimated **minutes** (typically 5–20 minutes depending on RAM pressure on a 16 GB laptop).

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 0: Convert cell_data to data.table and assign a row index
# ──────────────────────────────────────────────────────────────────────
setDT(cell_data)
cell_data[, row_idx := .I]

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build the edge list from the nb object (one-time cost)
#
# rook_neighbors_unique is an nb object: a list of length
# length(id_order), where element i contains integer indices into
# id_order of the neighbors of id_order[i].
# ──────────────────────────────────────────────────────────────────────
build_edge_list <- function(id_order, nb_obj) {
  # Pre-allocate: count total edges
  n_edges <- sum(vapply(nb_obj, function(x) {
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1)))

  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  pos <- 1L

  for (i in seq_along(nb_obj)) {
    nbrs <- nb_obj[[i]]
    # spdep::nb encodes "no neighbors" as a single 0
    if (length(nbrs) == 1L && nbrs[1] == 0L) next
    n <- length(nbrs)
    from_id[pos:(pos + n - 1L)] <- id_order[i]
    to_id[pos:(pos + n - 1L)]   <- id_order[nbrs]
    pos <- pos + n
  }

  data.table(focal_id = from_id, neighbor_id = to_id)
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)

# ──────────────────────────────────────────────────────────────────────
# STEP 2: Map each (focal_id, year) → its row_idx, and each
#         (neighbor_id, year) → the neighbor's row_idx, via keyed joins
# ──────────────────────────────────────────────────────────────────────

# Lookup table: id + year → row_idx
id_year_idx <- cell_data[, .(id, year, row_idx)]
setkey(id_year_idx, id, year)

# Get all unique years
all_years <- sort(unique(cell_data$year))

# Cross-join edges × years, then join to get focal and neighbor row indices
# To manage memory on a 16 GB laptop, we process year-by-year.

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-allocate result columns in cell_data
for (var_name in neighbor_source_vars) {
  cell_data[, paste0("max_", var_name, "_neighbor") := NA_real_]
  cell_data[, paste0("min_", var_name, "_neighbor") := NA_real_]
  cell_data[, paste0("mean_", var_name, "_neighbor") := NA_real_]
}

# Process year by year to limit peak memory (~230K edges × 1 year at a time)
setkey(edge_dt, focal_id)

for (yr in all_years) {

  # Subset rows for this year
  yr_data <- cell_data[year == yr, c("id", "row_idx", ..neighbor_source_vars)]
  setkey(yr_data, id)

  # Join edges to get focal row_idx
  # edge_dt: focal_id, neighbor_id
  # Join focal side
  edges_yr <- merge(edge_dt, yr_data[, .(id, focal_row_idx = row_idx)],
                    by.x = "focal_id", by.y = "id", allow.cartesian = TRUE)

  # Join neighbor side to get neighbor variable values
  neighbor_vals <- yr_data[, c("id", neighbor_source_vars), with = FALSE]
  setnames(neighbor_vals, "id", "neighbor_id")
  setkey(neighbor_vals, neighbor_id)
  setkey(edges_yr, neighbor_id)

  edges_yr <- merge(edges_yr, neighbor_vals, by = "neighbor_id",
                    allow.cartesian = FALSE)

  # Now edges_yr has columns:
  #   neighbor_id, focal_id, focal_row_idx, ntl, ec, pop_density, def, usd_est_n2
  # where the variable columns are the NEIGHBOR's values.

  # Group by focal_row_idx and compute stats for each variable
  agg_exprs <- list()
  for (var_name in neighbor_source_vars) {
    max_nm  <- paste0("max_", var_name, "_neighbor")
    min_nm  <- paste0("min_", var_name, "_neighbor")
    mean_nm <- paste0("mean_", var_name, "_neighbor")
    agg_exprs[[max_nm]]  <- call("max",  as.name(var_name), na.rm = TRUE)
    agg_exprs[[min_nm]]  <- call("min",  as.name(var_name), na.rm = TRUE)
    agg_exprs[[mean_nm]] <- call("mean", as.name(var_name), na.rm = TRUE)
  }

  # Build the aggregation call
  agg_call <- as.call(c(as.name("list"),
                         lapply(names(agg_exprs), function(nm) {
                           call("<-", as.name(nm), agg_exprs[[nm]])
                         })))

  # Simpler approach: use .SDcols
  stats_yr <- edges_yr[, {
    res <- list()
    for (vn in neighbor_source_vars) {
      v <- get(vn)
      v <- v[!is.na(v)]
      if (length(v) == 0L) {
        res[[paste0("max_", vn, "_neighbor")]]  <- NA_real_
        res[[paste0("min_", vn, "_neighbor")]]  <- NA_real_
        res[[paste0("mean_", vn, "_neighbor")]] <- NA_real_
      } else {
        res[[paste0("max_", vn, "_neighbor")]]  <- max(v)
        res[[paste0("min_", vn, "_neighbor")]]  <- min(v)
        res[[paste0("mean_", vn, "_neighbor")]] <- mean(v)
      }
    }
    res
  }, by = focal_row_idx]

  # Write results back into cell_data using row indices
  stat_cols <- setdiff(names(stats_yr), "focal_row_idx")
  cell_data[stats_yr$focal_row_idx, (stat_cols) := stats_yr[, ..stat_cols]]
}

# ──────────────────────────────────────────────────────────────────────
# STEP 3: Clean up helper column
# ──────────────────────────────────────────────────────────────────────
cell_data[, row_idx := NULL]

# ──────────────────────────────────────────────────────────────────────
# STEP 4: Random Forest prediction (unchanged — not the bottleneck)
# ──────────────────────────────────────────────────────────────────────
# The pre-trained RF model is loaded and used as-is.
# Example (preserving the original estimand):
#   predictions <- predict(rf_model, data = cell_data)
#   cell_data[, predicted := predictions$predictions]  # ranger
#   # or
#   cell_data[, predicted := predict(rf_model, newdata = cell_data)]  # randomForest
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **`build_neighbor_lookup`** | ~6.46M `lapply` iterations with `paste` + named-vector hash lookups | One-time edge-list construction + keyed `data.table` joins |
| **`compute_neighbor_stats`** | 5 × ~6.46M `lapply` iterations computing per-row `max/min/mean` | Grouped `data.table` aggregation per year (~344K groups × 28 years) |
| **String operations** | ~25.8M `paste()` calls | Zero — joins on integer keys |
| **Estimated runtime** | 86+ hours | ~5–20 minutes |
| **RF model** | Preserved (not retrained) | Preserved (not retrained) |
| **Numerical results** | Original `max/min/mean` neighbor stats | Identical `max/min/mean` neighbor stats |

The true bottleneck is the **row-level R-interpreted neighbor feature engineering**, not Random Forest inference. The fix is to vectorize the spatial lookups and aggregations using `data.table` keyed joins and grouped operations.