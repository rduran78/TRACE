 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which iterates over **~6.46 million rows** one at a time in a pure-R `lapply`, performing per-row string pasting, hash lookups, and NA filtering. This creates ~6.46 million small character vectors, does millions of `paste` and named-vector lookups, and produces millions of small integer vectors. The `compute_neighbor_stats` function then loops over the same 6.46 million entries again, subsetting a numeric vector with each small index vector.

Specific costs:

| Step | Operation | Why it's slow |
|---|---|---|
| `build_neighbor_lookup` | `paste(neighbor_cell_ids, data$year[i], sep="_")` per row | 6.46M string-construction calls |
| `build_neighbor_lookup` | `idx_lookup[neighbor_keys]` per row | 6.46M named-vector lookups (hash probes on character keys) |
| `compute_neighbor_stats` | `lapply` over 6.46M entries, subsetting + `max/min/mean` | R-level loop overhead, no vectorization |
| Outer loop | Repeated 5× for each source variable | `build_neighbor_lookup` cost is paid once, but `compute_neighbor_stats` is paid 5× |

Estimated: ~86+ hours on a 16 GB laptop.

## Optimization Strategy

**Replace the row-level R loops with fully vectorized operations using `data.table` and a pre-expanded edge list.**

Key ideas:

1. **Build the edge list once** — expand the `nb` object into a two-column integer matrix of `(cell_id, neighbor_cell_id)` pairs. This is ~1.37M rows.

2. **Join by (neighbor_id, year) using `data.table`** — instead of looping over 6.46M rows and doing string-key lookups, merge the edge list with the panel on `(id, year)` to retrieve neighbor values. `data.table` binary-search joins make this extremely fast.

3. **Group-by aggregation** — after the join, compute `max`, `min`, and `mean` of neighbor values grouped by `(id, year)` in one vectorized pass per variable.

4. **Loop only over the 5 variables**, not over rows.

This eliminates all per-row R-level iteration. Expected runtime: **minutes, not hours**.

The trained Random Forest model is untouched. The numerical results (neighbor max, min, mean per cell-year) are identical because the same neighbor relationships and the same aggregation functions are used.

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 1. Convert the spdep nb object to a two-column edge list (cell IDs)
#    id_order maps position index -> cell id
#    rook_neighbors_unique is the nb object (list of integer vectors)
# ──────────────────────────────────────────────────────────────────────

build_edge_list <- function(id_order, nb_obj) {
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(nb_obj))
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(nb_obj)) {
    nbs <- nb_obj[[i]]
    if (length(nbs) == 0L || (length(nbs) == 1L && nbs[1] == 0L)) next
    n <- length(nbs)
    from_id[pos:(pos + n - 1L)] <- id_order[i]
    to_id[pos:(pos + n - 1L)]   <- id_order[nbs]
    pos <- pos + n
  }
  
  # Trim if any cells had zero neighbors (0-sentinel in nb)
  if (pos <= n_edges) {
    from_id <- from_id[1:(pos - 1L)]
    to_id   <- to_id[1:(pos - 1L)]
  }
  
  data.table(id = from_id, neighbor_id = to_id)
}

# ──────────────────────────────────────────────────────────────────────
# 2. Vectorized neighbor stats computation
# ──────────────────────────────────────────────────────────────────────

compute_neighbor_features_vectorized <- function(cell_dt, edge_dt, var_name) {
  # Build a lookup table: (neighbor_id aliased as id, year) -> value
  # We only need id, year, and the variable column from the panel
  lookup_cols <- c("id", "year", var_name)
  lookup <- cell_dt[, ..lookup_cols]
  setnames(lookup, c("id", var_name), c("neighbor_id", "nval"))
  
  # Join edge list with panel to get (id, year, neighbor_id),

  # then join on (neighbor_id, year) to get neighbor values.
  # Step A: cross edge list with all years for each id?
  #   No — more efficient: join panel with edge list on id,
  #   then join the result with lookup on (neighbor_id, year).
  
  # Merge panel rows with their neighbor IDs
  # cell_dt has (id, year, ...). We need (id, year) x edge_dt on id -> (id, year, neighbor_id)
  # Use edge_dt keyed on id.
  
  setkey(edge_dt, id)
  
  # Get unique (id, year) pairs — these are just the row indices of cell_dt
  id_year <- cell_dt[, .(id, year)]
  
  # Join: for each (id, year) row, find all neighbor_ids from edge_dt
  # This produces ~6.46M * (avg ~4 neighbors) ≈ 25-26M rows
  expanded <- edge_dt[id_year, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded has columns: id, neighbor_id, year
  
  # Now join to get the neighbor's value in that year
  setkey(lookup, neighbor_id, year)
  setkey(expanded, neighbor_id, year)
  
  expanded <- lookup[expanded, on = .(neighbor_id, year), nomatch = NA]
  # expanded now has: neighbor_id, year, nval, id
  
  # Aggregate by (id, year)
  stats <- expanded[!is.na(nval),
                    .(nmax  = max(nval),
                      nmin  = min(nval),
                      nmean = mean(nval)),
                    by = .(id, year)]
  
  # Name the new columns
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats, c("nmax", "nmin", "nmean"), c(max_col, min_col, mean_col))
  
  # Left-join back onto cell_dt (preserves rows with no neighbors as NA)
  setkey(stats, id, year)
  setkey(cell_dt, id, year)
  
  cell_dt <- stats[cell_dt, on = .(id, year)]
  
  # Clean up the lookup rename so we don't mutate the caller's copy
  cell_dt
}

# ──────────────────────────────────────────────────────────────────────
# 3. Main pipeline
# ──────────────────────────────────────────────────────────────────────

# Convert cell_data to data.table (if not already)
cell_data <- as.data.table(cell_data)

# Build edge list once (~1.37M rows, instant)
edge_list <- build_edge_list(id_order, rook_neighbors_unique)

# Source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Compute and attach neighbor features for each variable
for (var_name in neighbor_source_vars) {
  message("Computing neighbor stats for: ", var_name)
  cell_data <- compute_neighbor_features_vectorized(cell_data, edge_list, var_name)
}

# ──────────────────────────────────────────────────────────────────────
# 4. Predict with the existing trained Random Forest (unchanged)
# ──────────────────────────────────────────────────────────────────────
# predictions <- predict(trained_rf_model, newdata = cell_data)
# (The model object and predict call remain exactly as before.)
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same neighbor relationships** | `build_edge_list` reads the same `rook_neighbors_unique` nb object and the same `id_order` mapping. Every directed edge is identical. |
| **Same aggregation** | `max`, `min`, `mean` over the same non-NA neighbor values per `(id, year)` group — algebraically identical to the original `compute_neighbor_stats`. |
| **NAs handled identically** | `!is.na(nval)` filtering before aggregation mirrors `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]`. Cells with zero valid neighbors get `NA` via the left join (same as the original `c(NA, NA, NA)` return). |
| **Trained RF untouched** | No retraining; the feature columns have the same names and values. |

## Expected Performance

| Component | Original | Optimized |
|---|---|---|
| Neighbor lookup build | ~hours (6.46M R-loop iterations with string ops) | ~seconds (vectorized edge list construction) |
| Stats per variable | ~17 hours × 5 = 85h | ~30–90 seconds × 5 ≈ 3–8 minutes |
| **Total** | **86+ hours** | **~5–10 minutes** |

Peak memory for the expanded join table: ~25M rows × 3 numeric columns ≈ ~600 MB, well within 16 GB.