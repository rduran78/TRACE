 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Converts an ID to a character and looks it up in a named vector — **O(1) amortized but with constant overhead from `as.character()` and named-vector hashing, repeated 6.46M times**.
2. Constructs `paste(..., sep="_")` key strings for every neighbor of every row — millions of string allocations.
3. Indexes into `idx_lookup` (a named character vector) — named vector lookup in R is hash-based but still has per-call overhead, and this is done for every neighbor of every row.

The total number of string constructions and hash lookups is approximately **6.46M rows × ~4 avg neighbors ≈ 25.8M paste + lookup operations**, all inside a sequential `lapply` with R-level overhead.

**`compute_neighbor_stats`** then loops over 6.46M entries again, computing `max/min/mean` in pure R — slow but less catastrophic since the inner operations are cheap. However, it is called 5 times (once per variable), totaling ~32.3M R-level function calls.

**Root cause summary:**
- Millions of R-level string allocations and hash lookups in a sequential loop.
- No vectorization or use of `data.table` merge/join semantics.
- `compute_neighbor_stats` uses `lapply` + `do.call(rbind, ...)` over millions of small vectors.

## Optimization Strategy

1. **Replace `build_neighbor_lookup` entirely** with a vectorized `data.table` equi-join approach: expand the neighbor graph into an edge list, join on `(neighbor_id, year)` to get row indices, then group by source row.
2. **Replace `compute_neighbor_stats`** with a single grouped `data.table` aggregation per variable — `max`, `min`, `mean` computed in C-level `data.table` internals, not R-level loops.
3. **Avoid materializing the full neighbor_lookup list at all** — go directly from edge list to aggregated statistics.

This reduces the problem to a merge + grouped aggregation, which `data.table` handles in seconds-to-minutes on data of this size.

## Optimized Working R Code

```r
library(data.table)

#' Build a directed edge list from the spdep nb object.
#' Returns a data.table with columns: source_id, neighbor_id
build_edge_list <- function(id_order, neighbors) {
  # neighbors is an nb object: list of integer index vectors
  n <- length(neighbors)
  # Pre-allocate by computing total edges
  lens <- vapply(neighbors, length, integer(1))
  total <- sum(lens)
  
  source_id   <- integer(total)
  neighbor_id <- integer(total)
  
  pos <- 1L
  for (i in seq_len(n)) {
    nb_i <- neighbors[[i]]
    k <- length(nb_i)
    if (k > 0L) {
      idx <- pos:(pos + k - 1L)
      source_id[idx]   <- id_order[i]
      neighbor_id[idx] <- id_order[nb_i]
      pos <- pos + k
    }
  }
  
  data.table(source_id = source_id, neighbor_id = neighbor_id)
}

#' Compute neighbor summary statistics for one variable using data.table joins.
#' Returns a data.table with columns: id, year, <var>_max, <var>_min, <var>_mean
compute_neighbor_stats_fast <- function(dt, edge_dt, var_name) {
  # dt must be a data.table with columns: id, year, row_idx, and <var_name>
  # edge_dt has columns: source_id, neighbor_id
  
  # Step 1: Cross edge list with years via join on neighbor side
  # We need: for each (source_id, year), find all (neighbor_id, year) rows and
  # aggregate var_name.
  
  # Create a keyed version for joining
  neighbor_vals <- dt[, .(neighbor_id = id, year, nval = get(var_name))]
  setkey(neighbor_vals, neighbor_id, year)
  
  # Expand edges × years: join edge_dt to dt to get (source_id, year) pairs,
  # then join to neighbor values.
  # More efficient: join edges to neighbor_vals, then join back source identity.
  
  # edges_with_vals: for each (source_id, neighbor_id), for each year,
  # get the neighbor's value
  edges_expanded <- edge_dt[neighbor_vals, on = .(neighbor_id), allow.cartesian = TRUE, nomatch = 0L]
  # edges_expanded now has: source_id, neighbor_id, year, nval
  
  # Step 2: Aggregate by (source_id, year)
  max_name  <- paste0(var_name, "_max")
  min_name  <- paste0(var_name, "_min")
  mean_name <- paste0(var_name, "_mean")
  
  agg <- edges_expanded[
    !is.na(nval),
    .(V_max = max(nval), V_min = min(nval), V_mean = mean(nval)),
    by = .(source_id, year)
  ]
  
  setnames(agg, c("source_id", "V_max", "V_min", "V_mean"),
           c("id", max_name, min_name, mean_name))
  
  agg
}

#' Main optimized pipeline: compute all neighbor features and merge into cell_data.
#' Preserves the original data and adds neighbor feature columns.
add_all_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                      neighbor_source_vars) {
  
  dt <- as.data.table(cell_data)
  
  # 1. Build edge list once (fast, ~1.37M rows)
  message("Building edge list...")
  edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
  
  # 2. For each variable, compute neighbor stats and merge
  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor features for: %s", var_name))
    
    agg <- compute_neighbor_stats_fast(dt, edge_dt, var_name)
    
    # Merge back onto dt by (id, year); unmatched rows get NA (preserving original behavior)
    max_name  <- paste0(var_name, "_max")
    min_name  <- paste0(var_name, "_min")
    mean_name <- paste0(var_name, "_mean")
    
    # Remove columns if they already exist (idempotency)
    for (col in c(max_name, min_name, mean_name)) {
      if (col %in% names(dt)) dt[, (col) := NULL]
    }
    
    dt <- merge(dt, agg, by = c("id", "year"), all.x = TRUE)
  }
  
  # 3. Return as data.frame to preserve downstream compatibility
  as.data.frame(dt)
}

# ── Usage ────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# cell_data now has the 15 new neighbor columns (_max, _min, _mean × 5 vars).
# Feed directly into the existing trained Random Forest for prediction:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed over the identical set of non-NA neighbor values for each `(id, year)` as the original code. The `data.table` join `on = .(neighbor_id), allow.cartesian = TRUE` replicates the same edge × year expansion, and the `!is.na(nval)` filter matches the original `neighbor_vals[!is.na(neighbor_vals)]`. Unmatched `(id, year)` pairs receive `NA` via `all.x = TRUE`, identical to the original fallback. |
| **Trained RF model** | No model retraining occurs. The code only constructs predictor columns. |
| **Column names** | Identical `<var>_max`, `<var>_min`, `<var>_mean` naming convention. |

## Expected Performance

| Step | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M R-level iterations with string ops | Eliminated; replaced by vectorized edge list build (~1.37M rows, <1 sec) |
| `compute_neighbor_stats` × 5 vars | ~32.3M R-level `lapply` calls + `do.call(rbind)` | 5 `data.table` keyed joins + grouped aggregations (C-level, parallelized internally) |
| **Total estimated time** | **86+ hours** | **~5–15 minutes** on 16 GB RAM laptop |

The `data.table` join expands ~1.37M edges × 28 years ≈ ~38.4M rows per variable, which fits comfortably in memory and is aggregated at C level. Peak memory usage is approximately 2–3 GB above the base dataset.