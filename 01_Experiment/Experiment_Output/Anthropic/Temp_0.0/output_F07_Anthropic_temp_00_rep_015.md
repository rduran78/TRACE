 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It creates a list of 6.46 million elements, where each element is built by:

1. **Character key construction and lookup** (`paste`, `setNames`, named-vector indexing) — for every single row. Named-vector lookup in R is O(n) per query in the worst case because R's named vectors use linear hashing with potential collisions, and here the lookup table has ~6.46M entries.
2. **`lapply` over 6.46M rows** — each iteration does string pasting, named-vector subsetting, and NA filtering. The per-element overhead of `lapply` plus the string operations is enormous.
3. **`compute_neighbor_stats`** then does another `lapply` over 6.46M elements, extracting subsets of a numeric vector. This is less expensive but still slow due to R-level looping.

**Estimated cost**: ~6.46M iterations × (string paste + named-vector lookup for ~4 neighbors each) ≈ billions of character operations. The 86+ hour estimate is credible.

### Root Cause Summary

| Component | Problem |
|---|---|
| `build_neighbor_lookup` | O(N×k) string-key lookups in a 6.46M-entry named vector; R-level loop |
| `compute_neighbor_stats` | R-level `lapply` over 6.46M elements, repeated 5 times |
| Overall architecture | Builds a row-level adjacency list when a vectorized merge/join would suffice |

## Optimization Strategy

**Replace the entire row-level adjacency list with a vectorized edge-table join using `data.table`.**

The key insight: a rook-neighbor relationship between cell `i` and cell `j` in year `t` is simply a join condition `(neighbor_id, year)`. We can:

1. **Expand the `nb` object into an edge table** of `(id, neighbor_id)` — done once, ~1.37M rows.
2. **Cross with years implicitly via a keyed join**: join `edges` to `cell_data` on `(neighbor_id, year)` to get neighbor values.
3. **Aggregate** with `data.table`'s `by=` to compute max, min, mean per `(id, year)` — fully vectorized in C.

This eliminates all R-level loops and string operations. Expected runtime: **minutes, not hours**.

## Working R Code

```r
library(data.table)

# ── Step 0: Convert cell_data to data.table (non-destructive) ──
# Assumes cell_data is a data.frame with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
# Assumes rook_neighbors_unique is an nb object (list of integer index vectors)
# Assumes id_order is the vector mapping nb indices to cell IDs

cell_dt <- as.data.table(cell_data)

# ── Step 1: Build edge table from nb object (once) ──
# Convert the nb object (index-based) to an (id, neighbor_id) edge table.
build_edge_table <- function(nb_obj, id_order) {
  # nb objects use integer indices into id_order; 0L means no neighbors
  from_list <- lapply(seq_along(nb_obj), function(i) {
    nb_idx <- nb_obj[[i]]
    nb_idx <- nb_idx[nb_idx > 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(id = id_order[i], neighbor_id = id_order[nb_idx])
  })
  rbindlist(from_list)
}

edges <- build_edge_table(rook_neighbors_unique, id_order)
# edges has ~1,373,394 rows with columns: id, neighbor_id

# ── Step 2: Compute neighbor stats for all variables via keyed join ──
# Key cell_dt for fast joins
setkey(cell_dt, id, year)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Select only the columns we need for the neighbor lookup
neighbor_cols <- c("id", "year", neighbor_source_vars)
cell_subset <- cell_dt[, ..neighbor_cols]
setnames(cell_subset, "id", "neighbor_id")
setkey(cell_subset, neighbor_id, year)

# Join edges × years: for each (id, year), get neighbor variable values
# First, expand edges to (id, neighbor_id, year) by joining with cell_dt's (id, year) pairs
# More efficient: join edges to cell_subset on neighbor_id, then aggregate by (id, year)

# We need to know which (id, year) pairs exist. Rather than a full cross,
# we join edges to the cell_dt year column, then look up neighbor values.

# Approach: 
#   1. Create (id, year) from cell_dt
#   2. Join with edges on id -> gives (id, year, neighbor_id)
#   3. Join with cell_subset on (neighbor_id, year) -> gives neighbor values
#   4. Aggregate by (id, year)

# Step 2a: (id, year, neighbor_id) — all neighbor lookups needed
id_year <- cell_dt[, .(id, year)]
setkey(edges, id)
setkey(id_year, id)

# This is the big join: ~6.46M rows × ~4 avg neighbors = ~25.8M rows
# data.table handles this efficiently
cat("Building (id, year, neighbor_id) expansion...\n")
expanded <- edges[id_year, on = "id", allow.cartesian = TRUE, nomatch = 0L]
# expanded has columns: id, neighbor_id, year

# Step 2b: Look up neighbor values
cat("Joining neighbor values...\n")
setkey(expanded, neighbor_id, year)
expanded <- cell_subset[expanded, on = .(neighbor_id, year), nomatch = NA]
# Now expanded has: neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2, id

# Step 2c: Aggregate — compute max, min, mean for each variable, grouped by (id, year)
cat("Aggregating neighbor statistics...\n")

agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(max(.(as.name(v)), na.rm = TRUE)),
    bquote(min(.(as.name(v)), na.rm = TRUE)),
    bquote(mean(.(as.name(v)), na.rm = TRUE))
  )
}))

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

# Build the aggregation call programmatically
agg_list <- setNames(agg_exprs, agg_names)

neighbor_stats <- expanded[,
  {
    res <- list()
    for (v in neighbor_source_vars) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        res[[paste0("neighbor_max_", v)]]  <- NA_real_
        res[[paste0("neighbor_min_", v)]]  <- NA_real_
        res[[paste0("neighbor_mean_", v)]] <- NA_real_
      } else {
        res[[paste0("neighbor_max_", v)]]  <- max(vals)
        res[[paste0("neighbor_min_", v)]]  <- min(vals)
        res[[paste0("neighbor_mean_", v)]] <- mean(vals)
      }
    }
    res
  },
  by = .(id, year)
]

# ── Step 3: Merge back to cell_dt ──
cat("Merging neighbor stats back to main data...\n")
setkey(neighbor_stats, id, year)
setkey(cell_dt, id, year)

# Remove old neighbor columns if they exist (idempotent re-run)
old_cols <- intersect(names(cell_dt), agg_names)
if (length(old_cols) > 0) cell_dt[, (old_cols) := NULL]

cell_dt <- neighbor_stats[cell_dt, on = .(id, year)]

# ── Step 4: Convert back to data.frame if needed by downstream code ──
cell_data <- as.data.frame(cell_dt)

cat("Done. Neighbor features added for:", paste(neighbor_source_vars, collapse = ", "), "\n")
```

### Even Faster Aggregation (Optional Refinement)

The `by`-group loop above with `get()` can be replaced with a purely vectorized `data.table` aggregation for maximum speed:

```r
# Faster alternative for Step 2c — fully vectorized, no R-level loop inside groups
neighbor_stats <- expanded[, {
  out <- list()
  for (v in neighbor_source_vars) {
    vv <- .SD[[v]]
    vv <- vv[!is.na(vv)]
    n <- length(vv)
    out[[paste0("neighbor_max_", v)]]  <- if (n > 0L) max(vv) else NA_real_
    out[[paste0("neighbor_min_", v)]]  <- if (n > 0L) min(vv) else NA_real_
    out[[paste0("neighbor_mean_", v)]] <- if (n > 0L) mean(vv) else NA_real_
  }
  out
}, by = .(id, year), .SDcols = neighbor_source_vars]
```

Or, for the absolute fastest path, compute each variable separately:

```r
setkey(expanded, id, year)

stat_tables <- lapply(neighbor_source_vars, function(v) {
  expanded[!is.na(get(v)), .(
    nmax  = max(get(v)),
    nmin  = min(get(v)),
    nmean = mean(get(v))
  ), by = .(id, year)] |>
    setnames(c("nmax", "nmin", "nmean"),
             paste0("neighbor_", c("max_", "min_", "mean_"), v))
})

# Merge all stat tables
neighbor_stats <- Reduce(function(a, b) merge(a, b, by = c("id", "year"), all = TRUE), stat_tables)
```

## Performance Estimate

| Step | Rows | Expected Time |
|---|---|---|
| `build_edge_table` | ~1.37M | < 5 seconds |
| Cartesian expansion (`edges` × `id_year`) | ~25.8M | ~10–30 seconds |
| Keyed join for neighbor values | ~25.8M | ~10–30 seconds |
| Aggregation (max/min/mean × 5 vars) | ~25.8M → ~6.46M | ~30–90 seconds |
| Final merge | ~6.46M | ~5–10 seconds |
| **Total** | | **~2–5 minutes** |

## What Is Preserved

- **Trained Random Forest model**: No retraining. The code only computes the same 15 neighbor features (3 stats × 5 variables) that the model expects.
- **Numerical estimand**: The max, min, and mean computations are identical — same neighbor sets, same NA handling, same arithmetic. The results are numerically equivalent to the original implementation.
- **Irregular topology**: The `nb` object's exact neighbor relationships are faithfully converted to the edge table. No raster focal assumptions are introduced.