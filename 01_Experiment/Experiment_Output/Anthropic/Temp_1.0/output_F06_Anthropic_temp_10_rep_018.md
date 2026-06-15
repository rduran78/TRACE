 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` — O(n²) row-level `lapply`

This function iterates over **~6.46 million rows**, and for each row it:
1. Looks up the cell's reference index in `id_to_ref`.
2. Retrieves neighbor cell IDs.
3. Constructs string keys (`paste(id, year, sep="_")`).
4. Looks up those keys in `idx_lookup`.

The string-paste-and-match pattern across 6.46M rows is extremely expensive. The named-vector lookup `idx_lookup[neighbor_keys]` has hidden O(n) hashing costs at scale, and the `lapply` over millions of rows in R's interpreted loop is inherently slow.

### Bottleneck 2: `compute_neighbor_stats` — Repeated row-level R loops

For each of the 5 variables, `compute_neighbor_stats` runs an `lapply` over 6.46M rows, subsetting a numeric vector by index, removing NAs, and computing `max`, `min`, `mean`. That's **5 × 6.46M = 32.3M** R-level function calls with per-element subsetting.

### Why raster focal/kernel operations aren't directly applicable

Raster focal operations (e.g., `terra::focal`) assume a regular rectangular grid with uniform kernel windows. While the 344K cells *might* be on a regular grid, the panel structure (28 years) and the fact that `rook_neighbors_unique` is a precomputed `spdep::nb` object (which can handle irregular boundaries, missing cells, coastal cells, etc.) means a focal approach could silently change results at edges and boundaries. The instructions say to **preserve the original numerical estimand**, so we must keep the exact neighbor structure. However, the *concept* of vectorized spatial operations inspires the solution: **vectorize using a sparse adjacency matrix and matrix algebra**.

---

## Optimization Strategy

### Strategy: Sparse Matrix Multiplication

The key insight: computing `mean` of neighbor values is equivalent to multiplying a **row-normalized sparse adjacency matrix** by the value vector. Similarly, `max` and `min` can be computed via sparse-matrix-guided grouped operations using `data.table`.

**Step-by-step plan:**

1. **Replace `build_neighbor_lookup`** with construction of a sparse adjacency matrix (cell × cell) using the `Matrix` package, then expand it to the panel level via a year-merge — or better, operate at the cell level per year using `data.table` grouping.

2. **Replace `compute_neighbor_stats`** with vectorized `data.table` grouped operations: for each cell-year, join to neighbors and compute `max`, `min`, `mean` in bulk.

3. **All 5 variables at once** in a single join pass, rather than 5 separate loops.

**Expected speedup:** From ~86 hours to ~2–10 minutes, because:
- The neighbor join is done once via `data.table` keyed merge (vectorized C code).
- Aggregation uses `data.table`'s optimized `GForce` for `max`, `min`, `mean`.
- No R-level `lapply` over millions of rows.

---

## Working R Code

```r
library(data.table)

# ── Step 0: Convert to data.table ──────────────────────────────────────────────
# cell_data must have columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: spdep::nb object (list of integer neighbor indices)
# id_order: vector of cell IDs in the order corresponding to rook_neighbors_unique

dt <- as.data.table(cell_data)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# ── Step 1: Build an edge list from the nb object ─────────────────────────────
# Each element of rook_neighbors_unique[[i]] gives the indices (into id_order)
# of neighbors of cell id_order[i].

build_edge_list <- function(id_order, nb_obj) {
  # Pre-allocate: count total edges
  n_cells <- length(nb_obj)
  edge_counts <- vapply(nb_obj, function(x) {
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1))
  total_edges <- sum(edge_counts)
  
  from_id <- integer(total_edges)
  to_id   <- integer(total_edges)
  
  pos <- 1L
  for (i in seq_len(n_cells)) {
    nbrs <- nb_obj[[i]]
    if (length(nbrs) == 1L && nbrs[1] == 0L) next
    n <- length(nbrs)
    from_id[pos:(pos + n - 1L)] <- id_order[i]
    to_id[pos:(pos + n - 1L)]   <- id_order[nbrs]
    pos <- pos + n
  }
  
  data.table(focal_id = from_id, neighbor_id = to_id)
}

edges <- build_edge_list(id_order, rook_neighbors_unique)
# edges now has columns: focal_id, neighbor_id
# This should have ~1,373,394 rows

cat("Edge list built:", nrow(edges), "directed edges\n")

# ── Step 2: Compute all neighbor stats in one vectorized pass per year ────────
# Strategy: join edges to data by (neighbor_id, year), then group by (focal_id, year)
# to compute max, min, mean for each variable.

# Ensure dt is keyed for fast joins
setkey(dt, id, year)

# Create neighbor data: for each edge and each year, look up the neighbor's values
# We do a single large join: edges × years

# Prepare a "neighbor lookup" table: for each (focal_id, year), we need all
# neighbor values. We achieve this by joining edges with dt on neighbor_id = id.

# Rename for clarity in the join
neighbor_dt <- dt[, c("id", "year", neighbor_source_vars), with = FALSE]
setnames(neighbor_dt, "id", "neighbor_id")
setkey(neighbor_dt, neighbor_id, year)

# Expand edges by year via join: each edge gets one row per year the neighbor exists
# This is the big join: ~1.37M edges × 28 years ≈ 38.4M rows (upper bound)
# In practice, not all cells exist in all years, so it may be less.

setkey(edges, neighbor_id)

# Perform the join: for each (focal_id, neighbor_id) pair, get all years
# where the neighbor has data
cat("Performing edge-year join...\n")
edge_years <- merge(edges, neighbor_dt, by = "neighbor_id", allow.cartesian = TRUE)
# edge_years columns: neighbor_id, focal_id, year, ntl, ec, pop_density, def, usd_est_n2

cat("Edge-year table:", nrow(edge_years), "rows\n")

# ── Step 3: Aggregate by (focal_id, year) ────────────────────────────────────
cat("Computing neighbor statistics...\n")

# Build aggregation expressions dynamically
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

# Use data.table's efficient grouped aggregation
# Construct the j expression
agg_call <- as.call(c(as.name("list"),
  setNames(agg_exprs, agg_names)
))

neighbor_stats <- edge_years[, eval(agg_call), by = .(focal_id, year)]

# Handle Inf/-Inf from max/min on all-NA groups (shouldn't happen if edges exist,
# but be safe)
for (col in agg_names) {
  vals <- neighbor_stats[[col]]
  vals[is.infinite(vals)] <- NA_real_
  set(neighbor_stats, j = col, value = vals)
}

cat("Neighbor stats computed:", nrow(neighbor_stats), "rows,",
    ncol(neighbor_stats) - 2, "new features\n")

# ── Step 4: Merge back into original data ─────────────────────────────────────
setnames(neighbor_stats, "focal_id", "id")
setkey(neighbor_stats, id, year)
setkey(dt, id, year)

dt <- merge(dt, neighbor_stats, by = c("id", "year"), all.x = TRUE)

cat("Final dataset:", nrow(dt), "rows,", ncol(dt), "columns\n")

# ── Step 5: Convert back to data.frame if needed for predict() ────────────────
cell_data <- as.data.frame(dt)

# ── Step 6: Apply the pre-trained Random Forest (unchanged) ───────────────────
# The trained model object (e.g., `rf_model`) is used as-is.
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Verification Script

To confirm numerical equivalence with the original implementation on a subset:

```r
# ── Verify numerical equivalence on a sample ──────────────────────────────────
verify_equivalence <- function(cell_data_original, cell_data_optimized,
                                neighbor_source_vars, n_sample = 1000) {
  set.seed(42)
  sample_rows <- sample(nrow(cell_data_original), min(n_sample, nrow(cell_data_original)))
  
  suffixes <- c("max_", "min_", "mean_")
  all_ok <- TRUE
  
  for (v in neighbor_source_vars) {
    for (s in suffixes) {
      col <- paste0("neighbor_", s, v)
      orig <- cell_data_original[[col]][sample_rows]
      optim <- cell_data_optimized[[col]][sample_rows]
      
      both_na <- is.na(orig) & is.na(optim)
      both_val <- !is.na(orig) & !is.na(optim)
      
      if (any(is.na(orig) != is.na(optim))) {
        cat("MISMATCH NA pattern in", col, "\n")
        all_ok <- FALSE
        next
      }
      
      if (any(both_val)) {
        max_diff <- max(abs(orig[both_val] - optim[both_val]))
        if (max_diff > 1e-10) {
          cat("MISMATCH values in", col, "- max diff:", max_diff, "\n")
          all_ok <- FALSE
        }
      }
    }
  }
  
  if (all_ok) cat("All neighbor statistics match within tolerance.\n")
  all_ok
}
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | `lapply` over 6.46M rows, string key matching | One-time edge list build from `nb` object (~1.37M rows) |
| **Stats computation** | 5 × `lapply` over 6.46M rows, per-row subsetting | Single `data.table` grouped aggregation on ~38M-row join |
| **Estimated runtime** | ~86+ hours | ~2–10 minutes |
| **Peak RAM** | Moderate (list of 6.46M vectors) | ~4–6 GB (edge-year table in memory) |
| **Numerical equivalence** | Baseline | Exact (same `max`, `min`, `mean` over same neighbor sets) |
| **RF model** | Unchanged | Unchanged — no retraining |

The optimization replaces two R-level interpreted loops over millions of rows with `data.table`'s vectorized C-level merge and `GForce`-optimized grouped aggregation, yielding orders-of-magnitude speedup while preserving exact numerical results.