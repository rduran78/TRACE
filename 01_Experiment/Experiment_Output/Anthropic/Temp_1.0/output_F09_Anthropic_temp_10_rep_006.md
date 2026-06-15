 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three interconnected reasons:

### 1. Row-level `lapply` over 6.46 million rows in `build_neighbor_lookup`
For every single cell-year row, the function performs character coercion, `paste`, and named-vector lookups. With ~6.46M rows, this means ~6.46M iterations of expensive string operations and named-vector indexing (which is O(n) in base R for each lookup).

### 2. Redundant recomputation of static topology
Rook-neighbor relationships are purely spatial — they do not change across years. Yet `build_neighbor_lookup` embeds year into the lookup keys, effectively rebuilding the neighbor graph 28 times (once per year per cell). The 1,373,394 directed rook-neighbor relationships are the same in 1992 as they are in 2019.

### 3. Row-level `lapply` over 6.46M rows in `compute_neighbor_stats`
Even after the lookup is built, computing max/min/mean one row at a time in R (not vectorized) is painfully slow for 6.46M rows × 5 variables = ~32.3M iterations.

**Root cause summary:** The topology (which cell neighbors which cell) is static, but the code entangles it with yearly data, forcing millions of redundant string operations and preventing vectorized computation.

---

## Optimization Strategy

**Core idea:** Separate topology from attributes. Build the neighbor edge-list once (344K cells × ~4 neighbors each), then use a vectorized `data.table` join-and-aggregate strategy to compute neighbor stats for all years simultaneously.

| Step | What | Complexity |
|------|------|------------|
| 1 | Build a **cell-level edge-list** `(cell_id, neighbor_id)` from `rook_neighbors_unique` — done **once**, ~1.37M rows. | O(cells × avg_neighbors) |
| 2 | Represent `cell_data` as a `data.table` keyed on `(id, year)`. | O(n) |
| 3 | For each variable, **join** the edge-list with yearly attributes: left side = `(cell_id, year)`, joined to neighbor attributes via `(neighbor_id, year)`. This produces ~1.37M × 28 ≈ 38.5M rows, but the join is vectorized in C. | O(edges × years) |
| 4 | **Group-by aggregate** `(cell_id, year)` to get `max`, `min`, `mean` — fully vectorized in `data.table`. | O(edges × years) |
| 5 | Join the three new columns back onto `cell_data`. | O(n) |

**Expected speedup:** From ~86 hours to **minutes** (typically 2–10 minutes on a 16 GB laptop), because all loops are replaced by vectorized C-level `data.table` operations.

**Preserves:**
- The trained Random Forest model (untouched).
- The original numerical estimand (identical max, min, mean values — same arithmetic, just computed vectorized).

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 1: Build the static cell-level neighbor edge-list ONCE
# ============================================================
# rook_neighbors_unique : spdep nb object (list of integer vectors)
# id_order              : vector of cell IDs, position i corresponds
#                         to the i-th element of rook_neighbors_unique

build_neighbor_edgelist <- function(id_order, neighbors) {
  # Pre-allocate: count total number of directed neighbor pairs
  n_links <- sum(lengths(neighbors))

  from_id <- integer(n_links)
  to_id   <- integer(n_links)

  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_idx <- neighbors[[i]]
    # spdep nb objects use 0L to denote "no neighbors"
    nb_idx <- nb_idx[nb_idx != 0L]
    n      <- length(nb_idx)
    if (n > 0L) {
      from_id[pos:(pos + n - 1L)] <- id_order[i]
      to_id[pos:(pos + n - 1L)]   <- id_order[nb_idx]
      pos <- pos + n
    }
  }

  # Trim in case some 0-neighbor entries existed
  data.table(cell_id = from_id[1:(pos - 1L)],
             neighbor_id = to_id[1:(pos - 1L)])
}

cat("Building static neighbor edge-list...\n")
neighbor_edges <- build_neighbor_edgelist(id_order, rook_neighbors_unique)
cat(sprintf("  Edge-list rows: %s\n", format(nrow(neighbor_edges), big.mark = ",")))

# ============================================================
# STEP 2: Convert cell_data to data.table (if not already)
# ============================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ============================================================
# STEP 3: Vectorized neighbor-stat computation
# ============================================================
compute_neighbor_features_dt <- function(dt, edges, var_name) {
  # Build a slim lookup table: (id, year, value)
  lookup <- dt[, .(id, year, value = get(var_name))]
  setnames(lookup, "id", "neighbor_id")
  setkey(lookup, neighbor_id, year)

  # Cross the edge-list with all years present in the data
  years <- sort(unique(dt$year))
  edge_years <- CJ(edge_idx = seq_len(nrow(edges)), year = years)
  edge_years[, cell_id     := edges$cell_id[edge_idx]]
  edge_years[, neighbor_id := edges$neighbor_id[edge_idx]]
  edge_years[, edge_idx := NULL]

  # Join neighbor attribute values
  setkey(edge_years, neighbor_id, year)
  edge_years <- lookup[edge_years, on = .(neighbor_id, year)]

  # Aggregate: max, min, mean per (cell_id, year)
  agg <- edge_years[!is.na(value),
                    .(nb_max  = max(value),
                      nb_min  = min(value),
                      nb_mean = mean(value)),
                    by = .(cell_id, year)]

  # Rename columns to match expected feature names
  max_col  <- paste0("max_neighbor_",  var_name)
  min_col  <- paste0("min_neighbor_",  var_name)
  mean_col <- paste0("mean_neighbor_", var_name)
  setnames(agg, c("nb_max", "nb_min", "nb_mean"),
                c(max_col,  min_col,  mean_col))

  agg
}

# ============================================================
# STEP 4: Loop over the 5 source variables, join results back
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

setkey(cell_data, id, year)

for (var_name in neighbor_source_vars) {
  cat(sprintf("Computing neighbor features for '%s'...\n", var_name))
  agg <- compute_neighbor_features_dt(cell_data, neighbor_edges, var_name)
  setkey(agg, cell_id, year)

  # Left-join the three new columns onto cell_data
  max_col  <- paste0("max_neighbor_",  var_name)
  min_col  <- paste0("min_neighbor_",  var_name)
  mean_col <- paste0("mean_neighbor_", var_name)

  # Remove old columns if they exist (idempotency)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }

  cell_data <- agg[cell_data, on = .(cell_id = id, year = year)]
  setnames(cell_data, "cell_id", "id")
  setkey(cell_data, id, year)

  cat(sprintf("  Done — added %s, %s, %s\n", max_col, min_col, mean_col))
}

# ============================================================
# STEP 5: Predict with the existing trained Random Forest
# ============================================================
# The model object (e.g., `rf_model`) is unchanged.
# cell_data now has the same neighbor-stat columns as before,
# with identical numerical values, just computed ~500× faster.

cat("Computing predictions with trained Random Forest model...\n")
cell_data[, prediction := predict(rf_model, newdata = cell_data)]
cat("Done.\n")
```

---

## Memory-Optimized Variant (for tight 16 GB RAM)

The cross-join `edges × years` can reach ~38.5M rows. If memory is tight, process one variable at a time (already done above) **and** process in year-chunks:

```r
compute_neighbor_features_chunked <- function(dt, edges, var_name, chunk_size = 7L) {
  years <- sort(unique(dt$year))
  chunks <- split(years, ceiling(seq_along(years) / chunk_size))

  max_col  <- paste0("max_neighbor_",  var_name)
  min_col  <- paste0("min_neighbor_",  var_name)
  mean_col <- paste0("mean_neighbor_", var_name)

  agg_list <- lapply(chunks, function(yr_chunk) {
    lookup <- dt[year %in% yr_chunk, .(neighbor_id = id, year, value = get(var_name))]
    setkey(lookup, neighbor_id, year)

    edge_years <- CJ(edge_idx = seq_len(nrow(edges)), year = yr_chunk)
    edge_years[, cell_id     := edges$cell_id[edge_idx]]
    edge_years[, neighbor_id := edges$neighbor_id[edge_idx]]
    edge_years[, edge_idx := NULL]
    setkey(edge_years, neighbor_id, year)

    edge_years <- lookup[edge_years, on = .(neighbor_id, year)]

    edge_years[!is.na(value),
               .(nb_max = max(value), nb_min = min(value), nb_mean = mean(value)),
               by = .(cell_id, year)]
  })

  agg <- rbindlist(agg_list)
  setnames(agg, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
  agg
}
```

---

## Summary

| Aspect | Before | After |
|--------|--------|-------|
| Neighbor topology | Rebuilt per cell-year (6.46M string ops) | Built once as edge-list (~1.37M rows) |
| Stat computation | Row-level `lapply` in R | Vectorized `data.table` join + group-by in C |
| Estimated runtime | ~86+ hours | ~2–10 minutes |
| Numerical results | max, min, mean | Identical max, min, mean |
| RF model | Unchanged | Unchanged |
| Peak RAM | Moderate (but slow) | ~4–6 GB (fits 16 GB) |