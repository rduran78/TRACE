 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations over 6.46M Rows

`build_neighbor_lookup` calls `lapply` over every row (6.46M iterations). Inside each iteration it:

1. **Pastes** string keys (`paste(neighbor_cell_ids, data$year[i], sep = "_")`) — allocating character vectors per row.
2. **Looks up** those keys in a named character vector (`idx_lookup[neighbor_keys]`) — which is an O(k) hash lookup per neighbor, but the constant factor of R's named-vector lookup on a 6.46M-element vector is large.

This means roughly **6.46M × ~8 neighbors × (paste + hash lookup)** = ~50 billion character operations. That's the 86-hour wall.

### Why It's a Broader Algorithmic Issue

The string-keying approach **conflates two independent index dimensions** (cell ID and year) into a single flat string key. But the neighbor structure is **time-invariant**: cell *i*'s neighbors are the same in every year. The lookup can be decomposed:

- **Step 1 (once):** For each of the 344,208 cells, find its neighbor cell indices (integer vector). This is purely spatial — done once.
- **Step 2 (vectorized):** For each cell-year row, map the cell's spatial neighbor list to row indices using integer arithmetic, not string hashing.

Because the panel is balanced (every cell appears in every year), the row index of cell `c` in year `y` is deterministic given a sort order. This converts the entire lookup from **6.46M string-hash operations** to **a single integer-arithmetic broadcast**.

---

## Optimization Strategy

| Aspect | Old | New |
|---|---|---|
| Neighbor mapping | Per-row string paste + hash lookup | One-time integer cell-index map + vectorized row-offset arithmetic |
| Complexity | O(R × K × string_ops) ≈ 50B char ops | O(C × K) integer ops ≈ 2.7M, then vectorized column extraction |
| `compute_neighbor_stats` | `lapply` over 6.46M rows | Vectorized `data.table` join or matrix-column operations |
| Estimated time | 86+ hours | Minutes |

**Key insight:** If the data is sorted by `(id, year)`, and every cell has all 28 years, then the row for cell-index `c` in year-index `t` is `(c - 1) * 28 + t`. Neighbor row indices are just the neighbor cell indices plugged into the same formula. No strings needed.

---

## Working R Code

```r
library(data.table)

# ==============================================================
# 0. Ensure data is a data.table sorted by (id, year)
# ==============================================================
cell_data <- as.data.table(cell_data)
setkey(cell_data, id, year)

# Verify balanced panel
n_years <- uniqueN(cell_data$year)          # 28
n_cells <- uniqueN(cell_data$id)            # 344,208
stopifnot(nrow(cell_data) == n_cells * n_years)

# ==============================================================
# 1. Build integer cell-index map (once, spatial only)
#    id_order is the vector of cell IDs matching rook_neighbors_unique
# ==============================================================
# Map from cell ID -> position in the sorted unique IDs in cell_data
unique_ids <- unique(cell_data$id)                       # already sorted by setkey
id_to_row_block <- setNames(seq_along(unique_ids), as.character(unique_ids))

# Map from id_order position -> cell_data block position
id_order_to_block <- id_to_row_block[as.character(id_order)]

# Build spatial neighbor list in terms of cell_data block indices
# rook_neighbors_unique[[j]] gives neighbor positions in id_order for cell id_order[j]
# We need: for each cell_data block index b, the neighbor block indices.

# Invert: for each block index b, which id_order index j does it come from?
block_to_idorder <- integer(n_cells)
block_to_idorder[id_to_row_block[as.character(id_order)]] <- seq_along(id_order)

# Now build neighbor list indexed by cell_data block position
# neighbor_blocks[[b]] = integer vector of cell_data block positions of neighbors of cell b
cat("Building spatial neighbor index...\n")
neighbor_blocks <- vector("list", n_cells)
for (b in seq_len(n_cells)) {
  j <- block_to_idorder[b]
  nb_in_idorder <- rook_neighbors_unique[[j]]
  if (length(nb_in_idorder) == 0L) {
    neighbor_blocks[[b]] <- integer(0)
  } else {
    nb_blocks <- id_order_to_block[nb_in_idorder]
    neighbor_blocks[[b]] <- as.integer(nb_blocks[!is.na(nb_blocks)])
  }
}
cat("Done. Spatial neighbor index built for", n_cells, "cells.\n")

# ==============================================================
# 2. Precompute a neighbor-row matrix (avoids per-row work entirely)
#
#    Row index of block b, year-offset t (1..28):
#        row = (b - 1) * n_years + t
#
#    For each row i with block b(i) and year-offset t(i):
#        neighbor rows = (neighbor_blocks[[b(i)]] - 1) * n_years + t(i)
#
#    We vectorize this by building a sparse "neighbor row" structure
#    as two parallel vectors: (source_row, neighbor_row), then use
#    data.table grouping.
# ==============================================================

cat("Building full neighbor-row map...\n")

# Precompute max number of neighbors for pre-allocation
max_nb <- max(vapply(neighbor_blocks, length, integer(1)))
cat("Max neighbors per cell:", max_nb, "\n")

# Pre-allocate edge list vectors
# Total directed edges = sum of neighbor counts * n_years
total_edges <- sum(vapply(neighbor_blocks, length, integer(1))) * n_years

src_rows <- integer(total_edges)
nbr_rows <- integer(total_edges)

ptr <- 0L
for (b in seq_len(n_cells)) {
  nb <- neighbor_blocks[[b]]
  k <- length(nb)
  if (k == 0L) next
  for (t in seq_len(n_years)) {
    src_row <- (b - 1L) * n_years + t
    nb_row  <- (nb - 1L) * n_years + t
    idx_range <- ptr + seq_len(k)
    src_rows[idx_range] <- src_row
    nbr_rows[idx_range] <- nb_row
    ptr <- ptr + k
  }
}

# Trim if any cells had zero neighbors
src_rows <- src_rows[seq_len(ptr)]
nbr_rows <- nbr_rows[seq_len(ptr)]

cat("Edge list built:", ptr, "directed edges.\n")

# ==============================================================
# 3. Compute neighbor stats for all variables (vectorized)
# ==============================================================
edge_dt <- data.table(src = src_rows, nbr = nbr_rows)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Computing neighbor stats for:", var_name, "\n")
  
  # Attach neighbor values
  edge_dt[, nbr_val := cell_data[[var_name]][nbr]]
  
  # Remove NA neighbor values
  valid <- edge_dt[!is.na(nbr_val)]
  
  # Aggregate by source row
  agg <- valid[, .(
    nb_max  = max(nbr_val),
    nb_min  = min(nbr_val),
    nb_mean = mean(nbr_val)
  ), by = src]
  
  # Initialize columns with NA
  max_col  <- paste0(var_name, "_nb_max")
  min_col  <- paste0(var_name, "_nb_min")
  mean_col <- paste0(var_name, "_nb_mean")
  
  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]
  
  # Fill in computed values
  cell_data[agg$src, (max_col)  := agg$nb_max]
  cell_data[agg$src, (min_col)  := agg$nb_min]
  cell_data[agg$src, (mean_col) := agg$nb_mean]
  
  cat("  Done:", var_name, "\n")
}

# Clean up large temporaries
rm(edge_dt, src_rows, nbr_rows, neighbor_blocks, block_to_idorder, id_order_to_block)
gc()

cat("All neighbor features computed.\n")
```

---

## If the Edge List Is Too Large for RAM (~580M edges × 2 × 4 bytes ≈ 4.6 GB)

With 16 GB RAM this should fit, but if it's tight, process year-by-year to reduce peak memory by 28×:

```r
# ==============================================================
# Memory-conservative variant: process one year at a time
# ==============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-initialize all output columns
for (var_name in neighbor_source_vars) {
  cell_data[, paste0(var_name, "_nb_max")  := NA_real_]
  cell_data[, paste0(var_name, "_nb_min")  := NA_real_]
  cell_data[, paste0(var_name, "_nb_mean") := NA_real_]
}

years <- sort(unique(cell_data$year))

for (t_idx in seq_along(years)) {
  yr <- years[t_idx]
  cat("Processing year:", yr, "(", t_idx, "/", length(years), ")\n")
  
  # Row range for this year: all rows with year-offset t_idx
  # Because data is sorted by (id, year), row for block b, year t_idx = (b-1)*n_years + t_idx
  yr_rows <- seq(from = t_idx, by = n_years, length.out = n_cells)  # one row per cell
  
  # Build edge list for this year only
  # For each cell b, neighbors are neighbor_blocks[[b]]
  # Source row = (b-1)*n_years + t_idx, neighbor row = (nb-1)*n_years + t_idx
  
  n_edges_yr <- sum(vapply(neighbor_blocks, length, integer(1)))
  src_yr <- integer(n_edges_yr)
  nbr_yr <- integer(n_edges_yr)
  
  ptr <- 0L
  for (b in seq_len(n_cells)) {
    nb <- neighbor_blocks[[b]]
    k <- length(nb)
    if (k == 0L) next
    idx_range <- ptr + seq_len(k)
    src_yr[idx_range] <- (b - 1L) * n_years + t_idx
    nbr_yr[idx_range] <- (nb - 1L) * n_years + t_idx
    ptr <- ptr + k
  }
  src_yr <- src_yr[seq_len(ptr)]
  nbr_yr <- nbr_yr[seq_len(ptr)]
  
  edge_yr <- data.table(src = src_yr, nbr = nbr_yr)
  
  for (var_name in neighbor_source_vars) {
    edge_yr[, nbr_val := cell_data[[var_name]][nbr]]
    valid <- edge_yr[!is.na(nbr_val)]
    
    if (nrow(valid) == 0L) next
    
    agg <- valid[, .(
      nb_max  = max(nbr_val),
      nb_min  = min(nbr_val),
      nb_mean = mean(nbr_val)
    ), by = src]
    
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")
    
    set(cell_data, i = agg$src, j = max_col,  value = agg$nb_max)
    set(cell_data, i = agg$src, j = min_col,  value = agg$nb_min)
    set(cell_data, i = agg$src, j = mean_col, value = agg$nb_mean)
  }
  
  rm(edge_yr, src_yr, nbr_yr)
}

cat("All neighbor features computed (year-by-year).\n")
```

---

## Summary

| Item | Detail |
|---|---|
| **Root cause** | String-key construction and hash lookup repeated 6.46M × ~8 times inside an R `lapply` |
| **Broader pattern** | The neighbor graph is time-invariant but was being re-resolved per row via string keys; the spatial and temporal dimensions were needlessly entangled |
| **Fix** | Decompose into (1) a one-time spatial integer neighbor map, (2) deterministic row-index arithmetic exploiting the balanced panel sort order, (3) vectorized `data.table` grouped aggregation |
| **Numerical equivalence** | Same `max`, `min`, `mean` over the same non-NA neighbor values — identical estimand |
| **RF model** | Untouched; only the feature-construction pipeline is changed |
| **Expected speedup** | From ~86 hours to **~5–15 minutes** on the same laptop |