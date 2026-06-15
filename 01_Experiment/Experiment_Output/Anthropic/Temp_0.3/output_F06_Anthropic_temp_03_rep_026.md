 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` — O(n) `lapply` over 6.46M rows

For every single cell-year row, the function:
1. Looks up the spatial cell's rook neighbors (fine — fast hash lookup).
2. Constructs `paste(neighbor_id, year, "_")` key strings for each neighbor.
3. Looks up those keys in `idx_lookup` (a named character vector).

**Problem:** `paste()` and named-vector lookup on 6.46 million rows, each with ~4 neighbors on average, generates ~25M string constructions and hash lookups. The `lapply` loop is single-threaded and allocates millions of small character vectors. The resulting `neighbor_lookup` list itself is a 6.46M-element list of integer vectors — enormous memory overhead from list metadata alone.

### Bottleneck B: `compute_neighbor_stats` — O(n) `lapply` over 6.46M rows, repeated 5 times

For each of the 5 source variables, the code iterates over all 6.46M rows, subsets a numeric vector by index, removes NAs, and computes `max`, `min`, `mean`. This is called 5 times = 32.3M R-level function calls. Each call has overhead from anonymous function dispatch, `is.na()`, subsetting, and `c()`.

### Combined effect
The two stages together produce an estimated 86+ hours because R's `lapply` over millions of elements with string operations and small-vector allocations is extremely slow.

### Why raster focal/kernel operations don't directly apply
Focal operations (e.g., `terra::focal`) assume a regular rectangular grid with a fixed kernel. Here, the neighbor structure comes from an irregular `spdep::nb` object (rook contiguity on an arbitrary polygon/grid layout), and the data is a panel (cell × year). Focal operations would only work if the spatial grid is perfectly regular **and** you reshape data into a raster stack per year. That's possible but fragile and risks subtle misalignment. The better approach is to vectorize the neighbor computation using sparse matrix algebra, which is both general and fast.

---

## 2. Optimization Strategy

### Strategy: Sparse Matrix Multiplication

The key insight: **max, min, and mean over neighbors can be computed using sparse matrix operations** that process all 6.46M rows simultaneously in compiled C code, eliminating all R-level loops.

**For `mean`:** If `W` is the row-normalized spatial weights matrix (expanded to cell-year dimensions), then `W %*% x` gives the neighbor mean for every row at once.

**For `max` and `min`:** We can't use standard matrix multiplication, but we can use the sparse structure to extract neighbor values in bulk and compute grouped max/min using `data.table` or vectorized operations.

**Step-by-step:**

1. **Build a sparse adjacency matrix** from the `nb` object (344K × 344K) — done once.
2. **Expand it to cell-year space** by exploiting the fact that neighbors are looked up within the same year. Instead of building a 6.46M × 6.46M matrix, we compute per-year or use a block-diagonal structure. Even simpler: we work in "spatial ID" space and use `data.table` joins by year.
3. **Vectorized grouped stats** using `data.table`: create an edge list `(row_i, neighbor_j)`, join the variable values by `(neighbor_j, year)`, then compute `max/min/mean` grouped by `(row_i, year)`.

**Expected speedup:** From 86+ hours to **~5–15 minutes**. The edge list has ~1.37M spatial edges × 28 years ≈ 38.5M rows — easily handled by `data.table` in seconds per variable.

**Memory:** The edge table is ~38.5M rows × 3 columns ≈ ~900 MB. With 16 GB RAM this is feasible, especially since we process one variable at a time.

---

## 3. Working R Code

```r
# ==============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Replaces: build_neighbor_lookup + compute_neighbor_stats loop
# Preserves: exact same numerical results (max, min, mean of rook neighbors)
# Preserves: trained Random Forest model (no retraining)
# ==============================================================================

library(data.table)
library(spdep)

# --------------------------------------------------------------------------
# Step 1: Build spatial edge list from the nb object (done once)
# --------------------------------------------------------------------------
build_spatial_edge_list <- function(id_order, neighbors_nb) {
  # neighbors_nb is an nb object (list of integer index vectors)
  # id_order is the vector of spatial cell IDs in the order matching the nb object
  
  n <- length(neighbors_nb)
  
  # Pre-calculate total edges for pre-allocation
  edge_counts <- vapply(neighbors_nb, length, integer(1))
  total_edges <- sum(edge_counts)
  
  from_id <- integer(total_edges)
  to_id   <- integer(total_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    nb_idx <- neighbors_nb[[i]]
    if (length(nb_idx) > 0 && !(length(nb_idx) == 1 && nb_idx[1] == 0L)) {
      len <- length(nb_idx)
      from_id[pos:(pos + len - 1L)] <- id_order[i]
      to_id[pos:(pos + len - 1L)]   <- id_order[nb_idx]
      pos <- pos + len
    }
  }
  
  # Trim if any nb entries were 0 (no-neighbor sentinel in spdep)
  if (pos <= total_edges) {
    from_id <- from_id[1:(pos - 1L)]
    to_id   <- to_id[1:(pos - 1L)]
  }
  
  data.table(from_id = from_id, to_id = to_id)
}

# --------------------------------------------------------------------------
# Step 2: Compute neighbor stats for one variable (vectorized)
# --------------------------------------------------------------------------
compute_neighbor_stats_fast <- function(cell_dt, edge_dt, var_name) {
  # cell_dt must be a data.table with columns: id, year, <var_name>
  # edge_dt has columns: from_id, to_id
  
  # Create a keyed lookup: for each (to_id, year), the variable value
  lookup <- cell_dt[, .(id, year, val = get(var_name))]
  setkey(lookup, id, year)
  
  # Expand edges across all years via join:
  # For each edge (from_id -> to_id), for each year, get the neighbor's value
  # 
  # Approach: join edge_dt with cell_dt on to_id = id, by year
  # This is effectively a cross of edges × years, but done efficiently via join
  
  # First, get all (from_id, year) combinations that exist in the data
  from_years <- cell_dt[, .(from_id = id, year)]
  
  # Join edges to get (from_id, to_id, year)
  # Use edge_dt and expand by year through the from_id
  edge_year <- edge_dt[from_years, on = .(from_id), allow.cartesian = TRUE, nomatch = 0L]
  # edge_year now has columns: from_id, to_id, year
  
  # Join to get neighbor values
  edge_year[lookup, val := i.val, on = .(to_id = id, year)]
  
  # Remove NA values (as in original code)
  edge_year <- edge_year[!is.na(val)]
  
  # Compute grouped stats
  stats <- edge_year[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = .(from_id, year)]
  
  # Rename columns to match original naming convention
  # Original code produces columns like: ntl_max_neighbor, ntl_min_neighbor, ntl_mean_neighbor
  max_col  <- paste0(var_name, "_max_neighbor")
  min_col  <- paste0(var_name, "_min_neighbor")
  mean_col <- paste0(var_name, "_mean_neighbor")
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
  
  stats
}

# --------------------------------------------------------------------------
# Step 3: Main pipeline
# --------------------------------------------------------------------------
optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique) {
  
  cat("Converting to data.table...\n")
  cell_dt <- as.data.table(cell_data)
  
  cat("Building spatial edge list...\n")
  edge_dt <- build_spatial_edge_list(id_order, rook_neighbors_unique)
  cat(sprintf("  Edge list: %d directed edges\n", nrow(edge_dt)))
  
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Computing neighbor stats for '%s'...\n", var_name))
    t0 <- proc.time()
    
    stats <- compute_neighbor_stats_fast(cell_dt, edge_dt, var_name)
    
    # Determine the new column names
    max_col  <- paste0(var_name, "_max_neighbor")
    min_col  <- paste0(var_name, "_min_neighbor")
    mean_col <- paste0(var_name, "_mean_neighbor")
    
    # Remove old columns if they exist (in case of re-run)
    for (col in c(max_col, min_col, mean_col)) {
      if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
    }
    
    # Left join stats back to cell_dt
    cell_dt <- merge(cell_dt, stats, by.x = c("id", "year"),
                     by.y = c("from_id", "year"), all.x = TRUE)
    
    elapsed <- (proc.time() - t0)[3]
    cat(sprintf("  Done in %.1f seconds\n", elapsed))
  }
  
  cat("All neighbor features computed.\n")
  
  # Convert back to data.frame if the downstream RF predict expects one
  as.data.frame(cell_dt)
}

# --------------------------------------------------------------------------
# Usage (drop-in replacement for the original loop)
# --------------------------------------------------------------------------
# cell_data <- optimize_neighbor_features(cell_data, id_order, rook_neighbors_unique)
#
# # Then proceed with prediction using the pre-trained RF model as before:
# # predictions <- predict(rf_model, newdata = cell_data)
```

---

## 4. Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M R-level `lapply` iterations with `paste` + named-vector lookup | Single vectorized edge-list construction (~1.37M edges), built once in seconds |
| **Stats computation** | 6.46M `lapply` iterations × 5 variables = 32.3M R function calls | 5 `data.table` grouped aggregations on ~38.5M-row join — compiled C internals |
| **Estimated runtime** | 86+ hours | **5–15 minutes** |
| **Memory** | ~6.46M-element list of integer vectors (high overhead) | ~38.5M-row data.table per variable (peak), freed after merge |
| **Numerical results** | max, min, mean of non-NA rook-neighbor values | **Identical** — same operations, same NA handling |
| **RF model** | Pre-trained, not modified | **Preserved** — only the feature-engineering step changes |

### Why not raster focal?

While `terra::focal()` with a rook kernel (3×3, center=FALSE, sides only) would be blazingly fast for a single regular raster, this pipeline has complications that make it fragile:
- The grid may not be perfectly regular (the `nb` object suggests arbitrary topology).
- Panel structure (28 years) would require reshaping into 28 raster layers per variable and back.
- Any cell-ID-to-raster-cell misalignment would silently corrupt results.

The sparse `data.table` join approach is nearly as fast, works with any `nb` topology, and is trivially verifiable against the original output.