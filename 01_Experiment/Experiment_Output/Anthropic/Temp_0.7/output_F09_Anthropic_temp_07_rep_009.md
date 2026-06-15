 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two compounding reasons:

### 1. `build_neighbor_lookup` is O(N) over all cell-year rows (~6.46M), but it rebuilds neighbor relationships redundantly for every year.
The neighbor topology is **purely spatial** — cell A's rook neighbors are the same in 1992 as in 2019. Yet the function creates a lookup keyed on `(id, year)` pairs, performing string-pasting and hash-table lookups **6.46 million times**, each time scanning the neighbor list. This means the same spatial neighbor resolution is repeated 28 times per cell, totaling ~6.46M `lapply` iterations with string operations and named-vector lookups inside each.

### 2. `compute_neighbor_stats` iterates over 6.46M rows in R-level `lapply`.
For each of the 5 variables, it loops through all 6.46M rows in interpreted R, extracting and summarizing neighbor values. That's ~32.3M R-level loop iterations total.

### 3. The combination is catastrophic.
String concatenation (`paste`), named vector lookups, and per-row `lapply` over millions of rows in base R are orders of magnitude slower than vectorized or table-join approaches.

---

## Optimization Strategy

**Core insight:** Separate the **spatial topology** (fixed) from the **temporal attributes** (varying by year). Build the neighbor edge table once (344K cells × ~4 neighbors each ≈ 1.37M edges), then use a vectorized `data.table` join-and-aggregate approach per year.

**Steps:**

1. **Build a static edge table** from the `nb` object: a two-column `data.table` with `(focal_id, neighbor_id)` — ~1.37M rows. This is done **once**.

2. **For each variable**, join the cell-year attribute data onto the edge table by `(neighbor_id, year)`, then group-by `(focal_id, year)` to compute `max`, `min`, `mean` — all in vectorized `data.table` operations.

3. **No R-level loops over 6.46M rows.** The `data.table` grouped aggregation is executed in C and handles the entire computation in seconds per variable.

**Expected speedup:** From ~86 hours to **~1–3 minutes total** (edge table build + 5 variable aggregations).

**Preservation guarantees:**
- The trained Random Forest model is untouched — we only compute the same input features.
- The numerical estimand is identical: `max`, `min`, `mean` of the same rook-neighbor values per cell-year.

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 1: Build the static spatial edge table (run ONCE)
# ============================================================
# Inputs:
#   id_order            — vector of cell IDs in the order matching the nb object
#   rook_neighbors_unique — spdep nb object (list of integer index vectors)

build_edge_table <- function(id_order, neighbors_nb) {
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(neighbors_nb))  # ~1.37M
  
  focal_ids    <- integer(n_edges)
  neighbor_ids <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors_nb)) {
    nb_idx <- neighbors_nb[[i]]
    # spdep nb objects use 0 to indicate no neighbors
    nb_idx <- nb_idx[nb_idx != 0L]
    n <- length(nb_idx)
    if (n > 0L) {
      focal_ids[pos:(pos + n - 1L)]    <- id_order[i]
      neighbor_ids[pos:(pos + n - 1L)] <- id_order[nb_idx]
      pos <- pos + n
    }
  }
  
  data.table(focal_id = focal_ids[1:(pos - 1L)],
             neighbor_id = neighbor_ids[1:(pos - 1L)])
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# edge_dt has ~1.37M rows, two columns: focal_id, neighbor_id

cat("Edge table built:", nrow(edge_dt), "directed edges\n")

# ============================================================
# STEP 2: Convert cell_data to data.table (if not already)
# ============================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Set key for fast joins
setkey(cell_data, id, year)

# ============================================================
# STEP 3: Compute neighbor stats for all variables (vectorized)
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# We join edge_dt × year with cell_data to get neighbor attribute values,
# then aggregate by (focal_id, year).

# Get the unique years
all_years <- sort(unique(cell_data$year))

# Expand edge table across all years: ~1.37M edges × 28 years ≈ 38.5M rows
# On 16GB RAM this is feasible (38.5M rows × 3 integer cols ≈ ~460 MB)
# But we can be smarter: do it per-variable to limit peak memory.

for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "...\n")
  
  t0 <- proc.time()
  
  # Extract only the columns we need for the join
  # cell_data[, .(id, year, <var>)]
  attr_dt <- cell_data[, .(id, year, val = get(var_name))]
  setkey(attr_dt, id, year)
  
  # Cross-join edge table with years, then join neighbor attributes
  # More memory-efficient: join edge_dt with attr_dt by neighbor_id and year
  # We need to pair each edge with each year, then look up the neighbor's value.
  
  # Approach: 
  #   1. Create edges_with_years by CJ of edge rows and years? No — 
  #      each edge applies to ALL years. So:
  #   2. Join attr_dt onto edge_dt by neighbor_id = id, for all years.
  
  # Rename for clarity in join
  setnames(attr_dt, "id", "neighbor_id")
  setkey(attr_dt, neighbor_id, year)
  
  # This join: for each (focal_id, neighbor_id) edge and each year,
  # get the neighbor's value. Result: ~38.5M rows.
  merged <- edge_dt[attr_dt, on = .(neighbor_id), allow.cartesian = TRUE, nomatch = 0L]
  # merged columns: focal_id, neighbor_id, year, val
  
  # Aggregate by (focal_id, year)
  agg <- merged[!is.na(val), 
                .(nb_max  = max(val),
                  nb_min  = min(val),
                  nb_mean = mean(val)),
                by = .(focal_id, year)]
  
  # Build target column names (matching original pipeline naming convention)
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  setnames(agg, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
  setnames(agg, "focal_id", "id")
  setkey(agg, id, year)
  
  # Remove old columns if they exist (from a prior run)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) {
      cell_data[, (col) := NULL]
    }
  }
  
  # Join aggregated stats back onto cell_data
  cell_data <- agg[cell_data, on = .(id, year)]
  setkey(cell_data, id, year)
  
  elapsed <- (proc.time() - t0)["elapsed"]
  cat("  Done in", round(elapsed, 1), "seconds\n")
  
  # Clean up to free memory
  rm(attr_dt, merged, agg)
  gc()
}

cat("All neighbor features computed.\n")
cat("cell_data now has", ncol(cell_data), "columns and", nrow(cell_data), "rows.\n")

# ============================================================
# STEP 4: Predict with the existing trained Random Forest
# ============================================================
# The trained RF model object is assumed to be already in memory (e.g., `rf_model`).
# cell_data now contains all ~110+ predictor columns including the new neighbor features.
# Predict exactly as before:

# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor resolution | Per cell-year (6.46M `lapply` iters with string ops) | Static edge table built once (~344K cells) |
| Stats computation | R-level `lapply` over 6.46M rows × 5 vars | Vectorized `data.table` join + grouped aggregation |
| String operations | ~6.46M `paste()` + named vector lookups | Zero — integer key joins only |
| Estimated runtime | **~86+ hours** | **~2–5 minutes** |
| Numerical output | max/min/mean of rook-neighbor values | **Identical** max/min/mean of rook-neighbor values |
| Trained RF model | Untouched | **Untouched** |
| Peak RAM (~16 GB) | Lower but irrelevant given runtime | ~4–6 GB peak (well within budget) |

The key principle: **the spatial topology is time-invariant, so encode it once as an edge table and let `data.table`'s C-level join and grouping engine do the rest.**