 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of **two fundamental bottlenecks**:

### 1. `build_neighbor_lookup` is O(n²)-like in practice
This function iterates over all **~6.46 million cell-year rows** and, for each row, performs:
- A named-vector lookup (`id_to_ref`) — O(1) amortized but with high constant factor for character hashing at scale.
- Construction of `neighbor_keys` by pasting cell IDs with years — millions of `paste()` calls.
- A named-vector lookup into `idx_lookup` (a named vector of length 6.46M) — extremely expensive repeated hash lookups.

The result is a **list of 6.46 million integer vectors**. This is the single largest bottleneck. The key insight is: **the neighbor topology is static across years**. Cell 42's rook neighbors are the same in 1992 as in 2019. Yet this function redundantly recomputes neighbor row indices for every cell-year combination, doing 28× the necessary topological work and creating a 6.46M-element list when a 344,208-element list would suffice.

### 2. `compute_neighbor_stats` uses an R-level `lapply` over 6.46M elements
Even after the lookup is built, computing max/min/mean via `lapply` over 6.46 million list elements is slow due to R's interpreter overhead per iteration.

---

## Optimization Strategy

**Core idea: Separate topology from attributes. Build the adjacency table once (344K cells), then join yearly attributes and compute stats vectorized using `data.table`.**

### Step-by-step:

1. **Build a static edge table** from `rook_neighbors_unique` (the `nb` object): a two-column `data.table` with columns `(id, neighbor_id)`. This has ~1.37M rows and is built **once**.

2. **For each variable**, join the cell-year attribute values onto the edge table by `(neighbor_id, year)`, then group by `(id, year)` and compute `max`, `min`, `mean` — all vectorized in `data.table`.

3. **Join the resulting neighbor stats back** onto the main `cell_data` table.

This eliminates:
- The 6.46M-element list construction entirely.
- All `paste()`-based key construction.
- All R-level `lapply` loops over millions of elements.

**Expected speedup**: From ~86 hours to **minutes** (roughly 2–10 minutes depending on disk I/O and RAM pressure).

**Preservation guarantees**:
- The trained Random Forest model is untouched — we only modify feature engineering.
- The numerical output is identical: same max, min, mean of the same neighbor values.

---

## Working R Code

```r
library(data.table)

# ==============================================================
# STEP 1: Build the static edge table ONCE from the nb object
# ==============================================================
# rook_neighbors_unique is a spdep::nb object (list of length 344,208).
# id_order is the vector mapping list index -> cell id.

build_edge_table <- function(id_order, neighbors_nb) {
  # Pre-allocate: count total edges
  n_edges <- sum(vapply(neighbors_nb, length, integer(1)))
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors_nb)) {
    nb_idx <- neighbors_nb[[i]]
    # spdep::nb encodes "no neighbors" as a single 0L; skip those
    if (length(nb_idx) == 1L && nb_idx == 0L) next
    n_nb <- length(nb_idx)
    from_id[pos:(pos + n_nb - 1L)] <- id_order[i]
    to_id[pos:(pos + n_nb - 1L)]   <- id_order[nb_idx]
    pos <- pos + n_nb
  }
  
  # Trim if any 0-neighbor cells caused over-allocation
  if (pos <= n_edges) {
    from_id <- from_id[1:(pos - 1L)]
    to_id   <- to_id[1:(pos - 1L)]
  }
  
  data.table(id = from_id, neighbor_id = to_id)
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

cat(sprintf("Edge table: %d directed edges\n", nrow(edge_dt)))
# Expected: ~1,373,394 rows

# ==============================================================
# STEP 2: Convert cell_data to data.table (if not already)
# ==============================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Set key for fast joins
setkey(cell_data, id, year)

# ==============================================================
# STEP 3: For each variable, compute neighbor stats via join
# ==============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_neighbor_features_dt <- function(cell_data, edge_dt, var_name) {
  # Columns we need from cell_data for the join: id, year, and the variable
  # We join edge_dt with cell_data on (neighbor_id = id, year) to get
  # each neighbor's value, then aggregate by (id, year).
  
  # Subset to only needed columns for memory efficiency
  val_dt <- cell_data[, .(id, year, val = get(var_name))]
  setkey(val_dt, id, year)
  
  # Add year to edge table by cross-joining with years
  # More efficient: join edge_dt onto cell_data to get (id, year, neighbor_id),
  # then join again to get neighbor values.
  
  # Step A: Get all (id, year) combinations that exist, paired with neighbor_id
  # This is: for each row in cell_data, look up its neighbors from edge_dt
  
  # Create (id, year) from cell_data, join with edge_dt on id
  id_year_dt <- cell_data[, .(id, year)]
  setkey(id_year_dt, id)
  setkey(edge_dt, id)
  
  # Join: for each (id, year), get all neighbor_ids
  # This produces ~1.37M * 28 ≈ 38.5M rows (but many cells don't have all 28 years)
  # Actually: each cell-year row gets its neighbor list, so total rows =
  # sum over all cell-years of (number of neighbors of that cell)
  # ≈ 6.46M rows * avg ~4 neighbors = ~25.8M rows. Fits in 16GB RAM.
  
  expanded <- edge_dt[id_year_dt, .(id, year, neighbor_id), 
                      on = "id", allow.cartesian = TRUE, nomatch = NULL]
  
  # Step B: Join neighbor values: look up val for (neighbor_id, year)
  setkey(expanded, neighbor_id, year)
  expanded[val_dt, neighbor_val := i.val, on = .(neighbor_id = id, year)]
  
  # Step C: Aggregate by (id, year), dropping NAs in neighbor_val
  stats <- expanded[!is.na(neighbor_val), 
                    .(nb_max  = max(neighbor_val),
                      nb_min  = min(neighbor_val),
                      nb_mean = mean(neighbor_val)),
                    by = .(id, year)]
  
  # Rename columns to match original naming convention
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
  
  # Step D: Join stats back onto cell_data
  # First, remove old columns if they exist (idempotency)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }
  
  setkey(stats, id, year)
  setkey(cell_data, id, year)
  cell_data[stats, (c(max_col, min_col, mean_col)) := 
              mget(paste0("i.", c(max_col, min_col, mean_col))),
            on = .(id, year)]
  
  # Rows with no valid neighbors remain NA (correct behavior, matches original)
  invisible(cell_data)
}

# ==============================================================
# STEP 4: Run for all neighbor source variables
# ==============================================================
for (var_name in neighbor_source_vars) {
  cat(sprintf("Computing neighbor features for: %s ...\n", var_name))
  t0 <- proc.time()
  cell_data <- compute_neighbor_features_dt(cell_data, edge_dt, var_name)
  elapsed <- (proc.time() - t0)["elapsed"]
  cat(sprintf("  Done in %.1f seconds.\n", elapsed))
}

# ==============================================================
# STEP 5: Predict with the existing trained Random Forest
# ==============================================================
# The trained model object (e.g., `rf_model`) is unchanged.
# Convert back to data.frame if the model's predict method requires it.

# cell_data_df <- as.data.frame(cell_data)  # only if needed
# predictions <- predict(rf_model, newdata = cell_data_df)

cat("Neighbor feature computation complete. Ready for RF prediction.\n")
```

---

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Topology computation** | Rebuilt for every cell-year (6.46M iterations) | Built once as edge table (344K cells, ~1.37M edges) |
| **Key lookup mechanism** | Named character vector hashing (`paste` + `[`) | `data.table` binary-search joins (keyed integer joins) |
| **Stats computation** | R-level `lapply` over 6.46M elements | Vectorized `data.table` grouped aggregation |
| **Memory pattern** | 6.46M-element list of integer vectors (fragmented) | Contiguous columnar `data.table` (cache-friendly) |
| **Estimated runtime** | ~86+ hours | **~2–10 minutes** |
| **Trained RF model** | ✅ Preserved (untouched) | ✅ Preserved (untouched) |
| **Numerical results** | Baseline | **Identical** (same max/min/mean over same neighbor sets) |

The key conceptual fix is **separating the static spatial topology from the time-varying attributes**, building the adjacency structure once, and leveraging `data.table`'s optimized join and group-by machinery instead of R-level loops over millions of list elements.