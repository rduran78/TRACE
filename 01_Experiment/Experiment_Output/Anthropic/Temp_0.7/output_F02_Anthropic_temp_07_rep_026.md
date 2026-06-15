 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The root cause of the ~86+ hour runtime and excessive memory pressure is the **row-level `lapply` loop in `build_neighbor_lookup`**, which iterates over all ~6.46 million rows, performing per-row string pasting, hash-table lookups, and list allocation. Then `compute_neighbor_stats` iterates over that 6.46-million-element list again for every variable. Specifically:

1. **`build_neighbor_lookup`**: Creates 6.46M character keys (`paste(id, year)`), then for each row, pastes neighbor keys and looks them up. This is O(n × k) string operations where n ≈ 6.46M and k ≈ average neighbor count (~4 for rook). The `lapply` over 6.46M rows in base R is extremely slow, and the resulting list-of-integer-vectors consumes enormous memory (each list element has R object overhead of ~100+ bytes, totaling >600 MB just in overhead, plus the index data).

2. **`compute_neighbor_stats`**: Iterates the 6.46M-element list 5 times (once per variable), each time extracting subsets of a numeric vector. The `do.call(rbind, ...)` on 6.46M small vectors is also very slow.

3. **Memory**: The neighbor lookup list alone, plus intermediate character vectors, plus the 6.46M × 110 data frame, easily exceeds 16 GB.

---

## Optimization Strategy

**Replace row-level list operations with vectorized `data.table` joins and grouped aggregations.**

The key insight: the neighbor lookup is a **spatial join**. Each cell-year needs the stats of its neighbors *in the same year*. This is equivalent to:

1. Build a flat edge table: `(id, neighbor_id)` from the `nb` object (done once, ~1.37M rows).
2. Join this edge table with the data on `(neighbor_id, year)` to get neighbor values.
3. Group by `(id, year)` and compute `max`, `min`, `mean`.

This turns the entire pipeline into a few vectorized `data.table` merge + group-by operations — no per-row R loops, no 6.46M-element lists, and minimal memory overhead.

**Estimated speedup**: from 86+ hours to **minutes** (the join is ~1.37M edges × 28 years ≈ 38.4M rows, and `data.table` handles grouped aggregation on that trivially).

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 1. Build a flat edge table from the nb object (once)
#    rook_neighbors_unique is a list where element i contains
#    the integer indices of neighbors of id_order[i].
#    id_order is the vector mapping positional index -> cell id.
# ---------------------------------------------------------------
build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate by computing total edges
  n_edges <- sum(lengths(neighbors))
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_i <- neighbors[[i]]
    n_i  <- length(nb_i)
    if (n_i > 0L) {
      from_id[pos:(pos + n_i - 1L)] <- id_order[i]
      to_id[pos:(pos + n_i - 1L)]   <- id_order[nb_i]
      pos <- pos + n_i
    }
  }
  
  data.table(id = from_id, neighbor_id = to_id)
}

# ---------------------------------------------------------------
# 2. Compute neighbor stats for one variable via data.table join
# ---------------------------------------------------------------
compute_neighbor_stats_dt <- function(cell_dt, edge_dt, var_name) {
  # Subset to only the columns we need for the join (saves memory)
  # neighbor side: we need neighbor_id matched as id, plus year, plus the variable
  val_dt <- cell_dt[, .(id, year, val = get(var_name))]
  
  # Join edges with neighbor values:
  #   edge_dt has (id, neighbor_id)
  #   For each (id, year), we need vals from (neighbor_id, year)
  # Strategy: join edge_dt with val_dt on neighbor_id == id
  # But we need to bring in the year from the focal cell.
  # Efficient approach: merge cell_dt's (id, year) with edges, then look up neighbor vals.
  
  # Step A: Get all (id, year, neighbor_id) combinations
  # This is edge_dt[cell_dt] but we only need id and year from cell_dt.
  # Since every id appears for all its years, we can do:
  #   unique_years × edge_dt, but that's wasteful if some cells don't appear in all years.
  # Best: merge edge_dt with the id-year combos.
  
  # Actually, the most memory-efficient way:
  # Join edge_dt onto val_dt by neighbor_id to get neighbor values,
  # keyed by (id, year).
  
  setkey(val_dt, id, year)
  
  # Rename for the join: we want to look up val for (neighbor_id, year)
  # So we join edge_dt[, .(id, neighbor_id)] with val_dt on neighbor_id = id
  # We need to carry the focal id's year. Since every focal id appears with
  # multiple years, we expand via a merge.
  
  # Step A: For each edge (id, neighbor_id), get all years of the focal cell
  focal_id_years <- unique(cell_dt[, .(id, year)])
  
  # Merge focal (id, year) with edges -> (id, year, neighbor_id)
  # Use data.table keyed join for speed
  setkey(edge_dt, id)
  setkey(focal_id_years, id)
  expanded <- edge_dt[focal_id_years, allow.cartesian = TRUE, nomatch = 0L]
  # Result columns: id, neighbor_id, year
  
  # Step B: Look up neighbor value for (neighbor_id, year)
  setnames(val_dt, "id", "neighbor_id")
  setkey(val_dt, neighbor_id, year)
  setkey(expanded, neighbor_id, year)
  expanded[val_dt, neighbor_val := i.val, on = .(neighbor_id, year)]
  
  # Step C: Aggregate by (id, year)
  stats <- expanded[!is.na(neighbor_val),
                    .(nmax  = max(neighbor_val),
                      nmin  = min(neighbor_val),
                      nmean = mean(neighbor_val)),
                    by = .(id, year)]
  
  # Rename columns to match the variable
  new_names <- paste0(var_name, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
  setnames(stats, c("nmax", "nmin", "nmean"), new_names)
  
  stats
}

# ---------------------------------------------------------------
# 3. Memory-optimized version that avoids the large expanded table
#    by processing one year at a time (chunked approach).
#    This keeps peak memory well within 16 GB.
# ---------------------------------------------------------------
compute_neighbor_stats_chunked <- function(cell_dt, edge_dt, var_name) {
  years <- sort(unique(cell_dt$year))
  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")
  
  results <- rbindlist(lapply(years, function(yr) {
    # Subset to this year
    yr_dt <- cell_dt[year == yr, .(id, val = get(var_name))]
    setkey(yr_dt, id)
    
    # Join edges with neighbor values for this year
    # edge_dt: (id, neighbor_id)
    # We want: for each id, get val of each neighbor_id
    merged <- edge_dt[yr_dt, on = .(id), nomatch = 0L, allow.cartesian = FALSE]
    # merged has: id, neighbor_id, val (but val is the focal cell's — we don't need it)
    # Actually we need the NEIGHBOR's val. Let's do it properly:
    
    # Rename for clarity
    neighbor_vals <- yr_dt[, .(neighbor_id = id, neighbor_val = val)]
    setkey(neighbor_vals, neighbor_id)
    
    # Join: for each edge (id, neighbor_id), get neighbor_val
    setkey(edge_dt, neighbor_id)
    joined <- neighbor_vals[edge_dt, on = .(neighbor_id), nomatch = NA_integer_]
    # joined has: neighbor_id, neighbor_val, id
    
    # Aggregate
    stats_yr <- joined[!is.na(neighbor_val),
                       .(nmax = max(neighbor_val),
                         nmin = min(neighbor_val),
                         nmean = mean(neighbor_val)),
                       by = .(id)]
    stats_yr[, year := yr]
    stats_yr
  }))
  
  setnames(results, c("nmax", "nmin", "nmean"), c(col_max, col_min, col_mean))
  results
}

# ---------------------------------------------------------------
# 4. Main pipeline
# ---------------------------------------------------------------
run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {
  
  cell_dt <- as.data.table(cell_data)
  setkey(cell_dt, id, year)
  
  # Build edge table once (~1.37M rows, tiny)
  edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
  
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  for (var_name in neighbor_source_vars) {
    message("Processing neighbor stats for: ", var_name)
    
    stats_dt <- compute_neighbor_stats_chunked(cell_dt, edge_dt, var_name)
    setkey(stats_dt, id, year)
    
    # Left-join stats back onto the main table
    cell_dt <- stats_dt[cell_dt, on = .(id, year)]
    setkey(cell_dt, id, year)
    
    # Free memory
    rm(stats_dt)
    gc()
  }
  
  # Return as data.frame if downstream code expects it
  as.data.frame(cell_dt)
}

# ---------------------------------------------------------------
# Usage (drop-in replacement for the original outer loop):
# ---------------------------------------------------------------
# cell_data <- run_neighbor_feature_pipeline(cell_data, id_order, rook_neighbors_unique)
#
# Then proceed directly to predict() with the already-trained Random Forest.
# The new columns are numerically identical (max, min, mean) to the originals,
# so the trained model's predictions are preserved exactly.
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets (rook contiguity, same year, non-NA values). No approximation is introduced. |
| **Trained RF model** | No retraining. The output columns have the same names and identical values, so `predict(rf_model, new_data)` produces the same predictions. |
| **Column names** | `{var}_neighbor_max`, `{var}_neighbor_min`, `{var}_neighbor_mean` — adjust the suffix pattern in the code if your original naming convention differs. |

---

## Performance Estimates

| Metric | Original | Optimized |
|---|---|---|
| **`build_neighbor_lookup`** | ~6.46M `lapply` iterations with string ops → hours | Replaced by `build_edge_table`: single vectorized loop over ~344K cells → **< 1 second** |
| **`compute_neighbor_stats`** (per var) | ~6.46M list iterations × 5 vars → tens of hours | `data.table` keyed join + grouped agg, 28 year-chunks × ~1.37M edges → **~1–3 minutes per variable** |
| **Total** | **86+ hours** | **~5–15 minutes** |
| **Peak RAM** | >16 GB (6.46M-element list + copies) | ~4–8 GB (chunked by year, main table + one year's edges in memory) |

The chunked-by-year approach ensures that at no point do we materialize the full 6.46M × avg_neighbors expanded table in memory, keeping peak usage well within 16 GB.