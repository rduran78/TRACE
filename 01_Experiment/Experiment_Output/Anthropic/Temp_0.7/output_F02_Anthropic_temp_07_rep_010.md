 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which runs an `lapply` over ~6.46 million rows, performing per-row string pasting, hash lookups, and NA filtering. This produces a list of 6.46 million integer vectors. Then, `compute_neighbor_stats` iterates over that same 6.46 million-element list once per variable (×5 variables), subsetting and computing summary statistics in pure R loops. Specific problems:

1. **`build_neighbor_lookup` creates ~6.46M character keys** (`paste(id, year)`) and does repeated named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) hashing per call, and doing it millions of times inside `lapply` is extremely slow.
2. **The neighbor lookup list itself is enormous in memory.** 6.46M list elements, each an integer vector, carries heavy R object overhead (~hundreds of bytes per list element even if short), easily consuming 5–10+ GB.
3. **`compute_neighbor_stats` is called 5 times**, each time iterating over the 6.46M-element list in pure R — another ~32M R-level function calls total.
4. **No vectorization or data.table/matrix operations** are used anywhere; everything is scalar/list-based R.

---

## Optimization Strategy

**Replace the per-row list-based approach with a fully vectorized, edge-list / `data.table` group-by approach:**

- Instead of building a 6.46M-element list mapping each row to its neighbors, construct a **long edge table** (`data.table`) where each row is a `(focal_row, neighbor_row)` pair. With ~1.37M directed neighbor relationships × 28 years ≈ ~38.5M rows, this is large but manageable in columnar form (~300 MB for a two-column integer table).
- Join the variable values onto the neighbor side, then **group-by `focal_row`** to compute `max`, `min`, `mean` in one vectorized `data.table` aggregation per variable.
- This eliminates all per-row `lapply`, all character key construction inside loops, and all R-level list overhead.
- Estimated speedup: from 86+ hours to **minutes**.
- Estimated peak RAM: well within 16 GB.

**The trained Random Forest model is never touched. The numerical outputs (max, min, mean of neighbor values) are identical — just computed faster.**

---

## Working R Code

```r
library(data.table)

# ─────────────────────────────────────────────────────────────────────
# Step 1: Build a long edge table (focal_id, neighbor_id) from the nb object
#         This is done ONCE and reused for all variables and all years.
# ─────────────────────────────────────────────────────────────────────

build_edge_dt <- function(id_order, neighbors) {
  # neighbors is an spdep::nb list: neighbors[[i]] gives integer indices

  # into id_order for the neighbors of id_order[i].
  n <- length(neighbors)
  
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(neighbors))
  
  focal_id    <- integer(n_edges)
  neighbor_id <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    nb_i <- neighbors[[i]]
    if (length(nb_i) == 0L || (length(nb_i) == 1L && nb_i[1] == 0L)) next
    len <- length(nb_i)
    idx <- pos:(pos + len - 1L)
    focal_id[idx]    <- id_order[i]
    neighbor_id[idx] <- id_order[nb_i]
    pos <- pos + len
  }
  
  # Trim if any nb entries were empty (0-sentinel)
  if (pos - 1L < n_edges) {
    focal_id    <- focal_id[1:(pos - 1L)]
    neighbor_id <- neighbor_id[1:(pos - 1L)]
  }
  
  data.table(focal_id = focal_id, neighbor_id = neighbor_id)
}

# ─────────────────────────────────────────────────────────────────────
# Step 2: Expand edges across years and map to row indices
# ─────────────────────────────────────────────────────────────────────

build_neighbor_edge_table <- function(cell_data_dt, id_order, neighbors) {
  # cell_data_dt must be a data.table with columns: id, year, and a .ROW_IDX column
  
  # 2a. Spatial edge list (id-level, ~1.37M rows)
  edge_dt <- build_edge_dt(id_order, neighbors)
  
  # 2b. Get the unique years
  years <- sort(unique(cell_data_dt$year))
  
  # 2c. Cross-join edges × years to get (focal_id, neighbor_id, year)
  #     ~1.37M × 28 ≈ 38.5M rows
  year_dt <- data.table(year = years)
  edge_year_dt <- edge_dt[, CJ_idx := 1L][year_dt[, CJ_idx := 1L], 
                                            on = "CJ_idx", 
                                            allow.cartesian = TRUE]
  edge_year_dt[, CJ_idx := NULL]
  
  # 2d. Map focal (id, year) -> row index in cell_data_dt
  id_year_to_row <- cell_data_dt[, .(id, year, .ROW_IDX)]
  
  # Join to get focal row index
  edge_year_dt <- merge(edge_year_dt, id_year_to_row, 
                        by.x = c("focal_id", "year"), 
                        by.y = c("id", "year"), 
                        all.x = FALSE)
  setnames(edge_year_dt, ".ROW_IDX", "focal_row")
  

  # Join to get neighbor row index
  edge_year_dt <- merge(edge_year_dt, id_year_to_row, 
                        by.x = c("neighbor_id", "year"), 
                        by.y = c("id", "year"), 
                        all.x = FALSE)
  setnames(edge_year_dt, ".ROW_IDX", "neighbor_row")
  
  # Keep only what we need
  edge_year_dt[, .(focal_row, neighbor_row)]
}

# ─────────────────────────────────────────────────────────────────────
# Step 3: Compute neighbor stats for one variable using vectorized groupby
# ─────────────────────────────────────────────────────────────────────

compute_neighbor_stats_fast <- function(cell_data_dt, edge_table, var_name) {
  # edge_table has columns: focal_row, neighbor_row
  # Attach the neighbor's value
  vals <- cell_data_dt[[var_name]]
  
  work <- edge_table[, .(focal_row, nval = vals[neighbor_row])]
  
  # Drop NA neighbor values
  work <- work[!is.na(nval)]
  
  # Aggregate
  agg <- work[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = focal_row]
  
  # Initialize output columns with NA
  n <- nrow(cell_data_dt)
  out_max  <- rep(NA_real_, n)
  out_min  <- rep(NA_real_, n)
  out_mean <- rep(NA_real_, n)
  
  out_max[agg$focal_row]  <- agg$nb_max
  out_min[agg$focal_row]  <- agg$nb_min
  out_mean[agg$focal_row] <- agg$nb_mean
  
  # Return as a data.table with proper column names
  prefix <- paste0("neighbor_", var_name)
  result <- data.table(
    v_max  = out_max,
    v_min  = out_min,
    v_mean = out_mean
  )
  setnames(result, c(paste0(prefix, "_max"), 
                      paste0(prefix, "_min"), 
                      paste0(prefix, "_mean")))
  result
}

# ─────────────────────────────────────────────────────────────────────
# Step 4: Main driver — drop-in replacement for the original outer loop
# ─────────────────────────────────────────────────────────────────────

add_all_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                      neighbor_source_vars = c("ntl", "ec", 
                                        "pop_density", "def", "usd_est_n2")) {
  
  cell_data_dt <- as.data.table(cell_data)
  cell_data_dt[, .ROW_IDX := .I]
  
  message("Building edge table (one-time cost)...")
  edge_table <- build_neighbor_edge_table(cell_data_dt, id_order, rook_neighbors_unique)
  message(sprintf("Edge table: %s rows (%.1f MB)", 
                  format(nrow(edge_table), big.mark = ","),
                  object.size(edge_table) / 1e6))
  
  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))
    stats <- compute_neighbor_stats_fast(cell_data_dt, edge_table, var_name)
    
    # Bind columns to the data.table
    cell_data_dt <- cbind(cell_data_dt, stats)
  }
  
  cell_data_dt[, .ROW_IDX := NULL]
  
  # Return as original class (data.frame or data.table)
  if (!is.data.table(cell_data)) {
    return(as.data.frame(cell_data_dt))
  }
  cell_data_dt
}

# ─────────────────────────────────────────────────────────────────────
# Usage (drop-in replacement for the original outer loop):
# ─────────────────────────────────────────────────────────────────────

cell_data <- add_all_neighbor_features(
  cell_data, 
  id_order, 
  rook_neighbors_unique,
  neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
)

# The trained Random Forest model is unchanged.
# Predictions proceed as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Works and What Changes

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M-element R list via `lapply` + character key hashing | One `data.table` edge table built via vectorized merge (~38.5M rows, two integer columns) |
| **Stats computation** | `lapply` over 6.46M elements, per-variable | `data.table` grouped aggregation (`[, .(max, min, mean), by = focal_row]`), per-variable |
| **Memory model** | ~6.46M R list elements (huge overhead) | ~38.5M × 2 integer columns ≈ ~310 MB + working copies ≈ ~2–3 GB peak |
| **Time estimate** | 86+ hours | ~5–15 minutes on a 16 GB laptop |
| **Numerical output** | max, min, mean of non-NA neighbor values | Identical max, min, mean of non-NA neighbor values |
| **RF model** | Untouched | Untouched |

The key insight is that the neighbor structure is **year-invariant** (same spatial grid, same rook neighbors every year), so the ~1.37M spatial edges can be crossed with 28 years once, producing a single reusable edge table. All subsequent per-variable computations are then a single vectorized `data.table` group-by — orders of magnitude faster than 6.46 million R-level function calls.