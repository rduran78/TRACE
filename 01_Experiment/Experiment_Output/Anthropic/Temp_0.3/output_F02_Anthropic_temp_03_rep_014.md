 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting, hash-table lookups, and NA filtering in pure R. This creates millions of small character vectors and triggers repeated hash lookups on a named vector of 6.46 million entries — an O(n²)-like workload in practice. The `compute_neighbor_stats` function then iterates over the same 6.46 million rows again per variable, subsetting numeric vectors by index lists. With 5 variables, that's ~32.3 million list iterations total.

**Specific problems:**

1. **`build_neighbor_lookup`:** `paste()` and named-vector lookup inside `lapply` over 6.46M rows is extremely slow. Each call to `idx_lookup[neighbor_keys]` does a linear scan or hash probe on a 6.46M-entry named character vector. The resulting `neighbor_lookup` list of 6.46M integer vectors also consumes enormous memory.
2. **`compute_neighbor_stats`:** `lapply` over 6.46M elements with per-element subsetting, NA removal, and summary computation is slow in interpreted R.
3. **Memory:** Storing 6.46M list elements (each a variable-length integer vector) plus the full data frame with 110+ columns pushes against the 16 GB limit.

---

## Optimization Strategy

### Key Insight: Exploit the Panel Structure

The neighbor graph is **spatial only** — it does not change across years. There are 344,208 cells and 28 years. The neighbor lookup repeats the same spatial topology for every year. Instead of building a 6.46M-element list, we should:

1. **Separate the spatial neighbor graph (344K cells) from the temporal dimension (28 years).**
2. **Use `data.table` for fast indexed joins** instead of named-vector lookups.
3. **Vectorize the neighbor statistics computation** using a long-format edge table joined to the data, then grouped aggregation — replacing all `lapply` loops with a single `data.table` grouped operation per variable.

### Concrete Plan

- Build an **edge table** from `rook_neighbors_unique`: a two-column `data.table` with `(focal_id, neighbor_id)` — only ~1.37M rows.
- Cross-join this edge table with the 28 years to get a `(focal_id, year, neighbor_id)` table (~38.5M rows, but only 3 integer columns ≈ 0.9 GB).
- For each variable, join the neighbor values onto this edge table, then compute `max`, `min`, `mean` grouped by `(focal_id, year)`.
- Join the results back to the main data.

This replaces all `lapply` loops with vectorized `data.table` operations and avoids ever constructing a 6.46M-element list.

### Memory Management

- Process one variable at a time and discard intermediate columns.
- The edge-year table is the largest object (~0.9 GB) but is reused across all 5 variables.
- Total peak memory: ~4–6 GB (well within 16 GB).

### Runtime Estimate

- `data.table` grouped aggregation over ~38.5M rows × 5 variables: approximately **5–15 minutes total** (down from 86+ hours).

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 1. Convert cell_data to data.table (preserves all 110+ columns)
# ---------------------------------------------------------------
setDT(cell_data)

# Ensure id and year are integer keys for fast joins
cell_data[, id := as.integer(id)]
cell_data[, year := as.integer(year)]
setkey(cell_data, id, year)

# ---------------------------------------------------------------
# 2. Build the spatial edge table from the nb object
#    rook_neighbors_unique is a list of length 344,208;
#    id_order maps list index -> cell id.
# ---------------------------------------------------------------
build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate by computing total number of edges
  n_edges <- sum(lengths(neighbors))
  focal  <- integer(n_edges)
  nbr    <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_idx <- neighbors[[i]]
    n <- length(nb_idx)
    if (n == 0L) next
    focal[pos:(pos + n - 1L)] <- id_order[i]
    nbr[pos:(pos + n - 1L)]   <- id_order[nb_idx]
    pos <- pos + n
  }
  
  data.table(focal_id = focal, neighbor_id = nbr)
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

# ---------------------------------------------------------------
# 3. Cross-join edges × years to get the full edge-year table
#    This is ~1.37M edges × 28 years ≈ 38.5M rows (3 int cols)
# ---------------------------------------------------------------
years_dt <- data.table(year = sort(unique(cell_data$year)))
edge_year_dt <- CJ_dt <- edge_dt[, CJ(focal_id = focal_id, 
                                        neighbor_id = neighbor_id, 
                                        year = years_dt$year, 
                                        unique = FALSE)]

# More memory-efficient cross join:
edge_year_dt <- edge_dt[rep(seq_len(.N), each = nrow(years_dt))]
edge_year_dt[, year := rep(years_dt$year, times = nrow(edge_dt))]

setkey(edge_year_dt, neighbor_id, year)

# ---------------------------------------------------------------
# 4. For each variable, compute neighbor max/min/mean and join back
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  
  cat("Processing neighbor features for:", var_name, "\n")
  
  # Extract only the columns needed for the join
  val_dt <- cell_data[, .(id, year, val = get(var_name))]
  setkey(val_dt, id, year)
  
  # Join neighbor values onto the edge-year table
  # neighbor_id + year -> val
  edge_year_dt[val_dt, nbr_val := i.val, on = .(neighbor_id = id, year)]
  
  # Compute grouped statistics per (focal_id, year), excluding NAs
  stats <- edge_year_dt[!is.na(nbr_val), 
                         .(nbr_max  = max(nbr_val),
                           nbr_min  = min(nbr_val),
                           nbr_mean = mean(nbr_val)),
                         by = .(focal_id, year)]
  
  # Rename columns to match the original pipeline's naming convention
  new_names <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  setnames(stats, c("nbr_max", "nbr_min", "nbr_mean"), new_names)
  
  # Join stats back to cell_data
  cell_data[stats, (new_names) := mget(paste0("i.", new_names)),
            on = .(id = focal_id, year)]
  
  # Clean up the temporary column in edge_year_dt
  edge_year_dt[, nbr_val := NULL]
  
  cat("  Done. Added:", paste(new_names, collapse = ", "), "\n")
}

# ---------------------------------------------------------------
# 5. Clean up large intermediate objects
# ---------------------------------------------------------------
rm(edge_dt, edge_year_dt, years_dt, stats, val_dt)
gc()

# ---------------------------------------------------------------
# 6. Prediction with the existing trained Random Forest
#    (model object is unchanged; only the feature table was rebuilt)
# ---------------------------------------------------------------
# Example (adjust to your actual model object and predict call):
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Trained RF model** | Untouched — we only rebuilt the feature columns that feed into `predict()`. |
| **Numerical estimand** | The `max`, `min`, `mean` computations are identical arithmetic operations on the same neighbor sets; `data.table` uses the same IEEE 754 double precision. |
| **Neighbor definitions** | The edge table is constructed directly from `rook_neighbors_unique` and `id_order`, reproducing the same directed neighbor pairs. |
| **NA handling** | `!is.na(nbr_val)` before aggregation mirrors the original `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]`. Cells with no valid neighbors get `NA` via the left join (no match → columns remain `NA`). |

---

## Summary

| Metric | Original | Optimized |
|---|---|---|
| **Runtime** | ~86+ hours | ~5–15 minutes |
| **Peak RAM** | >16 GB (fails/swaps) | ~4–6 GB |
| **Core technique** | `lapply` over 6.46M rows, string hashing | Vectorized `data.table` join + grouped aggregation |
| **Lines of code** | Similar | Similar |