 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` — O(n) `lapply` over 6.46M rows

For every single cell-year row, the function:
1. Looks up the spatial cell's rook neighbors (fine — dictionary lookup).
2. Constructs `paste(neighbor_id, year)` key strings for each neighbor.
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`) into a 6.46M-element named character vector.

Named vector lookup in R is **hash-based but slow at scale** because `paste` allocates millions of small character vectors and the named lookup has per-call overhead. With ~6.46M rows × ~4 neighbors each ≈ 25.8M string constructions and lookups, this alone can take many hours.

### Bottleneck B: `compute_neighbor_stats` — `lapply` over 6.46M rows, repeated 5×

Each call extracts neighbor values via index subsetting and computes `max/min/mean`. The `lapply` loop is called 5 times (once per variable). The `do.call(rbind, result)` on a 6.46M-element list of 3-vectors is also expensive.

### Why raster focal/kernel operations don't directly apply

Focal operations assume a regular complete grid with a fixed rectangular kernel. The panel has:
- Irregular boundaries (not all cells present every year, NA handling).
- Rook contiguity neighbors that may not map to a simple 3×3 kernel if the grid has missing cells or irregular shape.

However, the **analogy is useful**: focal operations are fast because they operate column-wise on matrices. We can replicate this by converting the neighbor structure into a **sparse adjacency matrix** and using **sparse matrix–dense matrix multiplication** to compute neighbor sums and counts, then derive max/min/mean.

**Caveat for max and min**: Matrix multiplication gives sums, not max/min. For max/min we need a different approach. We can use a **data.table join** strategy that is far faster than the per-row `lapply`.

---

## 2. Optimization Strategy

### Step 1: Replace `build_neighbor_lookup` entirely
Instead of building a per-row lookup list of 6.46M elements, build an **edge table** (data.table) of `(row_i, row_j)` pairs — i.e., for each cell-year row `i`, list all row indices `j` that are its rook neighbors in the same year. This edge table has ~25.8M rows (6.46M rows × ~4 neighbors), which is very manageable.

Construction: use `data.table` keyed joins — merge the spatial neighbor pairs with year to get row-index pairs. This replaces millions of `paste` + named-vector lookups with a single vectorized join.

### Step 2: Replace `compute_neighbor_stats` with vectorized group-by
Using the edge table, for each variable:
- Join the variable's values onto the edge table by `row_j`.
- Group by `row_i` and compute `max`, `min`, `mean` in one `data.table` aggregation.

This replaces 6.46M R-level loop iterations with a single vectorized `data.table` grouped aggregation over ~25.8M rows — typically seconds, not hours.

### Step 3: Compute all 5 variables in one pass (or 5 fast passes)

### Expected speedup
- From ~86+ hours → **~2–10 minutes** on a 16 GB laptop.

---

## 3. Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 0: Ensure cell_data is a data.table with a row-index column
# ──────────────────────────────────────────────────────────────────────
cell_data <- as.data.table(cell_data)
cell_data[, row_idx := .I]

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build the edge table (replaces build_neighbor_lookup)
#
# Inputs:
#   id_order             — vector of spatial cell IDs in the order used by
#                          the nb object (length = 344,208)
#   rook_neighbors_unique — spdep::nb object (list of length 344,208,
#                           each element = integer vector of neighbor
#                           positions in id_order)
#   cell_data            — data.table with columns: id, year, row_idx,
#                          plus all predictor columns
# ──────────────────────────────────────────────────────────────────────

build_edge_table <- function(cell_data, id_order, neighbors) {
  # --- 1a. Build spatial edge list (cell-id to cell-id) ---------------
  n_cells <- length(id_order)
  from_list <- vector("list", n_cells)
  to_list   <- vector("list", n_cells)
  
  for (k in seq_len(n_cells)) {
    nb_idx <- neighbors[[k]]
    if (length(nb_idx) == 0L) next
    from_list[[k]] <- rep(id_order[k], length(nb_idx))
    to_list[[k]]   <- id_order[nb_idx]
  }
  
  spatial_edges <- data.table(
    id_from = unlist(from_list, use.names = FALSE),
    id_to   = unlist(to_list,   use.names = FALSE)
  )
  # spatial_edges has ~1,373,394 rows (directed rook pairs)
  
  # --- 1b. Join with cell_data to get (row_i, row_j) for same year ---
  # Map: for each cell-year row, find its row_idx
  id_year_idx <- cell_data[, .(id, year, row_idx)]
  
  # Join "from" side: get row_idx of the focal cell and its year
  setkey(id_year_idx, id)
  # We need to cross spatial_edges with years.
  # But a full cross (1.37M edges × 28 years) = 38.4M rows — still fine.
  #
  # More efficient: join via cell_data directly.
  
  # Create a lookup: for each (id, year) → row_idx
  setkey(id_year_idx, id, year)
  
  # For each spatial edge (id_from, id_to), for each year that id_from
  # appears in, find the row_idx of id_from and id_to in that year.
  
  # Get all (id_from, year, row_idx_from)
  from_dt <- id_year_idx[, .(id_from = id, year, row_i = row_idx)]
  
  # Merge with spatial edges to get id_to for each (id_from, year)
  setkey(spatial_edges, id_from)
  setkey(from_dt, id_from)
  
  # This is the key join: for each (id_from, year) expand by all
  # neighbors of id_from
  edge_year <- spatial_edges[from_dt, on = "id_from",
                              allow.cartesian = TRUE,
                              nomatch = NULL]
  # edge_year has columns: id_from, id_to, year, row_i
  # ~1.37M × 28 ≈ 38.5M rows (but many id_from may not appear all years;
  # actual count ≈ 25.8M based on problem statement)
  
  # Now find row_idx for the neighbor (id_to, year)
  setnames(id_year_idx, c("id_to", "year", "row_j"))
  setkey(id_year_idx, id_to, year)
  setkey(edge_year, id_to, year)
  
  edge_year <- id_year_idx[edge_year, on = c("id_to", "year"),
                            nomatch = NA]
  # Keep only edges where the neighbor actually exists in that year
  edge_year <- edge_year[!is.na(row_j)]
  
  # Return minimal columns
  edge_year[, .(row_i, row_j)]
}

cat("Building edge table...\n")
system.time({
  edge_dt <- build_edge_table(cell_data, id_order, rook_neighbors_unique)
})
# Expected: ~20–60 seconds, ~25.8M rows, two integer columns ≈ 200 MB

setkey(edge_dt, row_i)

# ──────────────────────────────────────────────────────────────────────
# STEP 2: Compute neighbor stats for all 5 variables
#         (replaces compute_neighbor_stats + outer loop)
# ──────────────────────────────────────────────────────────────────────

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
system.time({
  for (var_name in neighbor_source_vars) {
    
    cat("  Processing:", var_name, "\n")
    
    # Extract the variable values indexed by row_idx
    vals <- cell_data[[var_name]]
    
    # Attach neighbor values to the edge table
    edge_dt[, nval := vals[row_j]]
    
    # Remove edges where the neighbor value is NA
    edge_valid <- edge_dt[!is.na(nval)]
    
    # Grouped aggregation: max, min, mean by focal row
    agg <- edge_valid[, .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ), by = row_i]
    
    # Create full-length result columns (NA for rows with no valid neighbors)
    n <- nrow(cell_data)
    col_max  <- rep(NA_real_, n)
    col_min  <- rep(NA_real_, n)
    col_mean <- rep(NA_real_, n)
    
    col_max[agg$row_i]  <- agg$nb_max
    col_min[agg$row_i]  <- agg$nb_min
    col_mean[agg$row_i] <- agg$nb_mean
    
    # Add to cell_data with the same column naming convention
    # (adjust names to match whatever compute_and_add_neighbor_features used)
    set(cell_data, j = paste0(var_name, "_neighbor_max"),  value = col_max)
    set(cell_data, j = paste0(var_name, "_neighbor_min"),  value = col_min)
    set(cell_data, j = paste0(var_name, "_neighbor_mean"), value = col_mean)
  }
  # Clean up temporary column
  edge_dt[, nval := NULL]
})
# Expected: ~1–3 minutes total for all 5 variables

# ──────────────────────────────────────────────────────────────────────
# STEP 3: Remove helper column and proceed to prediction
# ──────────────────────────────────────────────────────────────────────
cell_data[, row_idx := NULL]

# The trained Random Forest model is unchanged.
# Predict as before (example):
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## 4. Why This Preserves the Original Numerical Estimand

| Concern | Guarantee |
|---|---|
| **Same neighbor definition** | We use the identical `rook_neighbors_unique` nb object and `id_order`. The spatial edge list is a direct transcription of the same adjacency structure. |
| **Same year matching** | The `(id_to, year)` join ensures neighbors are matched within the same year, exactly as the original `paste(id, year)` key did. |
| **Same NA handling** | `!is.na(nval)` mirrors the original `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]`. Rows with zero valid neighbors get `NA` for all three stats, matching `return(c(NA, NA, NA))`. |
| **Same aggregation functions** | `max`, `min`, `mean` are identical base-R functions. No approximation. |
| **Random Forest untouched** | The pre-trained model is loaded and used for `predict()` only. No retraining. |

**Result**: Bit-identical outputs (up to floating-point associativity of `mean` in different element orderings, which is negligible and within machine epsilon). The optimization is purely computational — replacing R-level loops and string operations with vectorized `data.table` joins and grouped aggregations.

---

## 5. Summary of Expected Performance

| Stage | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~40–60 hours | ~30–60 seconds |
| Neighbor stats (5 vars × 6.46M rows) | ~26–30 hours | ~1–3 minutes |
| **Total** | **86+ hours** | **~2–5 minutes** |
| Peak RAM | ~6–10 GB (list overhead) | ~3–4 GB (edge table + data.table) |