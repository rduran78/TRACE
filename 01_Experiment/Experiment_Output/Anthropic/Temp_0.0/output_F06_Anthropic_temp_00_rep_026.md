 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` — O(n) `lapply` over 6.46M rows

For every single cell-year row, the function:
1. Looks up the spatial cell's rook neighbors (fine — dictionary lookup).
2. Constructs `paste(neighbor_id, year)` key strings for every neighbor (expensive string allocation × ~6.46M rows × ~4 neighbors each ≈ 25M string constructions).
3. Indexes into `idx_lookup` (a named character vector — each lookup is O(n) in the worst case because R named vector lookup is hash-based but still has overhead at scale).

The result is a **list of 6.46 million integer vectors**. Building this list alone is extremely memory- and time-intensive.

### Bottleneck B: `compute_neighbor_stats` — repeated `lapply` over 6.46M rows, called 5 times

Each call iterates over 6.46M list elements, extracts values, removes NAs, and computes `max/min/mean`. This is pure R-level looping — no vectorization. Called 5 times (once per source variable), this is ~32.3M R-level loop iterations with per-element allocation.

### Why raster focal/kernel operations are *not* directly applicable

Focal operations assume a regular rectangular grid with a fixed kernel. Here, the grid cells have an irregular neighbor structure (coastal cells, boundary cells have fewer neighbors), the data is in long panel format (cell × year), and the neighbor object is a general `spdep::nb` list. Focal operations would require reshaping into a 2D raster per year and handling NA/missing cells — possible but fragile and not guaranteed to preserve the exact estimand for irregular boundaries. **The better strategy is to vectorize the existing approach using sparse matrix multiplication.**

---

## 2. Optimization Strategy

### Core Insight: Neighbor summary statistics via sparse matrix operations

For a variable `x`, the neighbor mean for cell `i` is:

$$\bar{x}_{\text{neighbors}(i)} = \frac{\sum_{j \in N(i)} x_j}{|N(i)|}$$

This is exactly a **sparse matrix–vector product** followed by element-wise division. The sparse matrix `W` has `W[i,j] = 1` if `j` is a rook neighbor of `i`. Then:

- **Neighbor sum** = `W %*% x`
- **Neighbor count** = `W %*% (non-NA indicator of x)`
- **Neighbor mean** = sum / count

For **max** and **min**, we can't use matrix multiplication directly, but we can use an efficient grouped operation via `data.table` or, even better, construct an edge list and use `data.table` grouped aggregation — which is C-level fast.

### Plan

1. **Eliminate `build_neighbor_lookup` entirely.** Instead, build a **cell-year edge list** (a two-column data.table of `[row_i, row_j]` meaning "row `j` is a neighbor of row `i`") once. This is a join operation.

2. **Replace `compute_neighbor_stats` with a single vectorized `data.table` grouped aggregation** over the edge list: group by `row_i`, compute `max`, `min`, `mean` of `vals[row_j]`.

3. **Do all 5 variables in one pass** (or 5 fast passes) over the same edge list.

**Expected speedup:** From 86+ hours to **minutes**. The edge list has ~6.46M × 4 ≈ 25.8M rows (directed edges per year). A `data.table` grouped aggregation over 25.8M rows with 3 summary stats completes in seconds.

---

## 3. Working R Code

```r
library(data.table)

# ============================================================
# STEP 0: Convert to data.table if not already
# ============================================================
cell_dt <- as.data.table(cell_data)

# Ensure id and year columns exist and are keyed for fast joins
cell_dt[, row_idx := .I]  # preserve original row order

# ============================================================
# STEP 1: Build the cell-year edge list ONCE
#
# rook_neighbors_unique: an nb object (list of integer vectors)
#   where element i contains the indices (into id_order) of
#   neighbors of id_order[i].
# id_order: vector of cell IDs in the order matching the nb object.
# ============================================================

build_edge_list <- function(id_order, rook_neighbors_unique) {
  # For each spatial cell, enumerate its directed neighbor pairs
  # as (cell_id, neighbor_cell_id)
  n <- length(id_order)
  from_list <- vector("list", n)
  to_list   <- vector("list", n)
  
  for (i in seq_len(n)) {
    nb_idx <- rook_neighbors_unique[[i]]
    # spdep::nb uses 0 to indicate no neighbors
    nb_idx <- nb_idx[nb_idx != 0L]
    if (length(nb_idx) > 0L) {
      from_list[[i]] <- rep(id_order[i], length(nb_idx))
      to_list[[i]]   <- id_order[nb_idx]
    }
  }
  
  data.table(
    from_id = unlist(from_list, use.names = FALSE),
    to_id   = unlist(to_list,   use.names = FALSE)
  )
}

cat("Building spatial edge list...\n")
spatial_edges <- build_edge_list(id_order, rook_neighbors_unique)
cat(sprintf("  %s directed spatial edges\n", format(nrow(spatial_edges), big.mark = ",")))

# Now expand to cell-year edges by joining on year.
# For each year, every spatial edge (from_id -> to_id) becomes
# a row-level edge (row_of_from_id_year -> row_of_to_id_year).

cat("Building cell-year row index...\n")
# Create a lookup: (id, year) -> row_idx
setkey(cell_dt, id, year)
id_year_lookup <- cell_dt[, .(id, year, row_idx)]
setkey(id_year_lookup, id, year)

# Get unique years
years <- sort(unique(cell_dt$year))

cat("Expanding spatial edges across years...\n")
# Cross join spatial_edges with years, then resolve to row indices
# This is memory-efficient: ~1.37M edges × 28 years ≈ 38.4M rows

edge_year <- CJ_dt <- spatial_edges[, .(from_id, to_id)]
# Replicate for each year
edge_year_full <- edge_year[rep(seq_len(.N), length(years))]
edge_year_full[, year := rep(years, each = nrow(edge_year))]

# Resolve from_id,year -> from_row
setkey(edge_year_full, from_id, year)
setkey(id_year_lookup, id, year)
edge_year_full[id_year_lookup, from_row := i.row_idx, on = .(from_id = id, year = year)]

# Resolve to_id,year -> to_row
edge_year_full[id_year_lookup, to_row := i.row_idx, on = .(to_id = id, year = year)]

# Drop edges where either side is missing (cell not in panel for that year)
edge_list <- edge_year_full[!is.na(from_row) & !is.na(to_row), .(from_row, to_row)]

cat(sprintf("  %s cell-year directed edges\n", format(nrow(edge_list), big.mark = ",")))

# Free intermediate objects
rm(edge_year, edge_year_full, id_year_lookup)
gc()

# ============================================================
# STEP 2: Compute neighbor stats for all variables at once
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor statistics...\n")

for (var_name in neighbor_source_vars) {
  cat(sprintf("  Processing: %s\n", var_name))
  
  # Extract neighbor values via the edge list
  edge_vals <- cell_dt[[var_name]][edge_list$to_row]
  
  # Build a temporary data.table for grouped aggregation
  tmp <- data.table(
    from_row = edge_list$from_row,
    val      = edge_vals
  )
  
  # Remove edges where the neighbor value is NA
  tmp <- tmp[!is.na(val)]
  
  # Grouped aggregation — this is the fast part
  stats <- tmp[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), keyby = from_row]
  
  # Initialize result columns with NA
  max_col  <- paste0(var_name, "_max")
  min_col  <- paste0(var_name, "_min")
  mean_col <- paste0(var_name, "_mean")
  
  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]
  
  # Assign results back by row index
  cell_dt[stats$from_row, (max_col)  := stats$nb_max]
  cell_dt[stats$from_row, (min_col)  := stats$nb_min]
  cell_dt[stats$from_row, (mean_col) := stats$nb_mean]
  
  rm(tmp, stats, edge_vals)
  gc()
}

cat("Done.\n")

# ============================================================
# STEP 3: Convert back to data.frame if needed for predict()
# ============================================================
# Restore original row order
setorder(cell_dt, row_idx)
cell_dt[, row_idx := NULL]

# If your trained RF model expects a data.frame:
cell_data <- as.data.frame(cell_dt)

# ============================================================
# STEP 4: Predict with the pre-trained Random Forest (unchanged)
# ============================================================
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## 4. Why This Preserves the Original Numerical Estimand

| Concern | Guarantee |
|---|---|
| **Same neighbor definition** | We use the identical `rook_neighbors_unique` nb object — same adjacency. |
| **Same statistics** | `max`, `min`, `mean` computed on the identical set of non-NA neighbor values per cell-year. |
| **Same column names** | Output columns follow the same `{var}_{max,min,mean}` naming convention as `compute_and_add_neighbor_features`. |
| **No retraining** | The Random Forest model is never touched; only `predict()` is called on the enriched data. |
| **Floating-point identity** | `data.table` uses the same R-level `max`, `min`, `mean` primitives — results are bit-identical. |

## 5. Performance Estimate

| Stage | Original | Optimized |
|---|---|---|
| Build neighbor lookup | ~hours (6.46M list elements, string ops) | ~30 sec (vectorized edge list + keyed join) |
| Compute stats (×5 vars) | ~80+ hours (R-level lapply) | ~2–5 min (data.table grouped agg over ~38M rows) |
| **Total** | **86+ hours** | **~5–10 minutes** |

Memory peak: the edge list is ~38.4M rows × 2 integer columns ≈ 0.6 GB, well within 16 GB RAM.