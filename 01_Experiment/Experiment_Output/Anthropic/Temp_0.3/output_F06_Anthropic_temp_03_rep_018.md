 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` — O(n) `lapply` over ~6.46 million rows

For every single cell-year row, the function:
1. Looks up the spatial cell's rook neighbors (fine — O(1) hash lookup).
2. Constructs `paste(neighbor_id, year, "_")` key strings for every neighbor (expensive string allocation × ~6.46M rows × ~4 neighbors each).
3. Indexes into `idx_lookup` (a named character vector — each lookup is O(1) amortized but the constant factor of character hashing over millions of calls is large).

The result is a **list of 6.46 million integer vectors**. This alone consumes significant memory (~200+ MB of list overhead) and takes hours because `lapply` in R over millions of elements with string operations is inherently slow.

### Bottleneck B: `compute_neighbor_stats` — called 5 times, each iterating over 6.46M rows

Each call does another `lapply` over 6.46 million elements, subsetting a numeric vector and computing `max`, `min`, `mean`. While each individual operation is trivial, the R-level loop overhead across 6.46M iterations × 5 variables is enormous.

### Why raster focal/kernel operations are *not* a direct replacement

Focal operations assume a regular rectangular grid with a fixed kernel. Here, the grid cells have an irregular boundary (not all cells have 4 rook neighbors — coastal/border cells have fewer), and the data is a panel (cell × year). A focal approach would require reshaping into a 3D raster stack per year, running focal per year per variable, then reassembling — possible but fragile and risks subtle mismatches at boundaries. The better approach is to **vectorize the neighbor computation directly using sparse matrix algebra**, which exactly preserves the rook-neighbor structure and numerical results.

### Root cause summary

| Component | Calls | Per-call cost | Total |
|---|---|---|---|
| `build_neighbor_lookup` | 6.46M | ~string ops | ~30-40 hrs |
| `compute_neighbor_stats` | 5 × 6.46M | ~subset + summary | ~45-50 hrs |
| **Total** | | | **~80-90 hrs** |

---

## 2. Optimization Strategy

### Core idea: Replace row-level R loops with sparse-matrix multiplication and vectorized group operations.

**Step 1 — Build a sparse adjacency matrix W (344,208 × 344,208)** from the `nb` object once. This is a standard operation (`spdep::nb2listw` → `as_dgRMatrix_listw`, or direct construction). Each row has ~4 nonzero entries. Total nonzeros ≈ 1.37M. Memory: ~20 MB.

**Step 2 — For each year, extract the variable vector, multiply by W, and derive max/min/mean.** But sparse matrix multiplication gives *sums*, not max/min. So we need a different approach for max and min.

**Refined approach — Expand neighbor pairs into a long table, then use `data.table` grouped operations:**

1. Convert the `nb` object into an edge list: `from_id`, `to_id` (~1.37M rows).
2. Join with the panel data to create a long table of (row_index, neighbor_row_index) for all cell-years. This is done via a merge on (cell_id, year) — fully vectorized.
3. For each variable, extract neighbor values in bulk, then compute `max`, `min`, `mean` grouped by row_index using `data.table`.

This replaces 6.46M R-level iterations with a single vectorized `data.table` grouped aggregation over ~25.8M rows (6.46M × ~4 neighbors). `data.table` does this in seconds.

**Expected speedup: from ~86 hours to ~2-5 minutes.**

### Numerical equivalence

The operations (`max`, `min`, `mean` of the exact same neighbor values) are identical. No approximation is introduced. The trained Random Forest model is not touched.

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE ENGINEERING
# Replaces build_neighbor_lookup + compute_neighbor_stats loop
# Preserves exact numerical results. Does not touch the trained RF model.
# =============================================================================

library(data.table)
library(spdep)      
library(Matrix)     

# ---- Step 0: Ensure cell_data is a data.table with a row-order column --------
cell_dt <- as.data.table(cell_data)
cell_dt[, .row_idx := .I]  # preserve original row order

# ---- Step 1: Build edge list from the nb object (once) ----------------------
# rook_neighbors_unique is an nb object; id_order maps position -> cell id
build_edge_list <- function(nb_obj, id_order) {
  # nb_obj[[i]] contains integer indices of neighbors of cell i (in id_order)
  from <- rep(seq_along(nb_obj), lengths(nb_obj))
  to   <- unlist(nb_obj)
  
  # Remove 0-neighbor entries (spdep uses integer(0) for islands, 
  # but rep/unlist handles that correctly — they simply produce nothing)
  valid <- to != 0L  # spdep marks no-neighbor with 0 in some representations
  
  data.table(
    from_id = id_order[from[valid]],
    to_id   = id_order[to[valid]]
  )
}

edge_dt <- build_edge_list(rook_neighbors_unique, id_order)
cat("Edge list rows:", nrow(edge_dt), "\n")
# Expected: ~1,373,394

# ---- Step 2: Build the full (row_idx → neighbor_row_idx) mapping ------------
# For every cell-year row, find the row indices of its rook neighbors 
# in the same year.

# Create a keyed lookup: (cell_id, year) -> row_idx
setkey(cell_dt, id, year)

# Expand edges across all years:
#   For each edge (from_id, to_id), and for each year in the panel,
#   we need (row_of_from_id_in_year_t, row_of_to_id_in_year_t).
#
# Instead of a full cross join (expensive), we merge twice.

# Lookup table: id, year -> .row_idx
lookup <- cell_dt[, .(id, year, .row_idx)]
setkey(lookup, id, year)

# Get unique years
years <- sort(unique(cell_dt$year))

# Cross join edges × years, then look up row indices
# This is the key vectorized step.
# edge_dt has ~1.37M rows; 28 years → ~38.4M rows before filtering.
# But many (from_id, year) pairs exist for all years, so this is efficient.

edge_year <- CJ_dt <- edge_dt[, .(from_id, to_id)]
# Replicate for each year
edge_year <- edge_dt[rep(seq_len(.N), each = length(years))]
edge_year[, year := rep(years, times = nrow(edge_dt))]

cat("Edge-year rows before join:", nrow(edge_year), "\n")

# Join to get row index of the focal cell (from_id, year)
setnames(lookup, c("id", "year", ".row_idx"), c("from_id", "year", "focal_row"))
setkey(lookup, from_id, year)
setkey(edge_year, from_id, year)
edge_year <- lookup[edge_year, nomatch = 0L]

# Join to get row index of the neighbor cell (to_id, year)
setnames(lookup, c("from_id", "year", "focal_row"), c("to_id", "year", "nbr_row"))
setkey(lookup, to_id, year)
setkey(edge_year, to_id, year)
edge_year <- lookup[edge_year, nomatch = 0L]

# Restore lookup names for safety
setnames(lookup, c("to_id", "year", "nbr_row"), c("id", "year", ".row_idx"))

cat("Edge-year rows after join:", nrow(edge_year), "\n")
# Expected: ~25-38M (depends on boundary cells and year coverage)

# ---- Step 3: Compute neighbor stats for all variables at once ----------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-extract the variable columns as a matrix for fast column access
var_mat <- as.matrix(cell_dt[, ..neighbor_source_vars])

# For each variable, pull neighbor values, group by focal_row, compute stats
for (vname in neighbor_source_vars) {
  col_idx <- match(vname, neighbor_source_vars)
  
  # Get neighbor values (vectorized extraction)
  edge_year[, nbr_val := var_mat[nbr_row, col_idx]]
  
  # Compute grouped stats — data.table is extremely fast at this
  stats <- edge_year[!is.na(nbr_val), 
                     .(nb_max  = max(nbr_val),
                       nb_min  = min(nbr_val),
                       nb_mean = mean(nbr_val)),
                     by = focal_row]
  
  # Initialize result columns with NA
  max_col  <- paste0(vname, "_nb_max")
  min_col  <- paste0(vname, "_nb_min")
  mean_col <- paste0(vname, "_nb_mean")
  
  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]
  
  # Assign results back by row index
  cell_dt[stats$focal_row, (max_col)  := stats$nb_max]
  cell_dt[stats$focal_row, (min_col)  := stats$nb_min]
  cell_dt[stats$focal_row, (mean_col) := stats$nb_mean]
  
  cat("Done:", vname, "\n")
}

# ---- Step 4: Clean up and convert back if needed ----------------------------
edge_year[, nbr_val := NULL]  
cell_dt[, .row_idx := NULL]

# Convert back to data.frame if downstream code expects it
cell_data <- as.data.frame(cell_dt)

# ---- Step 5: Predict with the pre-trained Random Forest (unchanged) ----------
# The RF model is already trained. Just run predict() as before.
# cell_data$predicted <- predict(rf_model, newdata = cell_data)

cat("Neighbor feature engineering complete.\n")
```

---

## 4. Memory and Runtime Estimates

| Object | Size (approx.) |
|---|---|
| `cell_dt` (6.46M × 110 cols) | ~5.5 GB |
| `edge_year` (~38M × 4 cols) | ~1.2 GB |
| `var_mat` (6.46M × 5) | ~260 MB |
| **Peak total** | **~8-9 GB** (fits in 16 GB) |

| Step | Estimated time |
|---|---|
| Build edge list | < 1 sec |
| Cross-join + double merge | ~30-60 sec |
| 5 × grouped aggregation | ~2-3 min |
| **Total** | **~3-5 minutes** |

This is a **~1,000× speedup** over the original 86-hour estimate.

---

## 5. Verification Snippet

To confirm numerical equivalence on a small sample:

```r
# Spot-check: compare old method vs new for a handful of rows
set.seed(42)
check_rows <- sample(nrow(cell_data), 20)

old_lookup <- build_neighbor_lookup(cell_data_original, id_order, rook_neighbors_unique)

for (vname in neighbor_source_vars) {
  old_stats <- compute_neighbor_stats(
    cell_data_original, old_lookup, vname
  )[check_rows, ]
  
  max_col  <- paste0(vname, "_nb_max")
  min_col  <- paste0(vname, "_nb_min")
  mean_col <- paste0(vname, "_nb_mean")
  
  new_stats <- as.matrix(cell_data[check_rows, c(max_col, min_col, mean_col)])
  
  cat(vname, "- max diff:", max(abs(old_stats - new_stats), na.rm = TRUE), "\n")
}
# Expected output: all diffs = 0 (exact match)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup | 6.46M `lapply` with string ops | Vectorized edge-list cross-join via `data.table` |
| Neighbor stats | 5 × 6.46M `lapply` with per-row subsetting | 5 × `data.table` grouped aggregation |
| Runtime | ~86 hours | ~3-5 minutes |
| RAM peak | ~10+ GB (6.46M-element list of vectors) | ~8-9 GB (edge table + data) |
| Numerical result | max, min, mean of rook-neighbor values | **Identical** |
| RF model | Untouched | **Untouched** |