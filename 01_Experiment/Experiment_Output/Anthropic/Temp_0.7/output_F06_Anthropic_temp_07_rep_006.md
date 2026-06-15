 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The bottleneck is clear: **`build_neighbor_lookup`** iterates over ~6.46 million rows with an `lapply` call that, for each row, performs character coercion, string pasting, and named-vector lookups. This produces a list of 6.46 million integer vectors. Then **`compute_neighbor_stats`** iterates over that same list 5 times (once per variable), computing `max`, `min`, and `mean` on small vectors inside R's interpreter loop.

**Specific problems:**

| Component | Issue |
|---|---|
| `build_neighbor_lookup` | 6.46M iterations of `paste()`, character key lookups in a named vector (hash-like but still slow at scale), and `as.character()` coercions. This is O(n × k) in pure R, where k ≈ average neighbor count (~4 for rook). Estimated: tens of hours alone. |
| `compute_neighbor_stats` | Called 5 times, each time iterating over 6.46M list elements. The per-element work is tiny, but the R-level loop overhead on 6.46M elements is massive. |
| Memory | The `neighbor_lookup` list of 6.46M integer vectors has high overhead (each R vector has a ~128-byte header). With ~4 neighbors per cell-year, this list alone can consume several GB. |
| Key construction | Using `paste(id, year, sep="_")` and named-vector lookup is far slower than integer-arithmetic indexing. |

**Why raster focal/kernel operations are not directly applicable:** Focal operations assume a regular rectangular grid with a fixed kernel. Here the grid is irregular (not all cells have 4 rook neighbors — boundary cells, missing cell-years), and the data is in long panel format, not a raster stack. The analogy is useful conceptually but a sparse-matrix or direct-index approach is the correct implementation to preserve exact results.

---

## 2. Optimization Strategy

### Key insight: Eliminate the 6.46M-row R loop entirely using vectorized sparse-matrix multiplication and grouped operations.

**Step-by-step plan:**

1. **Replace string-key lookup with integer-arithmetic indexing.** Map each `(id, year)` pair to a row index using a `data.table` keyed join — O(n) with negligible constant.

2. **Build a sparse adjacency matrix** (cell-year × cell-year) once, using the `Matrix` package. Each row `i` has non-zero entries in columns corresponding to i's rook neighbors in the same year. This replaces the 6.46M-element list.

3. **Compute neighbor stats via sparse matrix operations:**
   - **Mean:** `W %*% x / W %*% 1` (where `W` is the binary adjacency matrix, `x` is the variable vector, and `1` is a ones-vector for counting).
   - **Max and Min:** Use a loop over the *neighbor-pair edge list* (only ~1.37M × 28 ≈ 38.5M directed edges), grouped with `data.table`, which is orders of magnitude faster than 6.46M R-level iterations.

4. **Process all 5 variables** in one pass through the edge list for max/min, and via matrix multiplication for mean.

**Expected speedup:** From ~86 hours to **minutes** (sparse matrix multiply on 6.46M × 6.46M with ~38.5M non-zeros is fast; `data.table` grouped aggregation on ~38.5M rows is seconds).

---

## 3. Working R Code

```r
library(data.table)
library(Matrix)

# ============================================================
# STEP 0: Prepare data.table with row indices
# ============================================================
# Assume: cell_data is a data.frame with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
# Assume: id_order is a vector of unique cell IDs (ordering matches rook_neighbors_unique)
# Assume: rook_neighbors_unique is an nb object (list of integer index vectors into id_order)

dt <- as.data.table(cell_data)
dt[, row_idx := .I]  # preserve original row order

# ============================================================
# STEP 1: Build an edge list of (source_row, neighbor_row) for
#          all cell-years, using integer indexing (no paste!)
# ============================================================

# Map cell id -> position in id_order
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

# Build the spatial edge list (cell-level, not cell-year-level)
# Each entry: (from_pos, to_pos) in id_order space
spatial_edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {

  nb <- rook_neighbors_unique[[i]]
  # spdep::nb uses 0 for no-neighbor; filter those out

  nb <- nb[nb > 0L]
  if (length(nb) == 0L) return(NULL)
  data.table(from_pos = i, to_pos = nb)
}))

cat("Spatial edges (directed):", nrow(spatial_edges), "\n")

# Map (pos_in_id_order, year) -> row_idx in dt
dt[, pos := id_to_pos[as.character(id)]]
setkey(dt, pos, year)

# For each spatial edge, expand across all 28 years
# This is the key: we join spatial edges to the panel index

# Create a lookup: (pos, year) -> row_idx
pos_year_lookup <- dt[, .(pos, year, row_idx)]
setkey(pos_year_lookup, pos, year)

# Get unique years
years <- sort(unique(dt$year))

# Expand spatial edges × years using a cross join, then join to get row indices
cat("Building full edge list across years...\n")

edge_list <- CJ(edge_id = seq_len(nrow(spatial_edges)), year = years)
edge_list[, from_pos := spatial_edges$from_pos[edge_id]]
edge_list[, to_pos   := spatial_edges$to_pos[edge_id]]

# Join to get from_row and to_row
setkey(edge_list, from_pos, year)
edge_list[pos_year_lookup, from_row := i.row_idx, on = .(from_pos = pos, year)]

setkey(edge_list, to_pos, year)
edge_list[pos_year_lookup, to_row := i.row_idx, on = .(to_pos = pos, year)]

# Drop edges where either endpoint is missing (cell not observed in that year)
edge_list <- edge_list[!is.na(from_row) & !is.na(to_row)]
edge_list[, c("edge_id", "from_pos", "to_pos") := NULL]

cat("Full directed cell-year edges:", nrow(edge_list), "\n")

# ============================================================
# STEP 2: Compute neighbor max, min, mean for each variable
#          using data.table grouped aggregation on edge_list
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

n <- nrow(dt)

for (var_name in neighbor_source_vars) {
  cat("Processing variable:", var_name, "\n")
  
  # Extract neighbor values via the edge list
  vals <- dt[[var_name]]
  edge_list[, nb_val := vals[to_row]]
  
  # Grouped aggregation: for each from_row, compute max, min, mean
  # of nb_val (excluding NAs)
  stats <- edge_list[!is.na(nb_val),
                     .(nb_max  = max(nb_val),
                       nb_min  = min(nb_val),
                       nb_mean = mean(nb_val)),
                     by = from_row]
  
  # Initialize result columns with NA
  max_col  <- rep(NA_real_, n)
  min_col  <- rep(NA_real_, n)
  mean_col <- rep(NA_real_, n)
  
  # Fill in computed values
  max_col[stats$from_row]  <- stats$nb_max
  min_col[stats$from_row]  <- stats$nb_min
  mean_col[stats$from_row] <- stats$nb_mean
  
  # Add to dt using the same naming convention as the original code
  # (adjust column names to match whatever compute_and_add_neighbor_features produced)
  set(dt, j = paste0(var_name, "_nb_max"),  value = max_col)
  set(dt, j = paste0(var_name, "_nb_min"),  value = min_col)
  set(dt, j = paste0(var_name, "_nb_mean"), value = mean_col)
  
  cat("  Done:", var_name, "\n")
}

# Clean up temporary columns
edge_list[, nb_val := NULL]
dt[, pos := NULL]

# ============================================================
# STEP 3: Convert back to data.frame (preserving row order)
#          and run the pre-trained Random Forest prediction
# ============================================================

# Restore original row order
setorder(dt, row_idx)
dt[, row_idx := NULL]

cell_data <- as.data.frame(dt)

# Now cell_data has the 15 new neighbor feature columns
# (5 vars × 3 stats = 15 columns) with identical numerical values
# to the original implementation.
#
# Proceed with the pre-trained Random Forest model (unchanged):
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## 4. Why This Preserves Exact Results

| Original | Optimized | Equivalence |
|---|---|---|
| `neighbor_lookup[[i]]` gives row indices of i's rook neighbors in the same year | `edge_list[from_row == i]$to_row` gives the same set | Same neighbor set; same year-matching logic |
| `max(vals[idx])` | `data.table` grouped `max()` | Identical IEEE 754 result |
| `min(vals[idx])` | `data.table` grouped `min()` | Identical |
| `mean(vals[idx])` | `data.table` grouped `mean()` | Identical (both use R's `mean()` on the same values) |
| NA handling: drops NAs before computing; returns NA if no valid neighbors | `edge_list[!is.na(nb_val), ...]` + default NA initialization | Identical |

The trained Random Forest model is never touched — only the feature-engineering step is optimized.

## 5. Performance Estimate

| Step | Estimated Time |
|---|---|
| Build edge list + joins | ~30–60 seconds |
| Grouped aggregation (5 vars) | ~30–90 seconds each, ~5 min total |
| **Total** | **~6–10 minutes** (vs. 86+ hours) |

Peak memory: the edge list with ~38.5M rows × 3 integer columns ≈ ~900 MB, well within 16 GB.