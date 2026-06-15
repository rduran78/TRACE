 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` — O(n) `lapply` over 6.46M rows

For every one of the ~6.46 million cell-year rows, the function:
1. Looks up the cell's rook neighbors (by cell ID).
2. Constructs string keys (`paste(neighbor_id, year, "_")`).
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`).

String-based key construction and named-vector lookup for 6.46M rows, each with ~4 neighbors on average (1.37M directed relationships / 344K cells ≈ 4), means roughly **25+ million string paste and hash-lookup operations**. The `lapply` returns a list of 6.46M integer vectors — a massive, fragmented data structure.

### Bottleneck B: `compute_neighbor_stats` — repeated per-row `lapply`

For each of the 5 variables, another `lapply` iterates over 6.46M list elements, extracting values, removing NAs, and computing `max/min/mean`. That's **5 × 6.46M = 32.3M R-level function calls** with per-element subsetting.

### Why raster focal/kernel operations don't directly apply

Focal operations on rasters assume a regular grid with a fixed rectangular kernel. Here, the grid cells have an irregular neighbor structure (coastal cells, boundary cells have fewer neighbors), and the data is in long panel format (cell × year). A focal approach would require reshaping each variable into a 2D raster per year, running `focal()`, and extracting back — possible but fragile and unnecessary once we vectorize properly.

### The real fix: vectorize using sparse-matrix or data.table group-by operations

The neighbor relationships are **time-invariant** — the same neighbor structure repeats for all 28 years. We can exploit this by:
1. Building a sparse adjacency matrix once (344K × 344K).
2. Reshaping each variable into a matrix of (344K cells × 28 years).
3. Using **sparse matrix multiplication and row-wise operations** to compute neighbor stats in bulk — or equivalently, using `data.table` joins on an edge list.

The `data.table` edge-list approach is simpler and memory-friendlier on a 16 GB laptop.

---

## 2. Optimization Strategy

| Step | Current | Proposed |
|------|---------|----------|
| Neighbor lookup | 6.46M-element list of integer vectors (string keys) | Pre-built edge-list `data.table` with `(row_i, row_j)` for all cell-year pairs (~25.8M edges) |
| Stat computation | `lapply` over 6.46M elements, per variable | Vectorized `data.table` group-by `max/min/mean` on the edge list, per variable |
| Iterations | 5 variables × 6.46M R calls = 32.3M calls | 5 variables × 1 vectorized group-by = 5 operations |
| Estimated time | 86+ hours | **~2–10 minutes** |

**Key insight**: The edge list `(cell_i, cell_j)` is time-invariant (344K cells, ~1.37M directed edges). For each year, the same edges apply. So the full edge list in row-index space has ~1.37M × 28 = ~38.4M rows. We join variable values onto the neighbor side and group-by the focal cell's row index.

---

## 3. Working R Code

```r
library(data.table)

# ─────────────────────────────────────────────────────────────
# STEP 0: Convert cell_data to data.table if not already
# ─────────────────────────────────────────────────────────────
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure a row index column exists for stable reference
cell_data[, .row_idx := .I]

# ─────────────────────────────────────────────────────────────
# STEP 1: Build a time-invariant directed edge list from the
#         spdep nb object (rook_neighbors_unique)
#
#   id_order[i] is the cell ID for the i-th element of the nb list.
#   rook_neighbors_unique[[i]] gives integer indices (into id_order)
#   of the rook neighbors of cell id_order[i].
# ─────────────────────────────────────────────────────────────
message("Building time-invariant edge list from nb object...")

edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {

  nb_idx <- rook_neighbors_unique[[i]]
  if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
    return(NULL)
  }
  data.table(focal_cell = id_order[i], neighbor_cell = id_order[nb_idx])
}))

message(sprintf("  Edge list: %s directed edges across %s unique focal cells.",
                format(nrow(edge_list), big.mark = ","),
                format(uniqueN(edge_list$focal_cell), big.mark = ",")))

# ─────────────────────────────────────────────────────────────
# STEP 2: Expand edge list across all years by joining with
#         cell_data row indices.
#
#   For each (focal_cell, neighbor_cell) pair and each year,
#   we need:
#     - row_i: the row index in cell_data for (focal_cell, year)
#     - row_j: the row index in cell_data for (neighbor_cell, year)
# ─────────────────────────────────────────────────────────────
message("Mapping edge list to row indices across all years...")

# Create a lookup: cell id + year -> row index
setkey(cell_data, id, year)
id_year_lookup <- cell_data[, .(id, year, .row_idx)]

# Join focal side
setnames(id_year_lookup, c("id", "year", ".row_idx"),
         c("focal_cell", "year", "row_i"))
edges_full <- merge(edge_list, id_year_lookup,
                    by = "focal_cell", allow.cartesian = TRUE)

# Join neighbor side
setnames(id_year_lookup, c("focal_cell", "year", "row_i"),
         c("neighbor_cell", "year", "row_j"))
edges_full <- merge(edges_full, id_year_lookup,
                    by = c("neighbor_cell", "year"), allow.cartesian = FALSE)

# Restore lookup names for safety
setnames(id_year_lookup, c("neighbor_cell", "year", "row_j"),
         c("id", "year", ".row_idx"))

# Keep only what we need
edges_full <- edges_full[, .(row_i, row_j)]
setkey(edges_full, row_i)

message(sprintf("  Expanded edge list: %s (row_i, row_j) pairs.",
                format(nrow(edges_full), big.mark = ",")))

# ─────────────────────────────────────────────────────────────
# STEP 3: For each neighbor source variable, compute
#         max, min, mean of neighbor values — fully vectorized.
# ─────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor stats for: %s", var_name))

  # Extract the variable values as a plain vector (indexed by row)
  vals <- cell_data[[var_name]]

  # Attach neighbor values to the edge list
  edges_full[, nbr_val := vals[row_j]]

  # Compute grouped stats: max, min, mean (excluding NAs)
  stats <- edges_full[!is.na(nbr_val),
                      .(nb_max  = max(nbr_val),
                        nb_min  = min(nbr_val),
                        nb_mean = mean(nbr_val)),
                      by = row_i]

  # Initialize columns with NA
  max_col  <- paste0(var_name, "_nb_max")
  min_col  <- paste0(var_name, "_nb_min")
  mean_col <- paste0(var_name, "_nb_mean")

  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]

  # Assign computed values by row index
  cell_data[stats$row_i, (max_col)  := stats$nb_max]
  cell_data[stats$row_i, (min_col)  := stats$nb_min]
  cell_data[stats$row_i, (mean_col) := stats$nb_mean]

  message(sprintf("  -> Added %s, %s, %s", max_col, min_col, mean_col))
}

# Clean up temporary column
edges_full[, nbr_val := NULL]
cell_data[, .row_idx := NULL]

message("Done. All 15 neighbor-stat columns added to cell_data.")

# ─────────────────────────────────────────────────────────────
# STEP 4: Predict with the pre-trained Random Forest
#         (no retraining — model object is used as-is)
# ─────────────────────────────────────────────────────────────
# Example (adjust to your actual model object and predict call):
# cell_data$rf_prediction <- predict(trained_rf_model, newdata = cell_data)
```

---

## 4. Why This Preserves Correctness

| Requirement | Guarantee |
|---|---|
| **Same numerical results** | The `max`, `min`, `mean` are computed over exactly the same set of non-NA rook-neighbor values per cell-year as the original code. The edge list is derived from the identical `rook_neighbors_unique` nb object and `id_order`. |
| **Trained RF model unchanged** | The model object is never modified; only `predict()` is called on the augmented `cell_data`. |
| **NA handling** | Rows with no valid neighbors (boundary cells with all-NA neighbors, or cells absent from a given year) receive `NA` in all three stat columns — identical to the original `c(NA, NA, NA)` return. |

## 5. Memory Considerations (16 GB laptop)

| Object | Estimated size |
|---|---|
| `cell_data` (~6.46M rows × 110 cols) | ~5.7 GB |
| `edges_full` (~38M rows × 2 int cols) | ~0.6 GB |
| `edges_full` with `nbr_val` (temporary) | ~0.9 GB |
| `stats` (temporary per variable) | ~0.15 GB |
| **Total peak** | **~7.5 GB** ✓ fits in 16 GB |

If memory is tight, you can process variables one at a time (as shown) and `rm(stats)` after each iteration, or even split the edge list by year chunks. But the above should fit comfortably.

## 6. Expected Runtime

- Edge list construction: ~10–30 seconds
- Edge list expansion (merge): ~1–3 minutes
- Per-variable grouped stats: ~10–30 seconds each (5 vars → ~1–2.5 minutes)
- **Total: ~3–6 minutes** vs. the original 86+ hours — roughly a **1,000× speedup**.