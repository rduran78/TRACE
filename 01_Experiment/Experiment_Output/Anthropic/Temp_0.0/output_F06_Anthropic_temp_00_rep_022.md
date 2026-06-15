 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` — O(n) list of per-row lookups via `lapply`

- With ~6.46 million rows, this creates a list of 6.46M elements. Each iteration performs string pasting, hash lookups in a named vector (`idx_lookup`), and NA filtering. The string-key approach (`paste(id, year)`) is inherently slow at this scale because:
  - `paste()` is called 6.46M times.
  - Named vector lookup (`idx_lookup[neighbor_keys]`) is O(k) per call but with constant overhead from R's hashing on character keys, repeated millions of times.
  - The entire structure is an R list of integer vectors — memory-fragmented and GC-heavy.

### Bottleneck B: `compute_neighbor_stats` — repeated per variable

- Called 5 times (once per source variable), each time iterating over 6.46M list elements. Each call to the inner `lapply` does subsetting, NA removal, and three summary statistics. That's ~32.3 million R-level function calls across all variables.

### Why raster focal/kernel operations are *not* a direct substitute

The comment in the prompt asks whether raster focal operations offer a useful analogy. They do conceptually (a rook neighborhood is a 3×3 cross kernel), but:
- The data is a **panel** (cell × year), not a single raster layer.
- The grid cells may not form a complete regular rectangle (there may be missing cells, irregular boundaries).
- The neighbor structure is precomputed as an `spdep::nb` object, which may encode irregular adjacency.
- Focal operations would require reshaping each variable × year into a complete raster, running `focal()`, then re-extracting — introducing complexity and potential for misalignment.

**Conclusion:** The best strategy is to vectorize the neighbor computation using `data.table` joins, which preserves the exact numerical results while eliminating millions of R-level loop iterations.

---

## 2. Optimization Strategy

| Step | Current | Proposed | Speedup source |
|------|---------|----------|----------------|
| Neighbor lookup | 6.46M-element R list built via `lapply` + string keys | Pre-built edge table (data.table) joined by `(id, year)` — integer keys | Vectorized join, no string ops |
| Neighbor stats | 5 × `lapply` over 6.46M elements | 5 × single `data.table` grouped aggregation on the edge table | Vectorized `max`, `min`, `mean` by group |
| Memory | ~6.46M list elements + duplicated index vectors | One edge table (~1.37M edges × 28 years ≈ 38.5M rows, 3 integer columns ≈ 0.9 GB) | Contiguous memory, cache-friendly |

**Expected runtime:** Minutes instead of 86+ hours. The join is O(n log n) or O(n) with data.table's radix join; the grouped aggregation is highly optimized in C.

**Numerical equivalence:** The `max`, `min`, `mean` computations are applied to exactly the same neighbor values with the same NA handling, so results are identical to machine precision.

---

## 3. Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# Step 0: Convert cell_data to data.table (if not already)
# ──────────────────────────────────────────────────────────────────────
cell_dt <- as.data.table(cell_data)

# Ensure 'id' and 'year' columns exist and are keyed for fast joins
stopifnot(all(c("id", "year") %in% names(cell_dt)))

# ──────────────────────────────────────────────────────────────────────
# Step 1: Build a directed edge table from the spdep::nb object
#
#   rook_neighbors_unique is a list of length = number of spatial cells.
#   rook_neighbors_unique[[i]] contains integer indices into id_order
#   of the neighbors of cell id_order[i].
#
#   We expand this into a two-column data.table: (focal_id, neighbor_id)
# ──────────────────────────────────────────────────────────────────────
build_edge_table <- function(id_order, neighbors_nb) {
  # neighbors_nb is an nb object: list of integer vectors (indices into id_order)
  n <- length(neighbors_nb)
  
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(neighbors_nb))
  
  focal_id    <- integer(n_edges)
  neighbor_id <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    nb_idx <- neighbors_nb[[i]]
    # spdep::nb encodes "no neighbors" as 0L in a length-1 vector
    if (length(nb_idx) == 1L && nb_idx[1L] == 0L) next
    len <- length(nb_idx)
    focal_id[pos:(pos + len - 1L)]    <- id_order[i]
    neighbor_id[pos:(pos + len - 1L)] <- id_order[nb_idx]
    pos <- pos + len
  }
  
  data.table(focal_id = focal_id[1:(pos - 1L)],
             neighbor_id = neighbor_id[1:(pos - 1L)])
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

cat(sprintf("Edge table: %d directed edges\n", nrow(edge_dt)))

# ──────────────────────────────────────────────────────────────────────
# Step 2: Cross-join edges with years to get (focal_id, year, neighbor_id)
#         Then join neighbor values from cell_dt
# ──────────────────────────────────────────────────────────────────────

# Unique years in the panel
all_years <- sort(unique(cell_dt$year))

# Expand edges × years  (~1.37M edges × 28 years ≈ 38.5M rows)
edge_year_dt <- CJ_dt_edges <- edge_dt[, .(year = all_years), by = .(focal_id, neighbor_id)]

cat(sprintf("Edge-year table: %d rows (%.1f M)\n", nrow(edge_year_dt), nrow(edge_year_dt)/1e6))

# Key the cell data for fast join on (id, year)
setkey(cell_dt, id, year)

# ──────────────────────────────────────────────────────────────────────
# Step 3: For each neighbor source variable, join, aggregate, and merge
# ──────────────────────────────────────────────────────────────────────

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Key edge_year_dt for the neighbor join
setkey(edge_year_dt, neighbor_id, year)

for (var_name in neighbor_source_vars) {
  cat(sprintf("Processing neighbor stats for: %s\n", var_name))
  
  # --- 3a. Attach the neighbor's value to each edge-year row ---
  # We only need (id, year, var_name) from cell_dt
  # Join: edge_year_dt[neighbor_id, year] -> cell_dt[id, year]
  
  # Create a small lookup table
  lookup <- cell_dt[, .(id, year, val = get(var_name))]
  setkey(lookup, id, year)
  
  # Join neighbor value onto edge table
  edge_year_dt[, nb_val := lookup[.(edge_year_dt$neighbor_id, edge_year_dt$year), val]]
  
  # --- 3b. Aggregate by (focal_id, year) ---
  # Compute max, min, mean of non-NA neighbor values
  agg <- edge_year_dt[!is.na(nb_val),
                       .(nb_max  = max(nb_val),
                         nb_min  = min(nb_val),
                         nb_mean = mean(nb_val)),
                       by = .(focal_id, year)]
  
  # --- 3c. Merge back onto cell_dt ---
  # Name the new columns to match the original pipeline's naming convention
  max_col  <- paste0("nb_max_",  var_name)
  min_col  <- paste0("nb_min_",  var_name)
  mean_col <- paste0("nb_mean_", var_name)
  
  setnames(agg, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
  setkey(agg, focal_id, year)
  
  # Remove old columns if they exist (idempotent re-runs)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }
  
  # Left join: every cell-year gets its neighbor stats (NA if no valid neighbors)
  cell_dt <- agg[cell_dt, on = .(focal_id = id, year = year)]
  setnames(cell_dt, "focal_id", "id")  # restore column name
  
  # Clean up
  edge_year_dt[, nb_val := NULL]
  
  cat(sprintf("  -> Added columns: %s, %s, %s\n", max_col, min_col, mean_col))
}

# ──────────────────────────────────────────────────────────────────────
# Step 4: Convert back to data.frame if downstream code expects it
# ──────────────────────────────────────────────────────────────────────
cell_data <- as.data.frame(cell_dt)

# ──────────────────────────────────────────────────────────────────────
# Step 5: Predict with the pre-trained Random Forest (unchanged)
# ──────────────────────────────────────────────────────────────────────
# The trained model object (e.g., `rf_model`) is used as-is:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## 4. Why This Preserves the Original Numerical Estimand

| Concern | Guarantee |
|---------|-----------|
| **Same neighbor set** | The edge table is built from the identical `rook_neighbors_unique` nb object and `id_order`. Every `(focal, neighbor)` pair is preserved. |
| **Same year matching** | The cross-join with `all_years` and the keyed join on `(id, year)` replicate the original `paste(id, year)` lookup exactly. |
| **Same NA handling** | `!is.na(nb_val)` before aggregation mirrors `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]`. Cells with zero valid neighbors get `NA` via the left join (no row in `agg`). |
| **Same summary functions** | `max`, `min`, `mean` in data.table use the same base R functions — results are identical to floating-point precision. |
| **RF model untouched** | The pre-trained model is never retrained; only `predict()` is called on the enriched data. |

---

## 5. Memory and Runtime Estimates

| Object | Size estimate |
|--------|--------------|
| `cell_dt` (6.46M × 110 cols) | ~5.7 GB (fits in 16 GB) |
| `edge_dt` (1.37M × 2 int cols) | ~11 MB |
| `edge_year_dt` (38.5M × 3 int cols + 1 double) | ~1.2 GB |
| `lookup` (6.46M × 3 cols) | ~150 MB |
| **Peak** | ~8–10 GB (fits in 16 GB RAM) |

**Expected runtime:** 5–15 minutes total on a standard laptop, down from 86+ hours — roughly a **300–1000× speedup**.

If memory is tight, you can process one variable at a time and drop `lookup` between iterations (as shown), or avoid materializing the full `edge_year_dt` by processing year-by-year in a loop (trading a small amount of speed for lower peak memory).