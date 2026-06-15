 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is catastrophically slow (~86+ hours) for three compounding reasons:

### 1. Row-level `lapply` over 6.46 million rows in `build_neighbor_lookup`
For every single cell-year row, the function performs:
- A named-vector lookup (`id_to_ref`) — O(1) amortized but with overhead per call.
- A `paste`/key construction for every neighbor of that row.
- A named-vector lookup (`idx_lookup`) on a 6.46M-entry character vector — this is extremely slow because R's named vector lookup is O(n) or at best O(log n) per access, and you do it **for every neighbor of every row**.
- The result is a list of 6.46 million integer vectors.

**Memory:** That list alone consumes enormous RAM (each list element has overhead ≈ 100+ bytes even when small, so ~600 MB+ just in list overhead, plus the integer vectors).

### 2. `compute_neighbor_stats` iterates row-by-row again
Another `lapply` over 6.46M elements, extracting subsets of a numeric vector via index vectors. This is inherently serial and R-loop-bound.

### 3. The neighbor topology is **year-invariant** but is recomputed per cell-year
The rook-neighbor structure is purely spatial. Cell 𝑖's neighbors are the same in 1992 as in 2019. Yet `build_neighbor_lookup` re-indexes everything at the cell-year level, blowing up the problem from ~344K cells to ~6.46M rows.

**Key insight:** The adjacency structure involves only **344,208 cells** with ~1.37M directed edges. The yearly attribute values should be **joined onto** this small, fixed graph, not used to rebuild a lookup for every row.

---

## Optimization Strategy

1. **Build the adjacency table once** — a two-column `data.table` of `(cell_id, neighbor_id)` with ~1.37M rows, derived from `rook_neighbors_unique` and `id_order`. This is year-invariant and tiny.

2. **Join yearly attributes onto the edge table** — For each year and each variable, join the cell-year attribute onto the `neighbor_id` column. This gives you the neighbor's value for that variable in that year.

3. **Aggregate** — Group by `(cell_id, year)` and compute `max`, `min`, `mean` of the neighbor values in one vectorized `data.table` operation.

4. **Join back** — Merge the resulting neighbor-stats columns back onto the main `cell_data` table.

This replaces ~6.46M R-level list operations with a handful of **vectorized `data.table` joins and grouped aggregations** on a ~1.37M-row edge table × 28 years ≈ 38.4M rows (which `data.table` handles in seconds).

**Expected speedup:** From ~86 hours to **< 5 minutes** on a 16 GB laptop.

**Numerical equivalence:** The max, min, and mean computations are identical — same neighbors, same values, same aggregation functions. The trained Random Forest model is never touched.

---

## Working R Code

```r
library(data.table)

# ===========================================================
# STEP 1: Build a year-invariant adjacency table (once)
# ===========================================================
# Inputs:
#   id_order             — integer/numeric vector of cell IDs (length 344,208)
#   rook_neighbors_unique — spdep nb object (list of integer index vectors)
#
# Output:
#   adj_dt — data.table with columns: cell_id, neighbor_id
#            (~1,373,394 rows — one per directed edge)

build_adjacency_table <- function(id_order, neighbors) {
  # Pre-allocate by computing total number of edges
  n_cells <- length(id_order)
  n_edges <- sum(vapply(neighbors, length, integer(1)))
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_len(n_cells)) {
    nb_idx <- neighbors[[i]]
    n_nb   <- length(nb_idx)
    if (n_nb > 0L) {
      from_id[pos:(pos + n_nb - 1L)] <- id_order[i]
      to_id[pos:(pos + n_nb - 1L)]   <- id_order[nb_idx]
      pos <- pos + n_nb
    }
  }
  
  data.table(cell_id = from_id, neighbor_id = to_id)
}

adj_dt <- build_adjacency_table(id_order, rook_neighbors_unique)

cat(sprintf("Adjacency table: %d directed edges among %d cells\n",
            nrow(adj_dt), length(id_order)))

# ===========================================================
# STEP 2: Convert cell_data to data.table (if not already)
# ===========================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure keyed for fast joins
setkey(cell_data, id, year)

# ===========================================================
# STEP 3: For each variable, join → aggregate → merge back
# ===========================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Expand adjacency table by year (all 28 years)
#   This creates ~1.37M × 28 ≈ 38.4M rows, manageable in RAM (~1 GB)
years <- sort(unique(cell_data$year))

# Cross join adjacency edges with years
adj_year_dt <- CJ(edge_idx = seq_len(nrow(adj_dt)), year = years)
adj_year_dt[, `:=`(
  cell_id     = adj_dt$cell_id[edge_idx],
  neighbor_id = adj_dt$neighbor_id[edge_idx]
)]
adj_year_dt[, edge_idx := NULL]

cat(sprintf("Expanded edge-year table: %d rows\n", nrow(adj_year_dt)))

# Key for joining neighbor attributes
setkey(adj_year_dt, neighbor_id, year)

for (var_name in neighbor_source_vars) {
  cat(sprintf("Processing neighbor stats for: %s\n", var_name))
  
  # Extract only the columns we need for the join
  # (neighbor's attribute value, keyed by id and year)
  val_dt <- cell_data[, .(id, year, val = get(var_name))]
  setkey(val_dt, id, year)
  
  # Join: attach the neighbor cell's value to each edge-year row
  # The join is on neighbor_id == id, year == year
  adj_year_dt[val_dt, nb_val := i.val, on = .(neighbor_id = id, year = year)]
  
  # Aggregate: for each (cell_id, year), compute max/min/mean of neighbor values
  stats <- adj_year_dt[
    !is.na(nb_val),
    .(
      nb_max  = max(nb_val),
      nb_min  = min(nb_val),
      nb_mean = mean(nb_val)
    ),
    by = .(cell_id, year)
  ]
  
  # Name the output columns to match the original pipeline's naming convention
  # (adjust these names if your compute_and_add_neighbor_features used different names)
  max_col  <- paste0(var_name, "_nb_max")
  min_col  <- paste0(var_name, "_nb_min")
  mean_col <- paste0(var_name, "_nb_mean")
  
  setnames(stats, c("nb_max", "nb_min", "nb_mean"),
                  c(max_col,  min_col,  mean_col))
  
  # Remove old columns if they exist (idempotent re-runs)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) {
      cell_data[, (col) := NULL]
    }
  }
  
  # Merge back onto cell_data
  cell_data <- merge(cell_data, stats,
                     by.x = c("id", "year"),
                     by.y = c("cell_id", "year"),
                     all.x = TRUE)
  
  # Clean up the temporary column
  adj_year_dt[, nb_val := NULL]
  
  cat(sprintf("  Done. Added columns: %s, %s, %s\n", max_col, min_col, mean_col))
}

# Re-key for any downstream operations
setkey(cell_data, id, year)

cat("All neighbor features computed.\n")

# ===========================================================
# STEP 4: Predict with existing Random Forest (unchanged)
# ===========================================================
# The trained model object (e.g., `rf_model`) is used as-is.
# cell_data now has the same neighbor-stat columns with the
# same numerical values, so predictions are identical.
#
# Example (adjust to your actual model/variable names):
#   cell_data$rf_prediction <- predict(rf_model, newdata = cell_data)
```

---

## Why This Is Numerically Equivalent

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor set** | `rook_neighbors_unique[[ref_idx]]` mapped to `id_order` | Same indices, same `id_order`, pre-materialized in `adj_dt` |
| **max/min/mean** | Computed via `max()`, `min()`, `mean()` on neighbor values | Identical functions in `data.table` `j` expression |
| **NA handling** | Drops `NA` neighbor values before aggregating | `!is.na(nb_val)` filter before grouping — identical |
| **Missing neighbors** | Returns `c(NA, NA, NA)` → columns get `NA` | `all.x = TRUE` merge leaves unmatched rows as `NA` |
| **RF model** | Not modified | Not modified — same object, same `predict()` call |

## Memory Budget (16 GB laptop)

| Object | Approximate size |
|---|---|
| `cell_data` (6.46M × 110 cols) | ~5.7 GB |
| `adj_dt` (1.37M × 2 int cols) | ~11 MB |
| `adj_year_dt` (38.4M × 3 cols + 1 temp) | ~1.2 GB |
| Temporary aggregation tables | ~300 MB peak |
| **Total peak** | **~7.2 GB** ✓ |

This fits comfortably within 16 GB with room for the R session overhead and the Random Forest model object.