 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of two fundamental bottlenecks:

1. **`build_neighbor_lookup` constructs a row-level lookup across all 6.46 million cell-year rows using `lapply` with string-key matching.** For each of the ~6.46M rows, it pastes neighbor cell IDs with that row's year, then looks them up in a named character vector (`idx_lookup`). This means ~6.46M iterations, each involving string concatenation, named-vector subsetting (which is O(n) hash lookup per key), and NA filtering. The named vector `idx_lookup` itself has 6.46M entries, making each lookup expensive in aggregate.

2. **The lookup is year-aware but the neighbor *topology* is year-invariant.** The spatial neighbor structure (which cell borders which cell) never changes across years. Yet `build_neighbor_lookup` re-resolves neighbor relationships at the row level (cell × year), effectively repeating the same topological work 28 times — once for every year — embedded inside a single massive loop over all 6.46M rows.

3. **`compute_neighbor_stats` then iterates over the 6.46M-element list with `lapply`, extracting values one row at a time.** This is pure R-level looping with no vectorization.

4. **Memory pressure:** A 6.46M-element list of integer vectors consumes substantial RAM and is slow to construct and traverse.

**In summary:** The pipeline treats a *static spatial topology + dynamic yearly attributes* problem as a monolithic row-level problem, foregoing the massive speedup available from separating structure from data.

---

## Optimization Strategy

### Core Idea: Build the neighbor edge table once, join yearly attributes, compute stats via vectorized grouped aggregation.

**Step 1 — Build a reusable directed edge table (cell-to-neighbor) once.**  
Convert `rook_neighbors_unique` (an `nb` object) into a two-column `data.table`: `(cell_id, neighbor_id)`. This table has ~1.37M rows and never changes.

**Step 2 — For each year, join cell attributes onto the edge table.**  
Using `data.table` keyed joins, attach the neighbor's attribute value to each edge row for a given year. This produces ~1.37M × 28 ≈ 38.5M rows (or we can process year-by-year to save RAM).

**Step 3 — Compute grouped max, min, mean.**  
Group by `(cell_id, year)` and compute the three summary statistics in one vectorized `data.table` aggregation.

**Step 4 — Join results back to the main `cell_data` table.**

This replaces 6.46M R-level iterations with vectorized `data.table` joins and grouped aggregations, reducing runtime from ~86 hours to **minutes**.

### Why this preserves correctness:
- The neighbor topology is identical (same `nb` object, same rook neighbors).
- The attribute values joined are the same original columns.
- Max, min, and mean are computed over the same neighbor sets.
- The trained Random Forest model is not retouched; we only recompute the input feature columns identically.

---

## Working R Code

```r
library(data.table)

# ==============================================================
# STEP 0: Convert cell_data to data.table if not already
# ==============================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ==============================================================
# STEP 1: Build the static directed edge table ONCE
#
# rook_neighbors_unique: an nb object (list of integer vectors)
#   where element i contains the indices (into id_order) of
#   neighbors of id_order[i].
# id_order: vector of cell IDs corresponding to the nb object.
# ==============================================================
build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(neighbors))
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors)) {
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

edge_table <- build_edge_table(id_order, rook_neighbors_unique)

cat("Edge table rows:", nrow(edge_table), "\n")
# Expected: ~1,373,394

# ==============================================================
# STEP 2: Compute neighbor stats for each variable
#
# For each variable, we:
#   a) Take the subset of cell_data with (id, year, variable).
#   b) Join onto edge_table × years to get neighbor values.
#   c) Aggregate max, min, mean grouped by (cell_id, year).
#   d) Join back onto cell_data.
# ==============================================================
compute_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {
  
  # Column names to create (must match what the RF model expects)
  col_max  <- paste0("neighbor_max_", var_name)
  col_min  <- paste0("neighbor_min_", var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  
  # Extract only what we need: id, year, and the variable
  # Using 'id' as the cell identifier column in cell_data
  attr_dt <- cell_dt[, .(id, year, value = get(var_name))]
  
  # Key the attribute table for fast join
  setkey(attr_dt, id, year)
  
  # Get unique years
  years <- sort(unique(attr_dt$year))
  
  # Process all years via a cross-join approach:
  # Expand edge_table × years, then join neighbor attributes
  
  # Create edge-year table: every edge exists in every year
  # ~1.37M edges × 28 years ≈ 38.5M rows — fits in 16 GB easily
  edge_year <- CJ(edge_idx = seq_len(nrow(edge_dt)), year = years)
  edge_year[, cell_id     := edge_dt$cell_id[edge_idx]]
  edge_year[, neighbor_id := edge_dt$neighbor_id[edge_idx]]
  edge_year[, edge_idx := NULL]
  
  # Join neighbor's attribute value
  setkey(edge_year, neighbor_id, year)
  setkey(attr_dt, id, year)
  edge_year[attr_dt, neighbor_val := i.value, on = .(neighbor_id = id, year = year)]
  
  # Remove rows where neighbor value is NA (matches original behavior)
  edge_year <- edge_year[!is.na(neighbor_val)]
  
  # Aggregate: max, min, mean by (cell_id, year)
  stats <- edge_year[, .(
    nmax  = max(neighbor_val),
    nmin  = min(neighbor_val),
    nmean = mean(neighbor_val)
  ), by = .(cell_id, year)]
  
  # Rename columns
  setnames(stats, c("nmax", "nmin", "nmean"), c(col_max, col_min, col_mean))
  
  # Remove old columns from cell_dt if they exist (idempotent re-runs)
  for (cc in c(col_max, col_min, col_mean)) {
    if (cc %in% names(cell_dt)) cell_dt[, (cc) := NULL]
  }
  
  # Join stats back onto cell_dt
  setkey(stats, cell_id, year)
  setkey(cell_dt, id, year)
  cell_dt <- merge(cell_dt, stats, by.x = c("id", "year"), by.y = c("cell_id", "year"), all.x = TRUE)
  
  # Clean up
  rm(edge_year, stats, attr_dt)
  gc()
  
  cell_dt
}

# ==============================================================
# STEP 3: Run for all 5 neighbor source variables
# ==============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "...\n")
  t0 <- proc.time()
  cell_data <- compute_neighbor_features_fast(cell_data, edge_table, var_name)
  elapsed <- (proc.time() - t0)[3]
  cat("  Done in", round(elapsed, 1), "seconds\n")
}

# ==============================================================
# STEP 4: Predict with the existing trained Random Forest
#
# The RF model object (e.g., rf_model) is already in memory
# and must NOT be retrained. The feature columns now exist in
# cell_data with identical names and identical numerical values.
# ==============================================================

# Example (adjust to your actual model object and prediction workflow):
# predictions <- predict(rf_model, newdata = cell_data)

cat("\nAll neighbor features computed. Ready for Random Forest prediction.\n")
cat("Rows:", nrow(cell_data), " Columns:", ncol(cell_data), "\n")
```

---

## Expected Performance

| Step | Original | Optimized |
|---|---|---|
| Build neighbor lookup | ~hours (6.46M string ops) | ~2 seconds (vectorized edge table) |
| Compute stats per variable | ~17 hours × 5 = 85h | ~1–3 minutes × 5 ≈ 5–15 min |
| **Total** | **~86+ hours** | **~5–20 minutes** |

### Memory footprint estimate:
- `edge_table`: ~1.37M rows × 2 int cols ≈ 11 MB
- `edge_year` (peak): ~38.5M rows × 4 cols ≈ 1.2 GB (transient, freed per variable)
- `cell_data`: 6.46M × ~125 cols ≈ 6.5 GB
- **Total peak: ~8–9 GB** — fits within 16 GB RAM

### Correctness guarantees:
- **Same topology:** Edge table is derived from the identical `rook_neighbors_unique` nb object and `id_order`.
- **Same statistics:** `max`, `min`, `mean` are computed over the same non-NA neighbor values per `(cell_id, year)` group.
- **Same column names:** Output columns follow the `neighbor_max_<var>`, `neighbor_min_<var>`, `neighbor_mean_<var>` naming convention (adjust the prefix pattern if your RF model expects different names).
- **Trained RF model untouched:** No retraining occurs; only the input feature matrix is recomputed identically.