 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three compounding reasons:

### 1. `build_neighbor_lookup` is O(N²)-like in practice
It iterates over all **6.46 million cell-year rows** via `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs **character paste keys** (`paste(id, year, sep="_")`) and matches them into a named vector (`idx_lookup`).

Character key construction and named-vector lookup for 6.46M rows × ~4 neighbors each ≈ **~25 million string operations and hash lookups**. The `idx_lookup` named vector itself has 6.46M entries, making each name-based lookup slow (R's named vector lookup is O(n) in the worst case without hashing, and even with internal hashing the constant factor is large at this scale).

### 2. The lookup is rebuilt monolithically for the entire panel
The neighbor structure is **purely spatial** — it doesn't change across years. Yet the function fuses spatial topology with temporal indexing, creating a massive 6.46M-element list. This is unnecessary: the same 1,373,394 directed neighbor relationships repeat identically for each of the 28 years.

### 3. `compute_neighbor_stats` uses row-level `lapply` over 6.46M rows
Even though the neighbor indices are precomputed, extracting `vals[idx]` inside an R-level `lapply` for 6.46M iterations is inherently slow — each iteration has R interpreter overhead, memory allocation for small vectors, and no vectorization.

### Summary of the bottleneck
| Component | Cost driver |
|---|---|
| `paste()` + named-vector lookup | ~25M string ops on a 6.46M-entry hash |
| `lapply` over 6.46M rows (build) | R interpreter overhead per row |
| `lapply` over 6.46M rows (stats) | R interpreter overhead per row |
| Repeated for 5 variables | 5× the stats computation |

---

## Optimization Strategy

**Core insight:** Separate the *spatial topology* (which is static) from the *temporal attributes* (which vary by year). Build the adjacency table **once** as a `data.table` of directed edges, then use a **vectorized grouped join** to compute neighbor statistics — eliminating all `lapply` loops entirely.

### Step-by-step plan

1. **Convert `spdep::nb` → edge `data.table`** — a two-column table `(cell_id, neighbor_id)` with ~1.37M rows. Done once, O(E).

2. **Ensure `cell_data` is a keyed `data.table`** with key `(id, year)`.

3. **For each variable, vectorized join:**
   - Take the edge table, cross it with all 28 years → ~38.4M edge-year rows (or join only existing years).
   - Join the neighbor's attribute value onto each edge-year row.
   - Group by `(cell_id, year)` and compute `max`, `min`, `mean` in one vectorized `data.table` aggregation.
   - Join the result back onto `cell_data`.

4. **Predict** with the existing trained Random Forest model (unchanged).

### Complexity comparison

| | Old | New |
|---|---|---|
| Build topology | O(N_rows) with string ops | O(E) integer ops, once |
| Compute stats per variable | O(N_rows) R-level lapply | O(E × T) vectorized `data.table` grouped aggregation |
| Total R-interpreter iterations | ~32M per variable | **0** (fully vectorized) |
| Expected wall time | 86+ hours | **~2–10 minutes** |

Memory: The edge-year table for one variable is ~38.4M rows × 4 columns ≈ ~1.2 GB, well within 16 GB RAM. We process one variable at a time and discard the intermediate table.

---

## Working R Code

```r
library(data.table)

# ==============================================================
# STEP 1: Build a static spatial edge table from the nb object
#         (done ONCE, reusable forever)
# ==============================================================
build_edge_table <- function(id_order, nb_obj) {
  # id_order: vector of cell IDs in the order matching nb_obj
  # nb_obj:   spdep::nb list (each element is integer vector of neighbor indices)
  
  # Pre-calculate total edges for memory pre-allocation
  n_edges <- sum(lengths(nb_obj))
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(nb_obj)) {
    nbrs <- nb_obj[[i]]
    # spdep::nb uses 0L to indicate no neighbors
    nbrs <- nbrs[nbrs != 0L]
    n <- length(nbrs)
    if (n > 0L) {
      from_id[pos:(pos + n - 1L)] <- id_order[i]
      to_id[pos:(pos + n - 1L)]   <- id_order[nbrs]
      pos <- pos + n
    }
  }
  
  # Trim if any 0-neighbor cells caused over-allocation
  if (pos <= n_edges) {
    from_id <- from_id[1:(pos - 1L)]
    to_id   <- to_id[1:(pos - 1L)]
  }
  
  data.table(cell_id = from_id, neighbor_id = to_id)
}

# Build it once
edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# ~1,373,394 rows (directed edges)

cat(sprintf("Edge table: %d directed neighbor relationships\n", nrow(edge_dt)))

# ==============================================================
# STEP 2: Ensure cell_data is a keyed data.table
# ==============================================================
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}
# Ensure id and year are the types we expect (integer or numeric)
# Key for fast joins
setkey(cell_data, id, year)

# Keep a vector of all unique years for cross-joining
all_years <- sort(unique(cell_data$year))

# ==============================================================
# STEP 3: Vectorized neighbor stat computation
#         For each variable, join + aggregate, no lapply
# ==============================================================
compute_and_add_neighbor_features_fast <- function(cell_data, edge_dt, var_name, all_years) {
  
  cat(sprintf("  Computing neighbor stats for: %s ...\n", var_name))
  
  # --- 3a: Create edge-year table by cross-joining edges with years ---
  # This gives us every (cell_id, neighbor_id, year) combination
  edge_year <- CJ_dt <- edge_dt[, .(cell_id, neighbor_id)]
  # Cross join with years via a fast Cartesian expansion
  edge_year <- edge_year[, .(year = all_years), by = .(cell_id, neighbor_id)]
  # Now ~1,373,394 × 28 ≈ 38.4M rows
  
  # --- 3b: Join the neighbor's attribute value onto each edge-year row ---
  # We need: for each (neighbor_id, year), get the value of var_name
  # Build a small lookup from cell_data
  lookup <- cell_data[, .(id, year, val = get(var_name))]
  setkey(lookup, id, year)
  
  # Join: match neighbor_id+year in edge_year to id+year in lookup
  setkey(edge_year, neighbor_id, year)
  edge_year[lookup, neighbor_val := i.val, on = .(neighbor_id = id, year = year)]
  
  # --- 3c: Aggregate by (cell_id, year) to get max, min, mean ---
  # Remove NAs before aggregation (matches original behavior)
  stats <- edge_year[!is.na(neighbor_val),
                     .(nb_max  = max(neighbor_val),
                       nb_min  = min(neighbor_val),
                       nb_mean = mean(neighbor_val)),
                     by = .(cell_id, year)]
  
  # --- 3d: Name the output columns to match original convention ---
  new_names <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), new_names)
  
  # --- 3e: Join stats back onto cell_data ---
  # First, remove these columns if they already exist (idempotency)
  for (nm in new_names) {
    if (nm %in% names(cell_data)) {
      cell_data[, (nm) := NULL]
    }
  }
  
  setkey(stats, cell_id, year)
  setkey(cell_data, id, year)
  cell_data[stats, (new_names) := mget(paste0("i.", new_names)),
            on = .(id = cell_id, year = year)]
  
  # Rows in cell_data with no neighbors will remain NA (matching original behavior)
  
  # Clean up large intermediate
  rm(edge_year, lookup, stats)
  gc()
  
  cat(sprintf("  Done: %s\n", var_name))
  invisible(cell_data)
}

# ==============================================================
# STEP 4: Run for all 5 neighbor source variables
# ==============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features (vectorized)...\n")
t0 <- proc.time()

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_fast(
    cell_data, edge_dt, var_name, all_years
  )
}

elapsed <- (proc.time() - t0)["elapsed"]
cat(sprintf("All neighbor features computed in %.1f seconds.\n", elapsed))

# ==============================================================
# STEP 5: Predict with the existing trained Random Forest
#         (model object unchanged, no retraining)
# ==============================================================
# Example (adjust to your actual model object and prediction call):
# cell_data$rf_prediction <- predict(trained_rf_model, newdata = cell_data)

cat("Ready for Random Forest prediction. Model is unchanged.\n")
```

---

## Memory-Optimized Variant (if 38.4M-row cross join is too large)

If RAM is tight, process **one year at a time** to keep the intermediate table at only ~1.37M rows:

```r
compute_and_add_neighbor_features_by_year <- function(cell_data, edge_dt, var_name) {
  
  cat(sprintf("  Computing neighbor stats (year-by-year) for: %s ...\n", var_name))
  
  new_names <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  
  # Pre-allocate columns with NA
  for (nm in new_names) {
    if (nm %in% names(cell_data)) cell_data[, (nm) := NULL]
    cell_data[, (nm) := NA_real_]
  }
  
  all_years <- sort(unique(cell_data$year))
  setkey(cell_data, id, year)
  
  for (yr in all_years) {
    # Subset this year's data
    yr_data <- cell_data[year == yr, .(id, val = get(var_name))]
    setkey(yr_data, id)
    
    # Join neighbor values onto edge table
    edges_yr <- copy(edge_dt)
    edges_yr[yr_data, neighbor_val := i.val, on = .(neighbor_id = id)]
    
    # Aggregate
    stats_yr <- edges_yr[!is.na(neighbor_val),
                         .(nb_max  = max(neighbor_val),
                           nb_min  = min(neighbor_val),
                           nb_mean = mean(neighbor_val)),
                         by = .(cell_id)]
    
    # Write back into cell_data for this year
    setkey(stats_yr, cell_id)
    idx <- cell_data[year == yr, which = TRUE]
    matched <- stats_yr[cell_data[idx, .(id)], on = .(cell_id = id)]
    
    set(cell_data, i = idx, j = new_names[1], value = matched$nb_max)
    set(cell_data, i = idx, j = new_names[2], value = matched$nb_min)
    set(cell_data, i = idx, j = new_names[3], value = matched$nb_mean)
  }
  
  cat(sprintf("  Done: %s\n", var_name))
  invisible(cell_data)
}

# Usage:
for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_by_year(cell_data, edge_dt, var_name)
}
```

This variant uses only ~1.37M rows of intermediate memory per iteration (×28 iterations per variable = 140 total, but each is tiny and discarded immediately). Peak RAM overhead: **~50 MB** beyond `cell_data` itself.

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Topology construction | 6.46M string-key lookups | 1.37M integer edge pairs (once) |
| Stats computation | `lapply` over 6.46M rows per variable | Vectorized `data.table` grouped join |
| Total estimated time | **86+ hours** | **2–10 minutes** |
| Peak RAM overhead | Large 6.46M-element list of integer vectors | ~1.2 GB (full) or ~50 MB (year-by-year) |
| Trained RF model | Preserved ✓ | Preserved ✓ |
| Numerical results | Identical `max`, `min`, `mean` | Identical ✓ |