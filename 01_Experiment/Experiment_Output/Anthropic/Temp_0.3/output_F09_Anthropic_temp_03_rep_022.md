 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is slow (~86+ hours) for three interconnected reasons:

### 1. `build_neighbor_lookup` is O(N²)-like in practice
The function iterates over **all 6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine),
- Pastes together character keys (`paste(id, year)`),
- Performs **named-vector character matching** (`idx_lookup[neighbor_keys]`).

Named character vector lookup in R is hash-based but still involves repeated string construction and hashing for every single row. With ~6.46M rows × ~4 neighbors each ≈ **25.8 million string-match operations**, this is extremely expensive. Worse, the entire lookup table `idx_lookup` is a named character vector of length 6.46M, so each lookup traverses a large hash table.

### 2. The lookup is **year-redundant**
The spatial neighbor topology is **identical across all 28 years**. The cell-neighbor adjacency is purely spatial—it never changes. Yet `build_neighbor_lookup` rebuilds the mapping for every cell-year combination, doing 28× more work than necessary. A cell's neighbors in 1992 are the same cells as in 2019; only the attribute values change.

### 3. `compute_neighbor_stats` uses row-wise `lapply`
Even after the lookup is built, computing stats iterates row-by-row over 6.46M rows in R-level `lapply`, calling `max`, `min`, `mean` individually. This prevents vectorization.

---

## Optimization Strategy

**Core insight:** Build the adjacency table **once** at the cell level (344,208 cells, not 6.46M cell-years), store it as a two-column data.table of `(cell_id, neighbor_id)`, then use **vectorized joins** to compute neighbor statistics in bulk.

### Steps:
1. **Build a static edge table** from `rook_neighbors_unique` — a simple two-column table: `(cell_id, neighbor_id)`. This has ~1.37M rows and is built once.
2. **Join yearly attributes onto the edge table** by `(neighbor_id, year)` to pull in each neighbor's variable value.
3. **Group-by aggregate** `(cell_id, year)` to compute `max`, `min`, `mean` in one vectorized pass per variable.
4. **Join results back** onto the main `cell_data`.

This replaces 6.46M R-level iterations with a handful of `data.table` joins and group-by operations, reducing runtime from ~86 hours to **minutes**.

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 0: Convert cell_data to data.table (if not already)
# ============================================================
cell_data <- as.data.table(cell_data)

# ============================================================
# STEP 1: Build the static spatial edge table ONCE
#         from the spdep nb object (rook_neighbors_unique)
#         and the id_order vector.
#
#         rook_neighbors_unique[[i]] contains integer indices
#         into id_order for the neighbors of id_order[i].
#         id_order is a vector of 344,208 cell IDs.
# ============================================================

build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate: count total edges
  n_cells <- length(id_order)
  edge_lengths <- vapply(neighbors, length, integer(1))
  total_edges <- sum(edge_lengths)
  
  # Build vectors directly
  from_id <- rep(id_order, times = edge_lengths)
  to_idx  <- unlist(neighbors, use.names = FALSE)
  to_id   <- id_order[to_idx]
  
  data.table(cell_id = from_id, neighbor_id = to_id)
}

edge_table <- build_edge_table(id_order, rook_neighbors_unique)
# edge_table has ~1,373,394 rows: (cell_id, neighbor_id)
# This is built ONCE and reused for every variable and every year.

cat("Edge table rows:", nrow(edge_table), "\n")

# ============================================================
# STEP 2: Function to compute neighbor max, min, mean for one
#         variable using vectorized data.table joins + group-by
# ============================================================

compute_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {
  # Create a slim lookup: just (id, year, value)
  val_col <- var_name
  lookup <- cell_dt[, .(id, year, value = get(val_col))]
  setkey(lookup, id, year)
  
  # Join: for each edge (cell_id, neighbor_id), cross with all years,
  # then pull the neighbor's value for that year.
  #
  # But we don't need a full cross: we only need years that exist in cell_dt.
  # Strategy: join edge_table onto cell_dt's (id, year) to get the
  # set of (cell_id, year, neighbor_id), then look up neighbor values.
  
  # Get the (cell_id, year) combinations that exist
  cell_years <- cell_dt[, .(cell_id = id, year)]
  
  # Expand: for each (cell_id, year), attach all neighbors
  # This is a join of cell_years with edge_table on cell_id
  setkey(cell_years, cell_id)
  setkey(edge_dt, cell_id)
  
  expanded <- edge_dt[cell_years, on = "cell_id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded has columns: cell_id, neighbor_id, year
  # Rows: ~6.46M * ~4 neighbors ≈ ~25.8M rows (fits in RAM)
  
  # Look up the neighbor's value for that year
  setkey(expanded, neighbor_id, year)
  expanded[lookup, on = c(neighbor_id = "id", "year"), neighbor_val := i.value]
  
  # Aggregate: group by (cell_id, year), compute stats
  # Drop NAs in neighbor_val before aggregation
  stats <- expanded[!is.na(neighbor_val),
                    .(nb_max  = max(neighbor_val),
                      nb_min  = min(neighbor_val),
                      nb_mean = mean(neighbor_val)),
                    by = .(cell_id, year)]
  
  # Rename columns to match expected output naming convention
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats, c("nb_max", "nb_min", "nb_mean"),
           c(max_col, min_col, mean_col))
  
  # Join back onto cell_dt
  # First remove old columns if they exist (for idempotency)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }
  
  cell_dt <- merge(cell_dt, stats,
                   by.x = c("id", "year"),
                   by.y = c("cell_id", "year"),
                   all.x = TRUE, sort = FALSE)
  
  cell_dt
}

# ============================================================
# STEP 3: Outer loop — compute neighbor features for all 5 vars
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "...\n")
  t0 <- Sys.time()
  cell_data <- compute_neighbor_features_fast(cell_data, edge_table, var_name)
  cat("  Done in", round(difftime(Sys.time(), t0, units = "secs"), 1), "seconds\n")
}

# ============================================================
# STEP 4: Predict with the existing trained Random Forest
#         (model object is unchanged; no retraining)
# ============================================================

# Convert back to data.frame if the RF model expects one
cell_data_df <- as.data.frame(cell_data)

# Predict (the trained model object name may vary; shown as `rf_model`)
cell_data_df$prediction <- predict(rf_model, newdata = cell_data_df)

cat("Prediction complete. Rows:", nrow(cell_data_df), "\n")
```

---

## Summary of Gains

| Aspect | Before (Current) | After (Optimized) |
|---|---|---|
| **Neighbor lookup** | 6.46M row-wise string hashes | 1 vectorized edge table build (~1.37M rows) |
| **Neighbor stats** | 6.46M × 5 vars R-level `lapply` | 5 × one `data.table` join + group-by |
| **String operations** | ~25.8M `paste` + hash lookups | Zero |
| **Estimated runtime** | ~86+ hours | **~5–15 minutes** (5 vars × ~1–3 min each) |
| **Peak RAM** | Moderate (but slow) | ~3–5 GB for the expanded table (fits in 16 GB) |
| **RF model** | Unchanged ✓ | Unchanged ✓ |
| **Numerical results** | Original estimand | Preserved identically (same max/min/mean logic) ✓ |

The key transformation is: **separate topology (static) from attributes (yearly), join them with vectorized operations, and aggregate in bulk** — eliminating millions of R-level loop iterations and string operations.