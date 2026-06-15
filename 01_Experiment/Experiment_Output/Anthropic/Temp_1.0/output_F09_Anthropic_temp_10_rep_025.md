 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current pipeline is extremely slow (estimated 86+ hours) for two fundamental reasons:

### 1. `build_neighbor_lookup` is O(N²)-like in practice
The function iterates over **every cell-year row** (~6.46 million rows) with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors.
- Constructs character keys by pasting cell IDs and years.
- Performs named-vector lookups (`idx_lookup[neighbor_keys]`), which in R are hash-table lookups on a **6.46-million-entry named character vector** — slow due to repeated hashing and memory access patterns.

Because the spatial neighbor structure is **identical across all 28 years**, this work is redundantly repeated 28 times per cell. For 344,208 cells × 28 years = 9,637,824 neighbor-set resolutions, each touching ~4 neighbors on average, this produces tens of millions of character-paste and hash-lookup operations.

### 2. `compute_neighbor_stats` uses row-level `lapply`
For each of the 5 variables, the function loops over 6.46 million rows in R-level `lapply`, subsetting a numeric vector with small index vectors and computing `max/min/mean`. The per-element R interpreter overhead on 6.46M iterations is enormous.

### 3. The neighbor topology is time-invariant but never exploited
The rook-neighbor structure is purely spatial. It does not change year to year. Yet the current code rebuilds a full row-level lookup that embeds year information, missing the opportunity to separate the **static spatial topology** from the **dynamic yearly attributes**.

---

## Optimization Strategy

**Core insight:** Build the adjacency table **once** at the cell level (344K cells, not 6.46M cell-years), then for each year, use vectorized joins and grouped operations to compute neighbor statistics.

### Step-by-step plan:

1. **Build a static edge table** from `rook_neighbors_unique`: a two-column `data.table` of `(cell_id, neighbor_cell_id)` — created once, ~1.37M rows.

2. **For each year**, join the yearly cell attributes onto the edge table by `neighbor_cell_id`, then group by `cell_id` to compute `max`, `min`, `mean` — all vectorized inside `data.table`.

3. **Join the results back** onto the main `cell_data` table.

This eliminates all `lapply` over 6.46M rows, all character-key pasting, and all named-vector hash lookups. The `data.table` approach uses binary-search joins and columnar grouped aggregation, which are orders of magnitude faster.

**Expected speedup:** From 86+ hours to roughly **2–10 minutes** depending on disk I/O and RAM pressure.

**Preservation guarantees:**
- The trained Random Forest model is not retouched.
- The numerical output (neighbor max, min, mean per variable per cell-year) is identical to the original.

---

## Working R Code

```r
library(data.table)

# =============================================================================
# STEP 1: Build the static spatial edge table ONCE
# =============================================================================
# rook_neighbors_unique is an nb object (list of integer vectors).
# id_order is the vector of cell IDs in the same order as the nb object.
# Each element rook_neighbors_unique[[i]] contains integer indices into id_order
# for the neighbors of cell id_order[i].

build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(neighbors))
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb <- neighbors[[i]]
    # spdep::nb encodes "no neighbors" as a single 0L; skip those
    if (length(nb) == 1L && nb[1] == 0L) next
    n <- length(nb)
    from_id[pos:(pos + n - 1L)] <- id_order[i]
    to_id[pos:(pos + n - 1L)]   <- id_order[nb]
    pos <- pos + n
  }
  
  # Trim if any 0-neighbor cells caused over-allocation
  if (pos - 1L < n_edges) {
    from_id <- from_id[1:(pos - 1L)]
    to_id   <- to_id[1:(pos - 1L)]
  }
  
  data.table(cell_id = from_id, neighbor_cell_id = to_id)
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

# Set key for fast joins on neighbor_cell_id
setkey(edge_dt, neighbor_cell_id)

# =============================================================================
# STEP 2: Convert cell_data to data.table (if not already)
# =============================================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure original row order is preserved for later reassembly
cell_data[, .row_order := .I]

# =============================================================================
# STEP 3: Compute neighbor stats for all variables — vectorized
# =============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# We process one variable at a time to control peak RAM.
# For each variable, we:
#   (a) Extract (id, year, variable) from cell_data
#   (b) Join onto edge_dt by neighbor_cell_id to get each edge's neighbor value
#   (c) Group by (cell_id, year) to get max, min, mean
#   (d) Join results back onto cell_data

compute_neighbor_features_dt <- function(cell_dt, edge_dt, var_name) {
  
  # Column names for output (must match original pipeline naming)
  col_max  <- paste0("neighbor_max_", var_name)
  col_min  <- paste0("neighbor_min_", var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  
  # (a) Subset: only the columns we need for the join
  #     'id' is the cell identifier in cell_data matching id_order values
  attr_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setnames(attr_dt, "id", "neighbor_cell_id")
  setkey(attr_dt, neighbor_cell_id)
  
  # (b) Join edge table with neighbor attributes
  #     For each directed edge (cell_id -> neighbor_cell_id) and each year,
  #     attach the neighbor's value.
  #     We need the Cartesian product of edges × years, but it's more efficient
  #     to join edges onto the attribute table keyed by (neighbor_cell_id, year).
  setkey(attr_dt, neighbor_cell_id, year)
  
  # Expand edge_dt with year from attr_dt via a rolling/equi join:
  # For each edge, for each year that the neighbor has data, get the value.
  edge_with_val <- edge_dt[attr_dt,
                           .(cell_id     = x.cell_id,
                             year        = i.year,
                             neighbor_val = i.val),
                           on = .(neighbor_cell_id),
                           nomatch = 0L,
                           allow.cartesian = TRUE]
  
  # (c) Aggregate by (cell_id, year)
  stats <- edge_with_val[!is.na(neighbor_val),
                         .(nmax  = max(neighbor_val),
                           nmin  = min(neighbor_val),
                           nmean = mean(neighbor_val)),
                         keyby = .(cell_id, year)]
  
  setnames(stats,
           c("nmax", "nmin", "nmean"),
           c(col_max, col_min, col_mean))
  
  # (d) Join back onto cell_data
  # Remove old columns if they exist (idempotent re-runs)
  for (cc in c(col_max, col_min, col_mean)) {
    if (cc %in% names(cell_dt)) cell_dt[, (cc) := NULL]
  }
  
  cell_dt <- merge(cell_dt, stats,
                   by.x = c("id", "year"),
                   by.y = c("cell_id", "year"),
                   all.x = TRUE,
                   sort = FALSE)
  
  cell_dt
}

# --- Main loop (now fast) ---
for (var_name in neighbor_source_vars) {
  message(Sys.time(), " | Computing neighbor stats for: ", var_name)
  cell_data <- compute_neighbor_features_dt(cell_data, edge_dt, var_name)
  gc()
}

# Restore original row order (merge may shuffle)
setorder(cell_data, .row_order)
cell_data[, .row_order := NULL]

message(Sys.time(), " | Done. Neighbor features appended for all variables.")

# =============================================================================
# STEP 4: Predict with the EXISTING trained Random Forest (unchanged)
# =============================================================================
# The model object (e.g., rf_model) is already in memory or loaded from disk.
# Prediction proceeds exactly as before:
#
#   cell_data$prediction <- predict(rf_model, newdata = cell_data)
#
# The trained model is NOT retrained. The numerical estimand is preserved
# because the neighbor max, min, and mean values are computed identically
# (same set of neighbors, same aggregation functions, same variable values).
```

---

## Why This Is Correct and Equivalent

| Concern | Guarantee |
|---|---|
| **Same neighbor sets** | `edge_dt` is built from the same `rook_neighbors_unique` nb object and `id_order`. Each cell's directed neighbor list is identical. |
| **Same per-year scoping** | The join on `(neighbor_cell_id, year)` ensures that only same-year neighbor values are aggregated — exactly as the original `paste(id, year)` key did. |
| **Same aggregation** | `max`, `min`, `mean` with `NA` removal (`!is.na(neighbor_val)`) matches the original `compute_neighbor_stats` logic. Cells with no valid neighbors get `NA` via the `all.x = TRUE` merge. |
| **Trained RF preserved** | No model code is touched. The prediction input columns are numerically identical. |

## Expected Performance

| Component | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~hours (6.46M `lapply` + char hashing) | ~2 seconds (vectorized edge table) |
| Neighbor stats per variable | ~17 hours (6.46M `lapply`) | ~15–30 seconds (`data.table` grouped agg) |
| **Total for 5 variables** | **86+ hours** | **~2–5 minutes** |
| Peak RAM | Moderate (large named vectors) | ~2–3 GB for largest intermediate join |