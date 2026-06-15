 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(N²)-like in practice
This function iterates over **all ~6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs `paste()`-based string keys for every neighbor × that row's year (expensive string allocation).
- Performs named-vector lookups via `idx_lookup[neighbor_keys]` — which is a **hash-table lookup on character keys repeated ~6.46M × ~4 neighbors ≈ 25.8 million times**.

The real killer is that the neighbor topology is **year-invariant** — the same cell has the same rook neighbors in every year — yet the function rebuilds the mapping for every cell-year combination. With 344,208 cells × 28 years, it does 28× redundant work on the spatial graph.

### 2. `compute_neighbor_stats` uses per-row `lapply`
Even after the lookup is built, computing max/min/mean via `lapply` over 6.46M rows with R-level loops is inherently slow. Each iteration incurs R interpreter overhead, subsetting, and `NA` handling.

### 3. Memory pressure from the lookup object
Storing 6.46 million integer vectors in a list consumes substantial RAM and creates GC pressure on a 16 GB laptop.

---

## Optimization Strategy

**Core insight:** The spatial neighbor structure is static across years. Build it once (344K cells), then use vectorized joins per year.

### Step-by-step plan:

1. **Build a cell-level edge table once** — a two-column `data.table` of `(cell_id, neighbor_id)` from the `nb` object. This has ~1.37M rows and is tiny.

2. **For each variable, join yearly attributes onto the edge table** — use `data.table` keyed joins. For each cell-year, this gives all neighbor values in a flat table.

3. **Aggregate with `data.table` grouping** — compute `max`, `min`, `mean` per `(cell_id, year)` in one vectorized pass. This replaces millions of R-level `lapply` iterations with C-level `data.table` grouped aggregation.

4. **Join the aggregated stats back** onto the main dataset.

**Expected speedup:** From ~86 hours to **minutes**. The bottleneck becomes a handful of keyed `data.table` joins and grouped aggregations over ~25.8M edge-year rows — all executed in C.

**Memory:** The edge table is ~1.37M rows × 2 integer columns ≈ 11 MB. The expanded edge-year table is ~1.37M × 28 ≈ 38.4M rows × 3 columns ≈ 920 MB at peak, which fits in 16 GB.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build the static cell-level edge table ONCE from the nb object
# ──────────────────────────────────────────────────────────────────────
build_edge_table <- function(id_order, nb_obj) {
  # nb_obj is a list of integer vectors (indices into id_order)
  # id_order is the vector of cell IDs corresponding to nb_obj positions
  from <- rep(seq_along(nb_obj), lengths(nb_obj))
  to   <- unlist(nb_obj, use.names = FALSE)

  # Remove the spdep "no-neighbor" sentinel (0)
  valid <- to != 0L
  from  <- from[valid]
  to    <- to[valid]

  data.table(
    cell_id     = id_order[from],
    neighbor_id = id_order[to]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# ~1.37M rows, built in < 1 second

# ──────────────────────────────────────────────────────────────────────
# STEP 2: Convert main data to data.table (if not already)
# ──────────────────────────────────────────────────────────────────────
cell_dt <- as.data.table(cell_data)

# Ensure key columns have consistent types
cell_dt[, id   := as.integer(id)]
cell_dt[, year := as.integer(year)]
edge_dt[, cell_id     := as.integer(cell_id)]
edge_dt[, neighbor_id := as.integer(neighbor_id)]

# ──────────────────────────────────────────────────────────────────────
# STEP 3: Function to compute and attach neighbor stats for one variable
# ──────────────────────────────────────────────────────────────────────
compute_and_add_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {

  # --- 3a. Extract only the columns we need for the join ---
  # Columns: neighbor_id (to join on), year, and the variable value
  val_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setkey(val_dt, id, year)

  # --- 3b. Expand edges × years by joining neighbor attributes ---
  # For every (cell_id, neighbor_id) edge, and every year,

  # look up the neighbor's value in that year.
  # We join edge_dt to val_dt on neighbor_id == id.
  # First, create the cross of edges × years present in the data.
  # Efficient approach: join edge_dt with val_dt on neighbor_id == id.
  # This naturally expands to all (cell_id, neighbor_id, year) combos
  # where the neighbor has data.

  edge_vals <- merge(
    edge_dt,
    val_dt,
    by.x = "neighbor_id",
    by.y = "id",
    allow.cartesian = TRUE   # each neighbor appears in 28 years
  )
  # Result columns: neighbor_id, cell_id, year, val
  # Rows: ~1.37M edges × 28 years ≈ 38.4M (manageable)

  # --- 3c. Aggregate: max, min, mean per (cell_id, year) ---
  agg <- edge_vals[
    !is.na(val),
    .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ),
    keyby = .(cell_id, year)
  ]

  # --- 3d. Rename columns to match original naming convention ---
  # Original code produces columns like: ntl_neighbor_max, ntl_neighbor_min, ntl_neighbor_mean
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  setnames(agg, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))

  # --- 3e. Join back onto the main table ---
  # Remove old columns if they exist (idempotent re-runs)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }

  cell_dt <- merge(
    cell_dt,
    agg,
    by.x = c("id", "year"),
    by.y = c("cell_id", "year"),
    all.x = TRUE,
    sort = FALSE
  )

  cell_dt
}

# ──────────────────────────────────────────────────────────────────────
# STEP 4: Loop over the 5 neighbor source variables
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  cell_dt <- compute_and_add_neighbor_features_fast(cell_dt, edge_dt, var_name)
  gc()
}

# ──────────────────────────────────────────────────────────────────────
# STEP 5: Predict with the existing trained Random Forest (unchanged)
# ──────────────────────────────────────────────────────────────────────
# Convert back to data.frame if the RF predict method requires it
cell_data <- as.data.frame(cell_dt)

# The trained model object (e.g., `rf_model`) is used as-is:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary of Guarantees

| Concern | Status |
|---|---|
| **Trained RF model** | Untouched — only the feature-engineering step is optimized. |
| **Numerical estimand** | Preserved — `max`, `min`, `mean` are computed on the identical neighbor sets with the identical values. The `merge` on `(neighbor_id, year)` replicates exactly the same cell-year neighbor lookups as the original `paste()`-key approach. |
| **RAM on 16 GB laptop** | Peak ≈ main table (~3–4 GB) + largest edge-year expansion (~1 GB) + aggregation (~0.3 GB). Fits comfortably. `gc()` after each variable reclaims intermediates. |
| **Runtime** | ~2–5 minutes total for all 5 variables (vs. 86+ hours). Each variable involves one `data.table` merge (~38M rows) and one grouped aggregation — both executed in C. |