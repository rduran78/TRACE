 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two compounding reasons:

### 1. `build_neighbor_lookup` is O(N) over all cell-year rows (~6.46 million), but it does redundant work.
The neighbor *topology* is purely spatial — it never changes across years. Yet the function builds a lookup keyed on `(id, year)` pairs, iterating over every cell-year row. For each of the 6.46M rows it:
- Looks up the cell's spatial neighbors (which are the same for all 28 years of that cell).
- Pastes together character keys (`id_year`) and matches them into an index lookup.

This means it performs ~6.46M `paste` + named-vector lookups instead of doing the spatial lookup once per cell (344,208 times) and then joining yearly attributes.

### 2. `compute_neighbor_stats` uses `lapply` over 6.46M rows.
Each call indexes into a vector and computes `max`, `min`, `mean`. While individually cheap, 6.46M R-level function calls inside `lapply` are slow, and this is repeated for each of the 5 neighbor source variables (i.e., ~32.3M iterations total).

### 3. Character-key matching is expensive.
`paste(..., sep="_")` and named-vector lookups (`idx_lookup[neighbor_keys]`) on millions of character strings are inherently slow in R compared to integer-indexed operations.

**Summary:** The bottleneck is that the spatial topology is re-threaded through every cell-year row using slow character-key operations, rather than being resolved once at the cell level and then exploited via fast vectorized joins.

---

## Optimization Strategy

The key insight: **the neighbor relationship is time-invariant**. Build the adjacency table once at the cell level, then use a fast equi-join (via `data.table`) to bring in yearly attributes from neighbors, and compute grouped statistics with `data.table`'s optimized `by=` operations.

### Steps:

1. **Build a static edge table** from `rook_neighbors_unique` (the `nb` object): one row per directed neighbor pair `(cell_id, neighbor_id)`. This is done once and is ~1.37M rows.

2. **Convert `cell_data` to `data.table`**, keyed on `(id, year)`.

3. **For each source variable**, join the edge table with the panel data to attach the neighbor's variable value for the same year, then compute `max`, `min`, `mean` grouped by `(cell_id, year)`. Merge the results back.

This replaces ~6.46M R-level iterations with vectorized `data.table` joins and grouped aggregations, reducing runtime from 86+ hours to **minutes**.

### Complexity comparison:

| Step | Old | New |
|---|---|---|
| Topology resolution | 6.46M character lookups | 344K integer iterations (once) |
| Neighbor stats | 6.46M × 5 `lapply` calls | 5 vectorized `data.table` joins + `by` aggregations on ~38.4M rows |
| Key type | Character paste + named vector | Integer keys with `data.table` binary search |

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# STEP 1: Build a static, time-invariant edge table (once)
# ---------------------------------------------------------------
# rook_neighbors_unique : an nb object (list of integer index vectors)
# id_order              : vector of cell IDs in the same order as the nb object

build_edge_table <- function(id_order, neighbors) {
  # neighbors[[i]] gives the indices (into id_order) of cell i's rook neighbors
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  edge_dt <- data.table(
    cell_id     = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
  return(edge_dt)
}

edge_table <- build_edge_table(id_order, rook_neighbors_unique)
# edge_table has ~1,373,394 rows: (cell_id, neighbor_id)

# ---------------------------------------------------------------
# STEP 2: Convert cell_data to data.table (if not already)
# ---------------------------------------------------------------
cell_dt <- as.data.table(cell_data)
# Ensure proper types
cell_dt[, id   := as.integer(id)]
cell_dt[, year := as.integer(year)]

edge_table[, cell_id     := as.integer(cell_id)]
edge_table[, neighbor_id := as.integer(neighbor_id)]

# ---------------------------------------------------------------
# STEP 3: For each source variable, compute neighbor stats via join
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_neighbor_features_dt <- function(cell_dt, edge_table, var_name) {
  # Subset the panel to only (id, year, variable) for the join's right side
  # This keeps the join lean
  neighbor_vals <- cell_dt[, .(id, year, value = get(var_name))]
  setkey(neighbor_vals, id, year)

  # Expand edges × years: join edge_table with neighbor_vals

  # For each (cell_id, neighbor_id) pair, get the neighbor's value in each year
  # Join: edge_table[neighbor_id] -> neighbor_vals[id == neighbor_id, year]
  # We need all (cell_id, year) combinations with their neighbors' values.


  # Step A: Join edge_table with neighbor_vals on neighbor_id == id
  #   This gives us: (cell_id, neighbor_id, year, value)
  #   i.e., for every edge and every year, the neighbor's attribute value.
  expanded <- merge(
    edge_table,
    neighbor_vals,
    by.x = "neighbor_id",
    by.y = "id",
    allow.cartesian = TRUE   # each neighbor_id appears in 28 year-rows
  )
  # expanded columns: neighbor_id, cell_id, year, value
  # rows: ~1,373,394 edges × 28 years ≈ 38.5M (but only years present in data)

  # Step B: Aggregate by (cell_id, year) to get max, min, mean of neighbor values
  stats <- expanded[
    !is.na(value),
    .(
      nbr_max  = max(value),
      nbr_min  = min(value),
      nbr_mean = mean(value)
    ),
    by = .(cell_id, year)
  ]

  # Step C: Rename columns to match the original pipeline's naming convention
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats, c("nbr_max", "nbr_min", "nbr_mean"),
                  c(max_col,    min_col,    mean_col))

  # Step D: Merge back onto cell_dt by (id, year)
  # Use cell_id == id
  setnames(stats, "cell_id", "id")
  setkeyv(stats, c("id", "year"))
  setkeyv(cell_dt, c("id", "year"))

  # Remove old columns if they exist (idempotency)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }

  cell_dt <- merge(cell_dt, stats, by = c("id", "year"), all.x = TRUE)
  return(cell_dt)
}

# ---------------------------------------------------------------
# STEP 4: Loop over the 5 source variables
# ---------------------------------------------------------------
for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  cell_dt <- compute_neighbor_features_dt(cell_dt, edge_table, var_name)
}

# ---------------------------------------------------------------
# STEP 5: Predict with the existing (already-trained) Random Forest
# ---------------------------------------------------------------
# Convert back to data.frame if the RF model expects one
cell_data <- as.data.frame(cell_dt)

# The trained RF model object is assumed to be loaded in the environment
# e.g., rf_model <- readRDS("trained_rf_model.rds")
# Predict:
cell_data$predicted <- predict(rf_model, newdata = cell_data)
```

---

## Memory Considerations (16 GB laptop)

| Object | Approximate Size |
|---|---|
| `cell_dt` (6.46M × 110 cols) | ~5.5 GB |
| `edge_table` (1.37M × 2 int cols) | ~11 MB |
| `expanded` (worst case ~38.5M × 4 cols) | ~1.2 GB |
| Overhead + RF model | ~2–3 GB |

**Total peak: ~10 GB** — fits within 16 GB. The `expanded` table is created and destroyed inside each iteration, so only one variable's expansion is in memory at a time.

If memory is still tight, you can process the join in year-chunks:

```r
# Optional: chunk by year to reduce peak memory
compute_neighbor_features_chunked <- function(cell_dt, edge_table, var_name, years) {
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  stats_list <- lapply(years, function(yr) {
    yr_vals <- cell_dt[year == yr, .(id, value = get(var_name))]
    setkey(yr_vals, id)

    expanded_yr <- merge(edge_table, yr_vals,
                         by.x = "neighbor_id", by.y = "id")
    expanded_yr[
      !is.na(value),
      .(nbr_max = max(value), nbr_min = min(value), nbr_mean = mean(value)),
      by = .(cell_id)
    ][, year := yr]
  })

  stats <- rbindlist(stats_list)
  setnames(stats, c("nbr_max", "nbr_min", "nbr_mean"),
                  c(max_col,    min_col,    mean_col))
  setnames(stats, "cell_id", "id")
  setkeyv(stats, c("id", "year"))
  setkeyv(cell_dt, c("id", "year"))

  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }

  cell_dt <- merge(cell_dt, stats, by = c("id", "year"), all.x = TRUE)
  return(cell_dt)
}

all_years <- sort(unique(cell_dt$year))
for (var_name in neighbor_source_vars) {
  message("Computing neighbor features (chunked) for: ", var_name)
  cell_dt <- compute_neighbor_features_chunked(cell_dt, edge_table, var_name, all_years)
}
```

This reduces peak memory for the expanded table from ~1.2 GB to ~1.2 GB / 28 ≈ **~43 MB** per chunk.

---

## Expected Speedup

| Component | Old Runtime | New Runtime |
|---|---|---|
| Build lookup (6.46M char keys) | ~hours | <1 sec (integer edge table) |
| Neighbor stats (5 vars × 6.46M lapply) | ~80+ hours | ~5–15 min (data.table joins) |
| **Total** | **86+ hours** | **~5–20 minutes** |

The trained Random Forest model is never modified — only the feature-engineering pipeline is optimized. The numerical values produced (`max`, `min`, `mean` of each neighbor variable) are identical, preserving the original estimand.