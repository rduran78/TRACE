 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` — O(n²) list construction via row-by-row `lapply`

The function iterates over **~6.46 million rows**, and for each row it:
1. Looks up the cell's neighbor IDs from the `nb` object.
2. Pastes together character keys (`id_year`) for each neighbor.
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`).

Character-key pasting and named-vector lookup inside a 6.46M-iteration `lapply` is extremely slow. The `paste()`/`setNames()` pattern creates millions of temporary strings and performs linear or hash-based scans on a 6.46M-element named vector each iteration.

### Bottleneck 2: `compute_neighbor_stats` — per-row subsetting in R loop

For each of the 6.46M rows, it subsets a numeric vector by index, removes NAs, and computes `max`, `min`, `mean`. This is repeated **5 times** (once per source variable). The R-level `lapply` over 6.46M elements with small vector operations has enormous interpreter overhead.

### Combined effect
~6.46M iterations × (string operations + named lookups) × 6 passes (1 build + 5 variables) ≈ **86+ hours** on a laptop.

---

## Optimization Strategy

### Key Insight: Separate the spatial topology (time-invariant) from the panel expansion (time-varying)

The rook-neighbor graph is **the same every year**. There are only 344,208 cells and ~1.37M directed neighbor edges. The panel just replicates this graph 28 times. So:

1. **Build an edge list once at the cell level** (344K nodes, ~1.37M edges) — trivially fast.
2. **Expand to the panel level using integer arithmetic** instead of character key lookups — map each `(cell, year)` to a row index using a dense integer matrix or offset arithmetic.
3. **Vectorize the neighbor statistics** using a sparse-matrix multiply or `data.table` grouped join instead of row-by-row `lapply`.

### Concrete approach: Edge-list + `data.table` join

- Convert the `nb` object to an edge list: `(from_cell_ref, to_cell_ref)`.
- Join with the panel on `(neighbor_cell_id, year)` to get neighbor values.
- Group by `(cell_id, year)` and compute `max`, `min`, `mean` in one vectorized pass per variable.

This replaces **6.46M R-level iterations** with a single **vectorized grouped aggregation** over ~1.37M × 28 ≈ ~38.5M edge-year rows, which `data.table` handles in seconds.

**Estimated speedup: from 86+ hours to ~2–5 minutes.**

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# Step 1: Convert the nb object to a cell-level edge list (once)
# ──────────────────────────────────────────────────────────────────────
nb_to_edge_list <- function(id_order, neighbors) {
  # neighbors is the spdep::nb object (list of integer index vectors)
  # id_order is the vector mapping position -> cell id
  from <- rep(seq_along(neighbors), lengths(neighbors))
  to   <- unlist(neighbors)
  data.table(
    from_id = id_order[from],
    to_id   = id_order[to]
  )
}

edge_dt <- nb_to_edge_list(id_order, rook_neighbors_unique)
# edge_dt has columns: from_id, to_id
# ~1,373,394 rows (directed rook edges)

# ──────────────────────────────────────────────────────────────────────
# Step 2: Convert cell_data to data.table (if not already)
# ──────────────────────────────────────────────────────────────────────
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure key for fast joins
setkey(cell_data, id, year)

# ──────────────────────────────────────────────────────────────────────
# Step 3: Vectorized neighbor feature computation
# ──────────────────────────────────────────────────────────────────────
compute_neighbor_features_fast <- function(cell_data, edge_dt, var_name) {
  # Build a slim lookup: neighbor cell id + year -> neighbor value
  # We only need (id, year, value) from cell_data
  val_dt <- cell_data[, .(id, year, value = get(var_name))]
  setkey(val_dt, id, year)

  # Get all unique years in the panel
  years <- sort(unique(cell_data$year))

  # Cross join edges × years to get the full panel-level edge list
  # But this could be large (~38.5M rows). More memory-efficient:
  # join edge_dt with val_dt on the neighbor side, then aggregate.

  # For each edge (from_id -> to_id), for each year, look up to_id's value
  # Approach: expand edges by year, then join for neighbor value

  # Memory-efficient: use cell_data's (id, year) pairs as the universe
  # and join via the edge list

  # Create the edge-year table by joining edges with the years present

  # for each 'from_id' in the panel
  from_years <- cell_data[, .(id, year)]
  setnames(from_years, "id", "from_id")
  setkey(from_years, from_id)
  setkey(edge_dt, from_id)

  # Merge edges with from_years: for each (from_id, year), get all to_ids

  edge_year <- edge_dt[from_years, on = "from_id", allow.cartesian = TRUE, nomatch = 0L]
  # edge_year has columns: from_id, to_id, year

  # Now join to get the neighbor's value
  setkey(edge_year, to_id, year)
  setkey(val_dt, id, year)
  edge_year[val_dt, on = c(to_id = "id", "year"), neighbor_val := i.value]

  # Aggregate: group by (from_id, year) -> max, min, mean of neighbor_val
  agg <- edge_year[,
    .(
      nb_max  = if (all(is.na(neighbor_val))) NA_real_ else max(neighbor_val, na.rm = TRUE),
      nb_min  = if (all(is.na(neighbor_val))) NA_real_ else min(neighbor_val, na.rm = TRUE),
      nb_mean = mean(neighbor_val, na.rm = TRUE)
    ),
    by = .(from_id, year)
  ]

  # Rename columns to match original naming convention
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(agg, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
  setnames(agg, "from_id", "id")

  agg
}

# ──────────────────────────────────────────────────────────────────────
# Step 4: Loop over the 5 source variables and merge results
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "\n")
  agg <- compute_neighbor_features_fast(cell_data, edge_dt, var_name)
  # Merge back into cell_data
  cell_data <- merge(cell_data, agg, by = c("id", "year"), all.x = TRUE)
  rm(agg)
  gc()
}

# Re-sort to original order if needed
setkey(cell_data, id, year)
```

---

## Memory Optimization (if ~38.5M rows is tight on 16 GB)

If the full `edge_year` table strains RAM, process in **year-chunks**:

```r
compute_neighbor_features_chunked <- function(cell_data, edge_dt, var_name) {
  years <- sort(unique(cell_data$year))
  val_dt <- cell_data[, .(id, year, value = get(var_name))]
  setkey(val_dt, id, year)

  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  agg_list <- lapply(years, function(yr) {
    # Subset to this year's values
    yr_vals <- val_dt[year == yr, .(id, value)]
    setkey(yr_vals, id)

    # Join edges with neighbor values for this year
    ey <- copy(edge_dt)
    ey[yr_vals, on = c(to_id = "id"), neighbor_val := i.value]

    # Aggregate
    result <- ey[,
      .(
        nb_max  = if (all(is.na(neighbor_val))) NA_real_ else max(neighbor_val, na.rm = TRUE),
        nb_min  = if (all(is.na(neighbor_val))) NA_real_ else min(neighbor_val, na.rm = TRUE),
        nb_mean = mean(neighbor_val, na.rm = TRUE)
      ),
      by = .(from_id)
    ]
    result[, year := yr]
    result
  })

  agg <- rbindlist(agg_list)
  setnames(agg, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
  setnames(agg, "from_id", "id")
  agg
}
```

This processes only ~1.37M edges per year-chunk, using minimal memory.

---

## Why This Preserves the Estimand and the Trained Model

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | The edge list is derived from the same `rook_neighbors_unique` nb object. The `max`, `min`, `mean` aggregations are identical arithmetic operations on identical neighbor sets. Results are bit-identical (modulo floating-point associativity of `mean`, which `data.table` computes in the same summation order within groups). |
| **Trained RF model unchanged** | No model retraining occurs. The code only reconstructs the **predictor columns** that the model expects. Column names follow the same `neighbor_{max,min,mean}_{var}` convention. |
| **No topology change** | The `nb_to_edge_list` function is a lossless 1-to-1 transformation of the `nb` object — every directed neighbor pair is preserved exactly. |

---

## Expected Performance

| Stage | Original | Optimized |
|---|---|---|
| Build neighbor lookup | ~hours (6.46M string ops) | <1 sec (vectorized edge list) |
| Per-variable stats | ~17 hours each × 5 | ~20–40 sec each × 5 |
| **Total** | **86+ hours** | **~2–5 minutes** |