 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that **Random Forest inference** is the bottleneck is **incorrect**. The code evidence points squarely to the **spatial neighbor feature engineering** — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` — as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, performing per-row character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is effectively a hash-table probe *per element*, and with ~1.37 million neighbor relationships spread across 6.46M rows, this function alone generates tens of millions of string allocations and lookups. This is an **O(N × k)** operation done entirely in interpreted R with no vectorization.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time running `lapply` over 6.46 million rows, subsetting a numeric vector, removing NAs, and computing `max`, `min`, `mean`. That's **~32.3 million** interpreted R function calls across the 5 variables, each with small-vector allocation overhead.

3. **The outer loop** compounds this: 5 variables × 6.46M rows × per-row interpreted work = the dominant cost.

4. By contrast, **Random Forest prediction** on a pre-trained model is a single call to `predict()` on a matrix of ~6.46M × 110 features. Packages like `ranger` or `randomForest` execute this in compiled C/C++ code. Even on a laptop, this completes in seconds to a few minutes — negligible compared to 86+ hours.

**Verdict:** The bottleneck is the neighbor feature engineering, not RF inference. The interpreted, row-level R loops over millions of rows with string operations are the cause of the 86+ hour runtime.

---

## Optimization Strategy

1. **Eliminate `build_neighbor_lookup` entirely** in its current form. Replace per-row string-key lookups with a **vectorized integer-index approach** using `data.table`. Pre-build a mapping from `(id, year)` → row index as a keyed `data.table` for O(1) amortized joins.

2. **Explode the neighbor list into an edge table** — a two-column data.table of `(source_row, neighbor_row)` — built via a single vectorized merge/join rather than 6.46M `lapply` iterations.

3. **Replace `compute_neighbor_stats`** with a single **`data.table` grouped aggregation** (`j = .(max, min, mean), by = source_row`) on the edge table joined to the variable column. This replaces 6.46M `lapply` calls with one compiled C-level group-by.

4. **Process all 5 variables** by joining the edge table to each variable column and aggregating — each variable becomes one `data.table` operation instead of millions of R-level iterations.

This reduces the algorithmic work from **O(N × k) interpreted R calls with string allocation** to **O(E) vectorized C-level operations** (where E ≈ total neighbor-row edges), yielding an expected speedup from 86+ hours to **minutes**.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 0.  Convert working data to data.table (non-destructive; preserves
#     the original data.frame semantics downstream).
# ──────────────────────────────────────────────────────────────────────
cell_dt <- as.data.table(cell_data)
cell_dt[, row_idx := .I]                 # original row position

# ──────────────────────────────────────────────────────────────────────
# 1.  Build a fast (id, year) → row_idx lookup via keyed data.table
# ──────────────────────────────────────────────────────────────────────
id_year_map <- cell_dt[, .(id, year, row_idx)]
setkey(id_year_map, id, year)

# ──────────────────────────────────────────────────────────────────────
# 2.  Build the id → reference-index mapping (same as id_to_ref in the
#     original code, but as a simple integer vector for speed).
# ──────────────────────────────────────────────────────────────────────
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

# ──────────────────────────────────────────────────────────────────────
# 3.  Explode the nb object into a directed edge list of
#     (source_id, neighbor_id).  This is done once and is vectorized.
#
#     rook_neighbors_unique is a list of length length(id_order) where
#     element [[j]] contains integer indices into id_order for the
#     neighbors of id_order[j].
# ──────────────────────────────────────────────────────────────────────
n_neighbors   <- lengths(rook_neighbors_unique)          # integer vector
source_ids    <- rep(id_order, times = n_neighbors)      # vectorized rep
neighbor_ids  <- id_order[unlist(rook_neighbors_unique)] # vectorized index

edge_dt <- data.table(source_id = source_ids,
                      neighbor_id = neighbor_ids)

# ──────────────────────────────────────────────────────────────────────
# 4.  Expand edges across all 28 years to create
#     (source_row_idx, neighbor_row_idx) pairs.
#
#     Instead of looping over 6.46 M rows, we do two keyed joins.
# ──────────────────────────────────────────────────────────────────────
years_dt <- data.table(year = sort(unique(cell_dt$year)))

# Cross-join edges × years  (≈ 1.37 M edges × 28 years ≈ 38.5 M rows,
# but fits comfortably in 16 GB as integer columns).
edge_year <- edge_dt[, CJ_idx := 1L][              # dummy for cross join
    years_dt[, CJ_idx := 1L],
    on = "CJ_idx",
    allow.cartesian = TRUE
]
edge_year[, CJ_idx := NULL]

# Attach source row index
setkey(id_year_map, id, year)
edge_year[id_year_map, source_row := i.row_idx,
          on = .(source_id = id, year = year)]

# Attach neighbor row index
edge_year[id_year_map, neighbor_row := i.row_idx,
          on = .(neighbor_id = id, year = year)]

# Drop edges where either side has no matching row (boundary / NA)
edge_year <- edge_year[!is.na(source_row) & !is.na(neighbor_row)]

# Keep only the columns we need going forward
edge_year <- edge_year[, .(source_row, neighbor_row)]
setkey(edge_year, source_row)

# ──────────────────────────────────────────────────────────────────────
# 5.  Compute neighbor stats for every source variable in one
#     data.table group-by per variable — replaces the entire
#     compute_neighbor_stats + lapply loop.
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {

    # Pull the variable values aligned to neighbor rows
    edge_year[, val := cell_dt[[var_name]][neighbor_row]]

    # Grouped aggregation (compiled C-level in data.table)
    stats <- edge_year[!is.na(val),
                       .(nb_max  = max(val),
                         nb_min  = min(val),
                         nb_mean = mean(val)),
                       by = source_row]

    # Initialise new columns with NA, then fill matched rows
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")

    cell_dt[, (max_col)  := NA_real_]
    cell_dt[, (min_col)  := NA_real_]
    cell_dt[, (mean_col) := NA_real_]

    cell_dt[stats, (max_col)  := i.nb_max,  on = .(row_idx = source_row)]
    cell_dt[stats, (min_col)  := i.nb_min,  on = .(row_idx = source_row)]
    cell_dt[stats, (mean_col) := i.nb_mean, on = .(row_idx = source_row)]

    # Clean up the temporary column in edge_year
    edge_year[, val := NULL]
}

# ──────────────────────────────────────────────────────────────────────
# 6.  Convert back to data.frame to feed the existing RF predict() call.
#     The trained model and all original numeric values are preserved.
# ──────────────────────────────────────────────────────────────────────
cell_dt[, row_idx := NULL]
cell_data <- as.data.frame(cell_dt)

# ──────────────────────────────────────────────────────────────────────
# 7.  Random Forest inference (unchanged — this was never the
#     bottleneck).
# ──────────────────────────────────────────────────────────────────────
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M interpreted `lapply` iterations with string ops | Eliminated; replaced by one vectorized edge-table expansion + two keyed `data.table` joins |
| `compute_neighbor_stats` | 6.46M × 5 = 32.3M interpreted `lapply` calls | 5 single-pass `data.table` group-by aggregations (compiled C) |
| Estimated runtime | 86+ hours | Minutes (edge table build + 5 group-bys on ~38.5M rows) |
| RF inference | Unchanged (single `predict()` call) | Unchanged |
| Numerical results | Preserved (same max/min/mean logic, same NA handling) | Preserved identically |
| Trained RF model | Not modified | Not modified |