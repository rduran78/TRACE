 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference — it is the two spatial neighbor feature construction stages:

### 1. `build_neighbor_lookup` — O(N) row-level `lapply` with per-row string pasting and named-vector lookups

For each of the **~6.46 million rows**, this function:
- Converts the cell ID to a reference index via a named character vector lookup.
- Retrieves neighbor cell IDs from the `nb` object.
- Pastes each neighbor cell ID with the current row's year to form string keys (one `paste` call per neighbor per row — roughly **1.37 million × 2 directions × 28 years ≈ tens of millions of string operations**).
- Performs named-vector lookup (`idx_lookup[neighbor_keys]`) on a **6.46 million-element named character vector** for every row. Named vector lookup in R is hash-based but still carries per-call overhead that is devastating at this scale.
- Returns an integer vector, per row.

**Cost:** ~6.46M iterations × (string construction + hash lookups per neighbor) ≈ many hours alone.

### 2. `compute_neighbor_stats` — O(N) row-level `lapply` with per-row subsetting, NA removal, and summary stats

For each of the ~6.46M rows, this extracts neighbor values, removes NAs, and computes `max`, `min`, `mean`. This is called **5 times** (once per source variable), yielding ~32.3M R-level iterations.

**Cost:** The `do.call(rbind, result)` on a 6.46M-element list of 3-vectors is also expensive (repeated memory allocation).

### Summary of root causes

| Cause | Impact |
|---|---|
| Row-level `lapply` over 6.46M rows (interpreted R loop) | Catastrophic |
| Per-row `paste()` string key construction | Very high |
| Per-row named-vector hash lookup on a 6.46M-key table | Very high |
| `do.call(rbind, ...)` on a 6.46M-element list | High |
| Repeated per-variable scan (5×) over the full lookup | High |
| No vectorization or join-based approach | Fundamental design issue |

**Estimated serial time:** 86+ hours is consistent with this analysis.

---

## Optimization Strategy

**Replace all row-level R loops with a single vectorized `data.table` equi-join + grouped aggregation.**

The key insight: the neighbor lookup is conceptually a **join**. For every `(cell_id, year)` row, we want the values of neighboring cells *in the same year*. This is a standard equi-join on `(neighbor_id, year)`, followed by a grouped aggregation (`max`, `min`, `mean`).

### Steps

1. **Build an edge table** (a two-column data.table of `id → neighbor_id`) from the `nb` object — done once, vectorized, ~1.37M rows.
2. **Join** the edge table with the panel data on `(neighbor_id, year)` to fetch neighbor values — a single `data.table` merge, fully vectorized in C.
3. **Aggregate** by `(id, year)` to get `max`, `min`, `mean` — a single `data.table` grouped operation per variable.
4. **Merge** the results back into the main data.

This eliminates all 6.46M-iteration R loops, all string key construction, and all per-row hash lookups.

**Expected speedup:** From 86+ hours to **minutes** (roughly 2–10 minutes depending on disk I/O and RAM pressure). All numerical results are preserved exactly.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 0 — Convert the nb object to a vectorized edge list (done once)
# ==============================================================================
build_edge_table <- function(id_order, neighbors) {
  # neighbors is a spdep::nb object: list of integer index vectors
  # id_order is the vector mapping positional index -> cell ID
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors)
  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

# ==============================================================================
# STEP 1 — Compute all neighbor features in a vectorized fashion
# ==============================================================================
compute_all_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                          neighbor_source_vars) {

  # --- Convert to data.table if needed (modifies in place for speed) ----------
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  # --- Build edge table once --------------------------------------------------
  edges <- build_edge_table(id_order, rook_neighbors_unique)

  # --- Ensure keys for fast joins ---------------------------------------------
  #     We will join edges with cell_data on (neighbor_id == id, year == year)
  #     so we need cell_data keyed by (id, year).
  setkey(cell_data, id, year)

  for (var_name in neighbor_source_vars) {
    message("Computing neighbor features for: ", var_name)

    # Subset only the columns we need for the join target
    # (id, year, and the current variable)
    val_dt <- cell_data[, .(id, year, val = get(var_name))]
    setkey(val_dt, id, year)

    # --- Join: for every (id, year) get all neighbor values -------------------
    #     edges has (id, neighbor_id).
    #     We add year via merge with cell_data's (id, year) universe,
    #     then look up neighbor values.
    #
    #     Efficient approach: expand edges × years using cell_data's own rows
    #     as the driver.

    # Driver: every (id, year) row with its neighbors
    # Merge cell_data's (id, year) with edges on id -> gives (id, year, neighbor_id)
    driver <- cell_data[, .(id, year)]
    setkey(driver, id)
    setkey(edges, id)
    expanded <- edges[driver, on = "id", allow.cartesian = TRUE, nomatch = 0L]
    # expanded columns: id, neighbor_id, year

    # Now look up the neighbor's value for that year
    setkey(expanded, neighbor_id, year)
    expanded[val_dt, on = c(neighbor_id = "id", "year"), neighbor_val := i.val]

    # --- Aggregate by (id, year) -----------------------------------------------
    stats <- expanded[!is.na(neighbor_val),
                      .(nbr_max  = max(neighbor_val),
                        nbr_min  = min(neighbor_val),
                        nbr_mean = mean(neighbor_val)),
                      by = .(id, year)]

    # Rename columns to match original naming convention
    max_col  <- paste0(var_name, "_max")
    min_col  <- paste0(var_name, "_min")
    mean_col <- paste0(var_name, "_mean")
    setnames(stats, c("nbr_max", "nbr_min", "nbr_mean"),
             c(max_col, min_col, mean_col))

    # --- Merge back into cell_data ---------------------------------------------
    # Remove old columns if they exist (idempotency)
    for (col in c(max_col, min_col, mean_col)) {
      if (col %in% names(cell_data)) cell_data[, (col) := NULL]
    }

    setkey(stats, id, year)
    setkey(cell_data, id, year)
    cell_data <- stats[cell_data, on = c("id", "year")]
    # Left join: NAs where a cell has no valid neighbors (matches original behavior)

    # Clean up to free RAM
    rm(expanded, stats, val_dt, driver)
    gc()
  }

  return(cell_data)
}

# ==============================================================================
# STEP 2 — Drop-in replacement for the original outer loop
# ==============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data              = cell_data,
  id_order               = id_order,
  rook_neighbors_unique  = rook_neighbors_unique,
  neighbor_source_vars   = neighbor_source_vars
)

# cell_data now has the same neighbor feature columns as the original pipeline.
# The trained Random Forest model can be applied directly to cell_data for prediction.
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` are computed on exactly the same set of non-NA neighbor values per `(id, year)` as the original code. The `data.table` join replicates the same `(cell_id, year) → neighbor_ids → same-year values` logic. |
| **NA handling** | `expanded[!is.na(neighbor_val), ...]` mirrors `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]`. Rows with no valid neighbors receive `NA` via the left join, matching `return(c(NA, NA, NA))`. |
| **Trained RF model** | No model retraining. The code only reconstructs the same feature columns that the model expects. |

---

## Performance Estimate

| Stage | Original | Optimized |
|---|---|---|
| Build neighbor lookup | ~hours (6.46M `lapply` + paste + hash) | ~seconds (vectorized `rep`/`unlist`, 1.37M rows) |
| Compute stats (×5 vars) | ~80+ hours (32.3M R-loop iterations) | ~2–8 minutes (5 × data.table join + grouped agg in C) |
| **Total** | **86+ hours** | **~3–10 minutes** |

Peak RAM usage will spike during the `allow.cartesian = TRUE` join (the expanded table has ~6.46M × avg. neighbors ≈ 26M rows per variable), but with 16 GB this is well within budget (~2–3 GB per iteration, freed each loop).