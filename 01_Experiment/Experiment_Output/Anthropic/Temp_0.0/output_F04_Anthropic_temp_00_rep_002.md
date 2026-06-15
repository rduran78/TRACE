 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It is the two spatial neighbor feature construction functions, and the root causes are:

### 1. `build_neighbor_lookup` — Row-level `lapply` over 6.46 million rows

Each iteration performs:
- A character coercion and named-vector lookup (`id_to_ref[as.character(data$id[i])]`).
- A `paste()` call to build neighbor keys.
- A named-vector lookup into `idx_lookup` (which is a 6.46M-element named character vector — each lookup is O(n) hash probe repeated per neighbor).
- Subsetting and `is.na` filtering.

Doing this 6.46 million times in an interpreted `lapply` loop, with repeated `paste` and named-vector lookups on a vector of ~6.46M names, is extremely slow. Named vector lookup in R degrades as the vector grows; at 6.46M entries, each probe is expensive, and you do ~1.37M neighbor probes per year × 28 years.

### 2. `compute_neighbor_stats` — Row-level `lapply` over 6.46 million rows (×5 variables)

Each of the 6.46M iterations calls `vals[idx]`, removes NAs, and computes `max`, `min`, `mean`. This is called 5 times (once per variable). That is ~32.3 million R-level function invocations with per-row overhead.

### 3. `do.call(rbind, result)` on a 6.46M-element list of 3-vectors

This is a well-known R anti-pattern. Binding millions of small vectors is very slow.

**Estimated cost breakdown:**
| Step | Calls | Dominant cost |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `lapply` iterations | `paste`, named-vector hash lookups on 6.46M keys |
| `compute_neighbor_stats` | 6.46M × 5 = 32.3M iterations | per-row subsetting, NA removal, summary stats |
| `do.call(rbind, ...)` | 5 calls binding 6.46M rows | memory allocation / copying |

---

## Optimization Strategy

**Principle:** Replace row-level R loops with vectorized, column-level operations using `data.table` joins and grouped aggregation.

| Original approach | Optimized approach |
|---|---|
| Build a 6.46M-element named lookup vector, probe it row-by-row | Build an edge-list `data.table` and do a keyed equi-join |
| `lapply` over every row to gather neighbor indices | A single merge produces all (focal-row, neighbor-row) pairs |
| Per-row `max`/`min`/`mean` in R | Grouped `data.table` aggregation: `[, .(max, min, mean), by=focal_row]` |
| `do.call(rbind, ...)` on millions of tiny vectors | Result is already a `data.table`; assign columns directly |

**Expected speedup:** From ~86+ hours to roughly **5–15 minutes** on the same laptop, because:
- The join is O(E) where E ≈ 1.37M edges × 28 years ≈ 38.4M rows — large but handled in C by `data.table`.
- Grouped aggregation over 3 statistics × 5 variables is extremely fast in `data.table`.
- No R-level per-row interpretation overhead.

**Memory:** The edge table is ~38.4M rows × 2 integer columns ≈ 0.6 GB. Joined with one numeric variable at a time, peak overhead is ~1 GB, well within 16 GB.

**Numerical equivalence:** The optimized code computes exactly the same `max`, `min`, `mean` of the same neighbor values, preserving the original estimand. The trained Random Forest model is untouched.

---

## Working R Code

```r
library(data.table)

optimize_neighbor_features <- function(cell_data,
                                       id_order,
                                       rook_neighbors_unique,
                                       neighbor_source_vars) {

  # -----------------------------------------------------------
  # Step 1: Build a directed edge list from the nb object

  #         (done once; ~1.37M edges)
  # -----------------------------------------------------------
  edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb_idx <- rook_neighbors_unique[[i]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
      return(NULL)
    }
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb_idx])
  }))

  # -----------------------------------------------------------
  # Step 2: Convert cell_data to data.table (if not already)
  #         and create a row-index column for later assignment
  # -----------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, .row_idx := .I]

  # Minimal keyed table for joining: (id, year) -> row index + variable values
  # We will join variable-by-variable to limit peak memory.

  # -----------------------------------------------------------
  # Step 3: For each source variable, compute neighbor stats
  #         via a single keyed join + grouped aggregation
  # -----------------------------------------------------------
  # Prepare a small focal table: for every row, its (id, year, .row_idx)
  focal <- dt[, .(focal_id = id, year, .row_idx)]

  # Join focal rows to their neighbor cell IDs (cross-year broadcast)
  # focal_id -> neighbor_id via edges, keeping year from focal
  setkey(edges, focal_id)
  setkey(focal, focal_id)

  # Merge: each focal row gets its neighbor_ids (same year implied)
  # Result: one row per (focal_row, neighbor_cell) pair, carrying the year
  focal_neighbors <- edges[focal, on = .(focal_id),
                           allow.cartesian = TRUE,
                           nomatch = NULL]
  # Columns: focal_id, neighbor_id, year, .row_idx
  # .row_idx refers to the focal row in dt

  for (var_name in neighbor_source_vars) {

    message("Computing neighbor features for: ", var_name)

    # Build a lookup: (id, year) -> value
    val_table <- dt[, .(neighbor_id = id, year, .val = get(var_name))]
    setkey(val_table, neighbor_id, year)

    # Join neighbor values onto the edge table
    joined <- val_table[focal_neighbors, on = .(neighbor_id, year),
                        nomatch = NA]
    # Columns: neighbor_id, year, .val, focal_id, .row_idx

    # Drop rows where the neighbor value is NA (matches original logic)
    joined <- joined[!is.na(.val)]

    # Grouped aggregation by focal row
    stats <- joined[, .(nb_max  = max(.val),
                        nb_min  = min(.val),
                        nb_mean = mean(.val)),
                    by = .(.row_idx)]

    # Initialise new columns to NA (rows with no valid neighbors stay NA)
    max_col  <- paste0("nb_max_",  var_name)
    min_col  <- paste0("nb_min_",  var_name)
    mean_col <- paste0("nb_mean_", var_name)

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Assign computed stats back by row index
    dt[stats$.row_idx, (max_col)  := stats$nb_max]
    dt[stats$.row_idx, (min_col)  := stats$nb_min]
    dt[stats$.row_idx, (mean_col) := stats$nb_mean]

    # Free memory
    rm(val_table, joined, stats)
  }

  # Clean up helper column
  dt[, .row_idx := NULL]

  # Return as data.frame if the downstream RF predict() expects one
  return(as.data.frame(dt))
}

# -----------------------------------------------------------
# Usage (drop-in replacement for the original outer loop)
# -----------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is unchanged.
# Proceed directly to prediction:
# cell_data$predicted_gdp <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M-iteration `lapply` with `paste` + named-vector probes | One `rbindlist` to build ~1.37M-row edge `data.table` |
| **Neighbor stat computation** | 6.46M-iteration `lapply` × 5 vars | One keyed `data.table` join (cartesian, ~38.4M rows) + grouped `[, .(max,min,mean), by=.row_idx]` per variable |
| **Result assembly** | `do.call(rbind, <6.46M-element list>)` × 5 | Direct column assignment by integer index |
| **Estimated wall time** | 86+ hours | ~5–15 minutes |
| **Peak RAM delta** | Moderate (millions of small R objects + GC pressure) | ~1–2 GB (dense edge table + one joined column at a time) |
| **Numerical result** | `max`, `min`, `mean` of non-NA rook-neighbor values | Identical |
| **Trained RF model** | — | Untouched |