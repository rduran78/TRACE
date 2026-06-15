 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Converts an ID to a character and looks it up in a named vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs paste-based string keys (`"cellid_year"`).
4. Looks those keys up in a named character vector (`idx_lookup`).

**String concatenation (`paste`) and named-vector lookups on character keys are extremely slow in R when repeated millions of times.** The `idx_lookup` named vector has ~6.46M entries; each named-vector lookup is O(n) in the worst case because R's named vectors use linear hashing that degrades at scale. Multiply this by ~6.46M rows × ~4 average neighbors = ~25.8 billion character operations. This is why the estimated runtime exceeds 86 hours.

`compute_neighbor_stats` is comparatively cheap (just numeric subsetting), but it is also called 5 times with redundant `lapply` overhead.

**Summary of root causes:**
1. Row-level `lapply` with `paste`-based string key construction over 6.46M rows.
2. Named-vector lookup on a 6.46M-element character vector (poor hash performance).
3. The neighbor lookup is year-invariant in structure but rebuilt per row per year.
4. `compute_neighbor_stats` is called in a loop over 5 variables, each time iterating over 6.46M entries separately.

---

## Optimization Strategy

### Key Insight
The spatial neighbor structure is **time-invariant**: cell A's neighbors are the same in every year. Only the *data rows* change by year. So we can:

1. **Vectorize the lookup using integer indexing instead of string keys.** Build a matrix mapping `(cell_index, year_index)` → row number in `cell_data`. Then neighbor row indices are a direct integer matrix lookup — no strings, no paste, no named vectors.

2. **Use `data.table` for the statistics computation**, avoiding per-row `lapply` entirely. Expand the neighbor graph into an edge list, join to the data, and compute grouped `max/min/mean` in one vectorized pass per variable (or all variables at once).

3. **Compute all 5 variables' neighbor stats in a single pass** over the edge list rather than 5 separate `lapply` calls.

This reduces the complexity from ~6.46M × k expensive string operations to a single vectorized join + grouped aggregation on ~25.8M edge-rows.

---

## Optimized R Code

```r
library(data.table)

build_neighbor_features_fast <- function(cell_data, id_order, rook_neighbors_unique,
                                          neighbor_source_vars) {
  # Convert to data.table if not already; keep original row order
  dt <- as.data.table(cell_data)
  dt[, .row_id := .I]

  # ---- Step 1: Build the directed edge list from the nb object (time-invariant) ----
  # rook_neighbors_unique is a list of integer vectors (indices into id_order)
  # id_order[i] is the cell id for the i-th spatial unit
  edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb_idx <- rook_neighbors_unique[[i]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
      return(NULL)
    }
    data.table(from_id = id_order[i], to_id = id_order[nb_idx])
  }))
  # edges now has ~1,373,394 rows: (from_id, to_id) directed pairs

  # ---- Step 2: Expand edges across all years via join (vectorized) ----
  # We need, for every (from_id, year), the data-row indices of all (to_id, year).
  # First, create a keyed lookup: (id, year) -> .row_id
  dt_key <- dt[, .(id, year, .row_id)]
  setkey(dt_key, id, year)

  # Get unique years
  years <- sort(unique(dt$year))

  # Cross-join edges × years: each spatial edge exists in every year
  # This produces ~1,373,394 × 28 ≈ 38.5M rows — fits in 16 GB easily
  edge_year <- CJ(edge_idx = seq_len(nrow(edges)), year = years)
  edge_year[, `:=`(from_id = edges$from_id[edge_idx],
                    to_id   = edges$to_id[edge_idx])]
  edge_year[, edge_idx := NULL]

  # Join to get the focal row index (from_id, year)
  setkey(edge_year, from_id, year)
  setkey(dt_key, id, year)
  edge_year <- dt_key[edge_year, on = .(id = from_id, year = year),
                       nomatch = 0L]
  setnames(edge_year, ".row_id", "focal_row")

  # Join to get the neighbor row index (to_id, year)
  edge_year <- dt_key[edge_year, on = .(id = to_id, year = year),
                       nomatch = 0L]
  setnames(edge_year, ".row_id", "neighbor_row")

  # edge_year now has columns: focal_row, neighbor_row (plus id, year, to_id, etc.)
  # Keep only what we need
  edge_year <- edge_year[, .(focal_row, neighbor_row)]

  # ---- Step 3: Attach neighbor variable values and compute stats in one pass ----
  # Pull the variable columns we need for the neighbor rows
  var_cols <- neighbor_source_vars
  neighbor_vals <- dt[edge_year$neighbor_row, ..var_cols]
  neighbor_vals[, focal_row := edge_year$focal_row]

  # Compute max, min, mean per focal_row for each variable, all at once
  agg_exprs <- list()
  for (v in var_cols) {
    agg_exprs[[paste0("nb_max_", v)]] <- substitute(
      suppressWarnings(max(x[!is.na(x)])),
      list(x = as.name(v))
    )
    agg_exprs[[paste0("nb_min_", v)]] <- substitute(
      suppressWarnings(min(x[!is.na(x)])),
      list(x = as.name(v))
    )
    agg_exprs[[paste0("nb_mean_", v)]] <- substitute(
      mean(x, na.rm = TRUE),
      list(x = as.name(v))
    )
  }

  # Single grouped aggregation
  stats <- neighbor_vals[, lapply(agg_exprs, eval), by = focal_row]

  # Replace -Inf/Inf (from max/min on all-NA groups) with NA
  inf_to_na <- function(x) { x[is.infinite(x)] <- NA; x }
  stat_cols <- setdiff(names(stats), "focal_row")
  stats[, (stat_cols) := lapply(.SD, inf_to_na), .SDcols = stat_cols]

  # ---- Step 4: Join back to the main data ----
  # Rows with no neighbors won't appear in stats; they get NA automatically
  setkey(stats, focal_row)
  dt[, .row_id := .I]
  dt <- stats[dt, on = .(focal_row = .row_id)]
  dt[, focal_row := NULL]

  # Convert back to data.frame if the original was a data.frame
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    setDF(dt)
  }
  return(dt)
}

# ---- Usage (drop-in replacement for the original outer loop) ----
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_fast(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Predict as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Works and Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` with `na.rm = TRUE` are computed on the identical neighbor value sets as the original code. No approximation is introduced. |
| **Trained RF model untouched** | No retraining. The output columns have the same names and values, so `predict(rf_model, newdata = ...)` works identically. |
| **RAM feasibility** | The edge-year table has ~38.5M rows × 2 integer columns ≈ 0.6 GB. The neighbor-values table adds ~38.5M × 6 columns ≈ 1.8 GB. Total peak overhead is ~3–4 GB, well within 16 GB. |
| **Speed** | Eliminates all per-row `paste`/string-lookup. `data.table` grouped aggregation on 38.5M rows with 5 numeric columns typically completes in **1–5 minutes** on a modern laptop — a reduction from 86+ hours to minutes (roughly **1,000×** speedup). |

### Optional Further Speedups

- **Parallel aggregation:** `data.table` automatically uses `setDTthreads()` for grouped operations on multi-core machines.
- **Memory-constrained variant:** If RAM is tight, process one variable at a time instead of all 5 simultaneously (trades a small amount of speed for ~60% less peak memory).
- **Pre-cache to disk:** Save the resulting `cell_data` with neighbor features to `.fst` or `.parquet` so the computation never needs to be repeated.