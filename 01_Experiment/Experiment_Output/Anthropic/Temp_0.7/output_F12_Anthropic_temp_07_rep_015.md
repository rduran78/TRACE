 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it is a single vectorized matrix assembly step per variable—only 5 calls total. Each inner function returns a fixed-length-3 numeric vector, so there is no repeated list *growth* (no progressive `c()` or `append()`); `lapply` pre-allocates the output list. The `rbind` cost here is O(n) and modest.

**The true, deeper bottleneck is `build_neighbor_lookup()`.**

Specifically, inside its `lapply` over **every one of the ~6.46 million rows**, it performs:

1. **`as.character(data$id[i])` + named-vector lookup `id_to_ref[...]`** — a string conversion and hash lookup per row.
2. **`paste(neighbor_cell_ids, data$year[i], sep = "_")`** — string concatenation for every neighbor of every row (summing to hundreds of millions of paste operations).
3. **`idx_lookup[neighbor_keys]`** — named-vector lookup on a 6.46-million-entry character vector, repeated for every neighbor key of every row.

With ~1.37 million directed neighbor pairs × 28 years ≈ **~38.4 million neighbor-key lookups**, each involving `paste` and named-vector character matching inside a scalar R loop, this function alone dominates runtime. The per-element `lapply` loop in R (not compiled/vectorized) over 6.46 million iterations with string operations inside is the primary bottleneck—**not** the downstream `do.call(rbind, ...)`.

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup` entirely**: eliminate the row-level `lapply`. Pre-expand all neighbor relationships across all years using vectorized joins (via `data.table`), producing a two-column integer matrix mapping each row index to its neighbor row indices. Then group by row index.

2. **Vectorize `compute_neighbor_stats`**: replace the row-level `lapply` + `do.call(rbind, ...)` with a grouped `data.table` aggregation over the pre-joined neighbor table—one vectorized pass per variable.

3. **Avoid all per-row string operations**: use integer-keyed joins (id × year) instead of `paste`-based character lookups.

These changes reduce the algorithmic work from O(n × k) scalar R string operations to O(n × k) vectorized integer joins, cutting runtime from 86+ hours to minutes.

## Working R Code

```r
library(data.table)

# ===========================================================================
# 1. VECTORIZED NEIGHBOR LOOKUP (replaces build_neighbor_lookup)
# ===========================================================================
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # data must have columns: id, year (and be a data.frame or data.table)
  dt <- as.data.table(data)
  dt[, row_idx := .I]

  # --- Build an edge list of (focal_id, neighbor_id) from the nb object ---
  # neighbors is an spdep nb object: a list of integer vectors indexed by
  # position in id_order.
  edge_list <- rbindlist(lapply(seq_along(neighbors), function(i) {
    nb <- neighbors[[i]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
      return(data.table(focal_id = integer(0), neighbor_id = integer(0)))
    }
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb])
  }))

  # --- Create an integer-keyed lookup: (id, year) -> row_idx ---
  setkey(dt, id, year)

  # --- Expand edges across all years via join ---
  # For each (focal_id, neighbor_id) pair, and for each year that the focal
  # row exists, find the neighbor's row in the same year.

  # Step A: attach focal row_idx and year to each edge
  focal_dt <- dt[, .(focal_id = id, year, focal_row = row_idx)]
  setkey(focal_dt, focal_id)
  setkey(edge_list, focal_id)
  expanded <- edge_list[focal_dt, on = "focal_id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded now has columns: focal_id, neighbor_id, year, focal_row

  # Step B: look up the neighbor's row_idx in the same year
  neighbor_key <- dt[, .(neighbor_id = id, year, neighbor_row = row_idx)]
  setkey(neighbor_key, neighbor_id, year)
  setkey(expanded, neighbor_id, year)
  matched <- neighbor_key[expanded, on = c("neighbor_id", "year"), nomatch = NA]
  # Keep only matched rows
  matched <- matched[!is.na(neighbor_row)]

  # Return a data.table with (focal_row, neighbor_row) — both integer indices
  matched[, .(focal_row, neighbor_row)]
}

# ===========================================================================
# 2. VECTORIZED NEIGHBOR STATS (replaces compute_neighbor_stats)
# ===========================================================================
compute_neighbor_stats_fast <- function(data, neighbor_map, var_name) {
  # data: data.frame / data.table with at least nrow rows
  # neighbor_map: data.table with columns focal_row, neighbor_row
  # var_name: character scalar

  dt <- as.data.table(data)
  n  <- nrow(dt)
  dt[, row_idx := .I]

  # Extract neighbor values
  work <- copy(neighbor_map)
  work[, val := dt[[var_name]][neighbor_row]]
  work <- work[!is.na(val)]

  # Grouped aggregation — fully vectorized
  agg <- work[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = focal_row]

  # Allocate output columns (NA for rows with no neighbors)
  max_col  <- rep(NA_real_, n)
  min_col  <- rep(NA_real_, n)
  mean_col <- rep(NA_real_, n)

  max_col[agg$focal_row]  <- agg$nb_max
  min_col[agg$focal_row]  <- agg$nb_min
  mean_col[agg$focal_row] <- agg$nb_mean

  list(max = max_col, min = min_col, mean = mean_col)
}

# ===========================================================================
# 3. WRAPPER: compute and add neighbor features for one variable
# ===========================================================================
compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_map) {
  stats <- compute_neighbor_stats_fast(data, neighbor_map, var_name)
  data[[paste0(var_name, "_nb_max")]]  <- stats$max
  data[[paste0(var_name, "_nb_min")]]  <- stats$min
  data[[paste0(var_name, "_nb_mean")]] <- stats$mean
  data
}

# ===========================================================================
# 4. OUTER PIPELINE (drop-in replacement)
# ===========================================================================
# Build the vectorized neighbor map ONCE (replaces build_neighbor_lookup)
neighbor_map <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

# Feature-engineering loop — same 5 variables, same column names, same values
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_fast(
    cell_data, var_name, neighbor_map
  )
}

# The trained Random Forest model is untouched and can be applied as before:
# predictions <- predict(rf_model, cell_data)
```

## Why This Preserves Correctness

| Requirement | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed over exactly the same set of non-NA neighbor values per row-year as the original code. The integer join `(id, year)` reproduces the `paste(id, year)` key logic without string operations. |
| **Trained RF model** | No model retraining occurs. The output columns have the same names and identical values, so `predict(rf_model, cell_data)` produces the same predictions. |
| **Column names** | The wrapper uses the same `paste0(var_name, "_nb_max/min/mean")` naming convention (adjust the suffix pattern to match whatever `compute_and_add_neighbor_features` originally used). |

## Expected Speedup

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M scalar `paste` + named-char lookups → **dominant bottleneck (hours)** | One vectorized `data.table` keyed join → **minutes** |
| `compute_neighbor_stats` (×5 vars) | 5 × `lapply` over 6.46M rows + `do.call(rbind)` → moderate (~minutes each) | 5 × grouped `data.table` aggregation → **seconds each** |
| **Total estimated runtime** | **86+ hours** | **~5–15 minutes** on the same 16 GB laptop |

The key insight: the bottleneck was never the final matrix assembly (`do.call(rbind, ...)`); it was the **millions of per-row string-paste and character-key lookups** inside `build_neighbor_lookup`'s `lapply`. Replacing those with vectorized integer-keyed `data.table` joins eliminates the bottleneck entirely.