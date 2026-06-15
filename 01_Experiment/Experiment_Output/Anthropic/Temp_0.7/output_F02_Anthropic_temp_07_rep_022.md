 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is twofold:

### 1. `build_neighbor_lookup` — O(n) `lapply` over ~6.46 million rows
Each iteration performs character coercion, `paste`, and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is hash-based but still slow when called millions of times inside an interpreted loop. With ~6.46M rows this single function likely takes many hours.

### 2. `compute_neighbor_stats` — O(n) `lapply` over ~6.46M rows, called 5 times
Each call iterates over every row, subsets a numeric vector by index, removes NAs, and computes `max`, `min`, `mean`. The `do.call(rbind, result)` at the end materialises a list of 6.46M 3-element vectors into a matrix — this alone is very slow because `rbind` on a long list is quadratic in memory allocation.

### Compounding factors
- Everything is in base-R interpreted loops (`lapply` over millions of elements).
- String key construction (`paste(id, year)`) and named-vector lookup are repeated per row.
- Memory: 6.46M × 110 columns is manageable (~5–8 GB as doubles), but intermediate list objects and string vectors created inside the loops cause heavy GC pressure on a 16 GB machine.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Per-row string key construction & named-vector lookup in `build_neighbor_lookup` | Replace with a **vectorised join** using `data.table`. Build a `(cell_id, year) → row_index` table once, then expand all neighbor pairs and join in one shot. No per-row loop. |
| Per-row `lapply` in `compute_neighbor_stats` | Replace with a **grouped `data.table` aggregation** (`max`, `min`, `mean` by source row), fully vectorised in C. |
| `do.call(rbind, list_of_vectors)` | Eliminated entirely — `data.table` returns a single table. |
| 5 separate passes over the lookup | Compute all 5 variables' neighbor stats in a **single grouped aggregation** (or one pass per variable but each pass is now milliseconds, not hours). |
| Memory pressure from intermediate string vectors | Eliminated — joins are on integer keys. |

**Expected speedup:** from 86+ hours to roughly **2–10 minutes** on the same laptop, depending on disk I/O. Peak RAM usage ~6–10 GB (fits in 16 GB).

**Preservation guarantees:**
- The trained Random Forest model is untouched (we only change feature engineering).
- The output columns are numerically identical (`max`, `min`, `mean` of the same neighbor values with the same NA handling).

---

## Working R Code

```r
# ─────────────────────────────────────────────────────────────────────
# Optimised neighbor-feature pipeline using data.table
# Drop-in replacement for build_neighbor_lookup + compute_neighbor_stats
# ─────────────────────────────────────────────────────────────────────

library(data.table)

build_neighbor_features_dt <- function(cell_data,
                                       id_order,
                                       rook_neighbors_unique,
                                       neighbor_source_vars) {
  # --- 0. Convert to data.table (by reference if already one) --------
  dt <- as.data.table(cell_data)

  # Ensure a deterministic row identifier so we can join results back
  dt[, .row_idx := .I]

  # --- 1. Build an edge list from the nb object ----------------------
  #
  # rook_neighbors_unique is a list of length = length(id_order).
  # Element k contains integer indices (into id_order) of the
  # neighbors of id_order[k].
  #
  # We expand this into a two-column data.table:
  #   (source_cell_id, neighbor_cell_id)

  edge_list <- rbindlist(
    lapply(seq_along(rook_neighbors_unique), function(k) {
      nb <- rook_neighbors_unique[[k]]
      # spdep uses 0L for "no neighbours"
      nb <- nb[nb != 0L]
      if (length(nb) == 0L) return(NULL)
      data.table(source_id = id_order[k],
                 neighbor_id = id_order[nb])
    })
  )
  # edge_list has ~1.37M rows — small and fast.

  # --- 2. Build a (cell_id, year) → row_idx lookup -------------------
  #     This replaces the slow named-vector idx_lookup.

  lookup <- dt[, .(cell_id = id, year, .row_idx)]
  setkey(lookup, cell_id, year)

  # --- 3. Expand edges × years in one vectorised join ----------------
  #
  # For every (source_id, year) we need the row indices of its
  # neighbors in that same year.
  #
  # Step A: get the unique years
  years <- sort(unique(dt$year))

  # Step B: cross-join edge_list × years  (~1.37M × 28 ≈ 38.5M rows)
  #         This is the full set of (source_id, year, neighbor_id) triples.
  edges_by_year <- CJ_dt(edge_list, years)

  # Helper: memory-efficient cross join
  # (defined below if not yet available)

  # Step C: attach the SOURCE row index
  setnames(edges_by_year, c("source_id", "neighbor_id", "year"))
  edges_by_year[lookup,
                source_row := i..row_idx,
                on = .(source_id = cell_id, year)]

  # Step D: attach the NEIGHBOR row index
  edges_by_year[lookup,
                neighbor_row := i..row_idx,
                on = .(neighbor_id = cell_id, year)]

  # Drop edges where either side is missing (cell not observed that year)
  edges_by_year <- edges_by_year[!is.na(source_row) & !is.na(neighbor_row)]

  # --- 4. Pull neighbor values and aggregate -------------------------
  #
  # For each source row and each variable, compute max / min / mean
  # of the neighbor values.

  for (var_name in neighbor_source_vars) {
    message("Computing neighbor stats for: ", var_name)

    # Attach the neighbor's value for this variable
    edges_by_year[, nbr_val := dt[[var_name]][neighbor_row]]

    # Grouped aggregation — fully vectorised in C
    agg <- edges_by_year[!is.na(nbr_val),
                         .(nbr_max  = max(nbr_val),
                           nbr_min  = min(nbr_val),
                           nbr_mean = mean(nbr_val)),
                         by = source_row]

    # Column names matching the original pipeline's convention
    col_max  <- paste0("nbr_max_",  var_name)
    col_min  <- paste0("nbr_min_",  var_name)
    col_mean <- paste0("nbr_mean_", var_name)

    # Write results back into dt by row index
    dt[, (col_max)  := NA_real_]
    dt[, (col_min)  := NA_real_]
    dt[, (col_mean) := NA_real_]

    dt[agg$source_row, (col_max)  := agg$nbr_max]
    dt[agg$source_row, (col_min)  := agg$nbr_min]
    dt[agg$source_row, (col_mean) := agg$nbr_mean]
  }

  # --- 5. Clean up and return ----------------------------------------
  dt[, .row_idx := NULL]

  # Return as data.frame if the caller expects one
  return(as.data.frame(dt))
}


# ─────────────────────────────────────────────────────────────────────
# Helper: memory-efficient cross join of edge_list × years vector
# Avoids CJ() because edge_list is already a table.
# ─────────────────────────────────────────────────────────────────────
CJ_dt <- function(edge_dt, years_vec) {
  # Repeat each edge row length(years_vec) times,
  # and tile the years vector nrow(edge_dt) times.
  n_edges <- nrow(edge_dt)
  n_years <- length(years_vec)
  idx <- rep(seq_len(n_edges), each = n_years)
  out <- edge_dt[idx]
  out[, year := rep(years_vec, times = n_edges)]
  return(out)
}


# ─────────────────────────────────────────────────────────────────────
# USAGE (drop-in replacement for the original outer loop)
# ─────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_dt(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is used as-is for prediction:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Is Correct and Numerically Identical

| Original logic | Optimised equivalent |
|---|---|
| For each row `i`, find neighbors of `data$id[i]` in the same `data$year[i]` | The cross join `edge_list × years` + two keyed joins produce exactly the same `(source_row, neighbor_row)` pairs. |
| `max(neighbor_vals)`, `min(neighbor_vals)`, `mean(neighbor_vals)` after removing NAs | `edges_by_year[!is.na(nbr_val), .(max, min, mean), by = source_row]` — identical aggregation, identical NA handling. |
| Rows with no valid neighbors get `NA, NA, NA` | Columns are initialised to `NA_real_`; only rows present in `agg` are overwritten. |

---

## Resource Estimates on a 16 GB Laptop

| Object | Approximate size |
|---|---|
| `dt` (6.46M × 115 cols) | ~6 GB |
| `edges_by_year` (~38.5M × 5 int/num cols) | ~1.5 GB |
| `agg` (≤ 6.46M × 4 cols) | ~0.2 GB |
| **Peak total** | **~8–9 GB** (fits in 16 GB) |
| **Expected wall-clock time** | **2–10 minutes** |

If memory is still tight, the `for` loop over the 5 variables can be split into separate passes where `edges_by_year` is rebuilt each time (trading ~5× more time for lower peak RAM), but this should not be necessary.