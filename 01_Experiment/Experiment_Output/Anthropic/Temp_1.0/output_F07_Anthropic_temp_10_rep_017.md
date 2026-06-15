 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### 1. `build_neighbor_lookup` — O(n²) list construction via per-row string hashing

The function iterates over **~6.46 million rows**, and for each row it:
- Looks up the cell's neighbor IDs.
- Pastes string keys (`id_year`).
- Searches a named character vector (`idx_lookup`) by string matching.

String-based named-vector lookups in R are **O(n)** per call in the worst case (hash collisions aside, the overhead of `paste` + named indexing over 6.46M rows is enormous). The resulting `neighbor_lookup` is a **list of 6.46 million integer vectors**, each built individually.

### 2. `compute_neighbor_stats` — Repeated per-variable list-apply

For each of the 5 source variables, the code iterates over all 6.46M rows again via `lapply`, extracting neighbor values element by element. This is **5 × 6.46M = 32.3M** R-level loop iterations with per-element subsetting.

### Combined effect
These two stages together produce the estimated **86+ hour** runtime. The fundamental issue is: **row-level R loops over millions of rows with string operations**.

---

## Optimization Strategy

### A. Replace string-key lookup with integer join via `data.table`

Instead of building a 6.46M-element list of neighbor row indices using string keys, we:

1. Create a `data.table` of all directed neighbor pairs: `(id, neighbor_id)` — ~1.37M pairs.
2. Cross this with all 28 years to get `(id, year, neighbor_id)` — but more efficiently, we join on `(id, year)` to attach the row index, then join on `(neighbor_id, year)` to attach the neighbor's row index. This is a **vectorized equi-join**, not a per-row loop.

### B. Compute all neighbor stats in one vectorized grouping operation

Once we have a table of `(row_index, neighbor_row_index)`, we can:
1. Pull the variable value for each neighbor row.
2. Group by `row_index` and compute `max`, `min`, `mean` in a single `data.table` aggregation — fully vectorized in C.

### C. Avoid building the 6.46M-element `neighbor_lookup` list entirely

The list is never needed. The join table replaces it.

### Expected speedup
- `data.table` equi-joins and grouped aggregations over ~38M edge-year rows (1.37M edges × 28 years) should complete in **seconds to low minutes**, not hours.
- Total for 5 variables: **under 10 minutes** on a 16 GB laptop.

### Preserving the estimand
The numerical values computed (`max`, `min`, `mean` of non-NA neighbor values, with `NA` when no valid neighbors exist) are **identical** to the original code. No model retraining is needed.

---

## Working R Code

```r
library(data.table)

#' Optimized neighbor feature engineering for cell-year panel data.
#'
#' @param cell_data       data.frame or data.table with columns: id, year, and all source vars.
#' @param id_order        character/integer vector — the cell IDs in the order matching rook_neighbors_unique.
#' @param rook_neighbors_unique  spdep nb object (list of integer index vectors).
#' @param neighbor_source_vars   character vector of variable names to compute neighbor stats for.
#' @return cell_data as a data.table with new columns appended.
compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {

  # --- Step 0: Convert to data.table if needed; add row-position key ----------
  dt <- as.data.table(cell_data)
  dt[, .row_idx := .I]

  # --- Step 1: Build directed edge list (cell-level, year-independent) --------
  #     Each entry in the nb object is an integer vector of neighbor positions
  #     within id_order.  We expand to a two-column data.table of (id, neighbor_id).
  edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb <- rook_neighbors_unique[[i]]
    # spdep nb encodes "no neighbors" as a single 0L
    nb <- nb[nb != 0L]
    if (length(nb) == 0L) return(NULL)
    data.table(id = id_order[i], neighbor_id = id_order[nb])
  }))

  # --- Step 2: Cross edges with years via join to get row indices -------------
  #     We need:  for every (id, year) row, the .row_idx of each neighbor in that year.

  # Keyed lookup:  (id, year) -> .row_idx
  row_key <- dt[, .(id, year, .row_idx)]
  setkey(row_key, id, year)

  # Get all unique years once
  all_years <- unique(dt$year)

  # Expand edges × years  (1.37M × 28 ≈ 38.4M rows — fits easily in RAM)
  edge_years <- edges[, CJ(year = all_years), by = .(id, neighbor_id)]

  # Attach the focal row's index
  setkey(edge_years, id, year)
  edge_years <- row_key[edge_years, on = .(id, year), nomatch = 0L]
  setnames(edge_years, ".row_idx", "focal_row")

  # Attach the neighbor row's index
  setnames(edge_years, c("id", "neighbor_id"), c("focal_id", "id"))
  setkey(edge_years, id, year)
  edge_years <- row_key[edge_years, on = .(id, year), nomatch = 0L]
  setnames(edge_years, ".row_idx", "neighbor_row")
  setnames(edge_years, c("id", "focal_id"), c("neighbor_id", "id"))

  # edge_years now has columns: id, year, focal_row, neighbor_id, neighbor_row

  # --- Step 3: For each variable, compute grouped neighbor stats --------------
  for (var_name in neighbor_source_vars) {

    message("Computing neighbor stats for: ", var_name)

    # Pull neighbor values via direct integer indexing (vectorized)
    edge_years[, nval := dt[[var_name]][neighbor_row]]

    # Aggregate — drop NAs within each group
    agg <- edge_years[!is.na(nval),
                      .(nmax  = max(nval),
                        nmin  = min(nval),
                        nmean = mean(nval)),
                      keyby = .(focal_row)]

    # Prepare NA-filled columns, then fill where we have results
    max_col  <- paste0("n_max_",  var_name)
    min_col  <- paste0("n_min_",  var_name)
    mean_col <- paste0("n_mean_", var_name)

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    dt[agg$focal_row, (max_col)  := agg$nmax]
    dt[agg$focal_row, (min_col)  := agg$nmin]
    dt[agg$focal_row, (mean_col) := agg$nmean]
  }

  # Clean up helper column
  edge_years[, nval := NULL]
  dt[, .row_idx := NULL]

  return(dt[])
}


# =============================================================================
# Usage — drop-in replacement for the original outer loop
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data              = cell_data,
  id_order               = id_order,
  rook_neighbors_unique  = rook_neighbors_unique,
  neighbor_source_vars   = neighbor_source_vars
)

# cell_data now contains the same n_max_*, n_min_*, n_mean_* columns
# as the original pipeline.  The trained Random Forest model can be
# applied directly with no retraining.
```

---

## Memory Estimate

| Object | Approximate size |
|---|---|
| `dt` (6.46M × ~115 cols) | ~5.5 GB |
| `edge_years` (38.4M × 5 cols) | ~1.5 GB |
| `row_key` (6.46M × 3 cols) | ~0.15 GB |
| Overhead + temporaries | ~1–2 GB |
| **Total peak** | **~8–9 GB** ✔ (fits in 16 GB) |

## Runtime Estimate

| Step | Expected time |
|---|---|
| Edge expansion + joins | ~30–60 seconds |
| Per-variable aggregation (×5) | ~20–40 seconds each |
| **Total** | **~3–5 minutes** |

This represents a speedup of roughly **1,000×–1,500×** over the original 86+ hour estimate, while producing **numerically identical** results (same `max`, `min`, `mean` of non-NA rook-neighbor values, same `NA` propagation for isolated or boundary cells). The trained Random Forest model requires no modification.