 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It is the two spatial neighbor feature construction functions, and specifically the combination of:

1. **`build_neighbor_lookup`**: An `lapply` over ~6.46 million rows that, for each row, performs character coercion, string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is effectively a hash-table probe *per element*, but constructing ~6.46 million string keys and doing millions of named lookups is extremely slow in an interpreted row-wise loop.

2. **`compute_neighbor_stats`**: Another `lapply` over ~6.46 million rows computing `max`, `min`, `mean` on small vectors. This is called **5 times** (once per source variable), yielding ~32.3 million R-level function calls to summary statistics.

**Quantified cost of `build_neighbor_lookup`:**
- 6.46M iterations × (string paste + named vector subset) ≈ tens of hours alone.
- The string-keyed lookup `idx_lookup[neighbor_keys]` is O(k) per row where k = number of neighbors (≈4 for rook), but the constant factor of R-level string operations is enormous at this scale.

**Quantified cost of `compute_neighbor_stats`:**
- 5 variables × 6.46M rows × 3 summary stats = ~97M scalar computations wrapped in R `lapply` overhead.

**Root cause summary:** Row-level R loops with string manipulation and named-vector lookups over 6.46 million rows, repeated across 5 variables.

---

## Optimization Strategy

The strategy is to **eliminate all row-level R loops** by converting to **vectorized join and grouped aggregation** using `data.table`:

1. **Replace `build_neighbor_lookup`** with a single `data.table` equi-join. Pre-build an edge table (`cell_id` → `neighbor_cell_id`) and join it against the data keyed on `(id, year)`. This produces a tall table of (row_index, neighbor_row_index) pairs — no string keys, no `lapply`.

2. **Replace `compute_neighbor_stats`** with a single grouped `data.table` aggregation: group by the focal row index, compute `max`, `min`, `mean` of the neighbor values in one vectorized pass.

3. **Process all 5 variables** in a single pass over the neighbor-joined table (or 5 fast vectorized passes), eliminating redundant joins.

**Expected speedup:** From ~86+ hours to **minutes** (typically 5–15 minutes on a 16 GB laptop), because `data.table` joins and grouped aggregations are implemented in C and operate on integer keys without string construction.

**Numerical equivalence:** The operations (`max`, `min`, `mean` of the same neighbor sets) are identical, preserving the original estimand. The trained Random Forest model is untouched.

---

## Working R Code

```r
library(data.table)

#' Vectorized spatial neighbor feature construction.
#' Replaces build_neighbor_lookup + compute_neighbor_stats + outer loop.
#'
#' @param cell_data        data.frame or data.table with columns: id, year, and
#'                         all neighbor_source_vars columns.
#' @param id_order         integer vector of cell IDs in the order matching the
#'                         nb object (i.e., id_order[i] is the cell ID for the
#'                         i-th element of rook_neighbors_unique).
#' @param neighbors        spdep nb object (list of integer index vectors).
#' @param neighbor_source_vars character vector of variable names to summarize.
#' @return cell_data (data.table) with new columns appended:
#'         <var>_neighbor_max, <var>_neighbor_min, <var>_neighbor_mean
#'         for each var in neighbor_source_vars.

add_all_neighbor_features <- function(cell_data,
                                      id_order,
                                      neighbors,
                                      neighbor_source_vars) {

  # --- Step 0: Convert to data.table (by reference if already one) -----------
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  # Preserve original row order for downstream RF prediction
  cell_data[, .row_idx := .I]

  # --- Step 1: Build directed edge table (focal_id -> neighbor_id) -----------
  #
  # This replaces the per-row string-paste + named-vector lookup in

  # build_neighbor_lookup with a single vectorized construction.

  # Number of neighbors per node (0 for isolates encoded as list(0L) by spdep)
  n_neighbors <- vapply(neighbors, function(x) {
    if (length(x) == 1L && x[0 + 1] == 0L) 0L else length(x)
  }, integer(1))
  # Handle spdep convention: a neighbor list entry of integer(0) or 0L means

  # no neighbors.
  n_neighbors_safe <- vapply(neighbors, function(x) {
    nx <- x[x != 0L]
    length(nx)
  }, integer(1))

  focal_indices <- rep(seq_along(id_order), times = n_neighbors_safe)
  neighbor_indices <- unlist(lapply(neighbors, function(x) x[x != 0L]),
                             use.names = FALSE)

  edges <- data.table(
    focal_id    = id_order[focal_indices],
    neighbor_id = id_order[neighbor_indices]
  )

  # --- Step 2: Join edges with data to get neighbor row indices --------------
  #
  # Key idea: for every (focal_id, year) we need the data-row indices of

  # (neighbor_id, year).  We achieve this with two keyed joins — no strings.

  # Minimal keyed reference for focal rows: maps (id, year) -> .row_idx
  focal_ref <- cell_data[, .(id, year, .row_idx)]
  setkey(focal_ref, id, year)

  # Expand edges by year: each edge applies to every year present for the

  # focal cell.  Instead of a full cross-join (expensive in memory), we join
  # edges onto the data.


  # Join 1: attach focal .row_idx and year to each edge
  setkey(edges, focal_id)
  focal_years <- cell_data[, .(focal_id = id, year, focal_row = .row_idx)]
  setkey(focal_years, focal_id)

  # This is the big join: for every (edge × year-of-focal), get focal_row

  edge_year <- edges[focal_years, on = .(focal_id), allow.cartesian = TRUE,
                     nomatch = 0L]
  # edge_year columns: focal_id, neighbor_id, year, focal_row

  # Join 2: look up the neighbor's row index for the same year
  neighbor_ref <- cell_data[, .(neighbor_id = id, year, neighbor_row = .row_idx)]
  setkey(neighbor_ref, neighbor_id, year)
  setkey(edge_year, neighbor_id, year)

  edge_year <- neighbor_ref[edge_year, on = .(neighbor_id, year), nomatch = NA]
  # Keep only rows where the neighbor actually exists in that year
  edge_year <- edge_year[!is.na(neighbor_row)]

  # --- Step 3: Vectorized grouped aggregation per variable -------------------
  #
  # For each source variable, pull neighbor values via integer indexing
  # (vectorized), then group-by focal_row to compute max/min/mean.

  for (var_name in neighbor_source_vars) {
    vals_vec <- cell_data[[var_name]]

    # Attach neighbor values by integer index — fully vectorized
    edge_year[, nval := vals_vec[neighbor_row]]

    # Remove NAs in neighbor values before aggregation
    valid <- edge_year[!is.na(nval)]

    agg <- valid[, .(
      nmax  = max(nval),
      nmin  = min(nval),
      nmean = mean(nval)
    ), keyby = focal_row]

    # Initialize new columns to NA (handles rows with no valid neighbors)
    col_max  <- paste0(var_name, "_neighbor_max")
    col_min  <- paste0(var_name, "_neighbor_min")
    col_mean <- paste0(var_name, "_neighbor_mean")

    set(cell_data, j = col_max,  value = NA_real_)
    set(cell_data, j = col_min,  value = NA_real_)
    set(cell_data, j = col_mean, value = NA_real_)

    # Write aggregated values back by integer index — vectorized
    set(cell_data, i = agg$focal_row, j = col_max,  value = agg$nmax)
    set(cell_data, i = agg$focal_row, j = col_min,  value = agg$nmin)
    set(cell_data, i = agg$focal_row, j = col_mean, value = agg$nmean)
  }

  # Clean up helper column
  cell_data[, .row_idx := NULL]
  edge_year[, nval := NULL]  # free memory

  return(cell_data)
}


# ============================================================================
# USAGE — drop-in replacement for the original outer loop
# ============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  neighbors            = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Proceed directly to prediction:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Memory Budget Check

| Object | Approximate Size |
|---|---|
| `cell_data` (6.46M × 110 cols, numeric) | ~5.4 GB |
| `edge_year` (≈6.46M × 4 neighbors × 5 cols, integer) | ~1.5 GB |
| `agg` (temporary, ≤6.46M × 4 cols) | ~0.2 GB |
| Overhead / copies | ~2 GB |
| **Total peak** | **~9–10 GB** |

This fits within 16 GB RAM. If memory is tight, the `for` loop over the 5 variables already reuses the `nval` column in place, and `agg` is overwritten each iteration.

---

## Why This Preserves the Original Estimand

- The edge table encodes exactly the same directed rook-neighbor relationships as the original `nb` object.
- The `(focal_id, year)` → `(neighbor_id, year)` join is logically identical to the original string-key lookup.
- `max`, `min`, `mean` are computed on exactly the same neighbor value sets (after the same `!is.na` filter).
- The trained Random Forest model is never retrained — only the input feature matrix is reconstructed faster.
- Column names follow the same `<var>_neighbor_{max,min,mean}` convention assumed by the downstream model.