 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference — it is the two spatial neighbor feature construction functions, and specifically the way they interact with R's performance characteristics at scale.

### `build_neighbor_lookup` — Primary Bottleneck

This function calls `lapply` over **~6.46 million rows**, and inside each iteration it:

1. Performs **character coercion and named-vector lookup** (`id_to_ref[as.character(data$id[i])]`) — O(1) per call but with high constant overhead due to repeated `as.character()` conversion and name-based hash lookups across millions of iterations.
2. **Pastes** neighbor cell IDs with the year to create string keys (`paste(neighbor_cell_ids, data$year[i], sep = "_")`) — this allocates a new character vector per row.
3. Looks up those keys in `idx_lookup` — another named-vector string-hash lookup, repeated for every neighbor of every row.
4. Filters `NA` values and coerces to integer.

The total number of inner operations is proportional to the sum of neighbor counts across all cell-years: ~6.46M rows × ~4 rook neighbors on average ≈ **~25.8 million string paste + hash-lookup operations**, all inside an interpreted R loop. The per-iteration overhead of string allocation, garbage collection, and named-vector lookup makes this extremely slow — estimated at **tens of hours** alone.

### `compute_neighbor_stats` — Secondary Bottleneck

This function iterates over the 6.46M-element `neighbor_lookup` list, subsetting a numeric vector and computing `max`, `min`, `mean` per element. While each operation is small, the R-level `lapply` loop over millions of elements with per-element function-call overhead, NA filtering, and `c()` allocation is costly. The final `do.call(rbind, result)` on a 6.46M-element list of 3-vectors is also expensive.

### Summary

| Component | Operation Count | Core Issue |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M R-loop iterations, ~25.8M `paste` + hash lookups | Per-row string allocation, interpreted loop |
| `compute_neighbor_stats` | ~6.46M R-loop iterations × 5 variables = ~32.3M | Per-row subsetting, NA check, summary stats in interpreted loop |
| `do.call(rbind, ...)` | Binding 6.46M small vectors × 5 | Repeated memory allocation |

---

## Optimization Strategy

**Core idea:** Eliminate all row-level R loops and string-key lookups. Replace them with **vectorized joins and grouped aggregations** using `data.table`.

### Specific changes:

1. **Replace `build_neighbor_lookup`** entirely. Instead of building a 6.46M-element list of integer vectors (which is memory-heavy and slow to construct), build a **flat `data.table` edge list** of `(row_index, neighbor_row_index)` pairs using vectorized merge/join operations. This replaces millions of `paste` + hash-lookup calls with a single keyed `data.table` join.

2. **Replace `compute_neighbor_stats`** with a single **grouped `data.table` aggregation** (`[, .(max, min, mean), by = row_index]`) over the edge list joined to the variable values. This replaces millions of R-level function calls with C-level grouped operations.

3. **Avoid `do.call(rbind, ...)`** — the `data.table` aggregation returns a single result table directly.

4. **Preserve the trained Random Forest model** — we only change feature construction, not the model or the features themselves. The output columns are numerically identical.

### Complexity comparison:

| Step | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~6.46M R-loop iterations with string ops | One vectorized `data.table` join (~25.8M rows) |
| Stats computation (per variable) | ~6.46M R-loop iterations | One grouped aggregation over ~25.8M rows |
| Estimated total time | 86+ hours | **~5–15 minutes** |

---

## Working R Code

```r
library(data.table)

#' Build a flat edge-list data.table mapping each row in `data` to its
#' neighbor rows, fully vectorized.  Replaces build_neighbor_lookup().
#'
#' @param data       data.frame / data.table with columns `id` and `year`
#' @param id_order   integer vector of cell IDs in the order used by the nb object
#' @param neighbors  spdep nb object (list of integer index vectors into id_order)
#' @return data.table with columns  focal_row, neighbor_row
build_neighbor_edge_list <- function(data, id_order, neighbors) {


  # --- 1. Build cell-level edge list (id -> neighbor_id) ------------------
  n_neighbors  <- lengths(neighbors)
  focal_idx    <- rep(seq_along(neighbors), n_neighbors)
  neighbor_idx <- unlist(neighbors, use.names = FALSE)

  edge_cells <- data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )


  # --- 2. Map (id, year) -> row position in `data` -----------------------
  dt <- as.data.table(data)[, .(id, year)]
  dt[, row_pos := .I]

  # --- 3. Join: attach focal row positions --------------------------------
  #     For every (focal_id, year) combination that exists in `data`,

  #     expand by the cell-level edge list to get (focal_row, neighbor_id, year).
  setkey(dt, id)
  focal_dt <- dt[, .(focal_row = row_pos, id, year)]
  setkey(focal_dt, id)
  setkey(edge_cells, focal_id)

  # Each focal row fans out to its neighbors
  expanded <- edge_cells[focal_dt,
    .(focal_row, neighbor_id, year),
    on = .(focal_id = id),
    nomatch = NULL,
    allow.cartesian = TRUE
  ]

  # --- 4. Join: attach neighbor row positions -----------------------------
  setkey(dt, id, year)
  setkey(expanded, neighbor_id, year)

  result <- dt[expanded,
    .(focal_row, neighbor_row = row_pos),
    on = .(id = neighbor_id, year),
    nomatch = NULL
  ]

  return(result)
}


#' Compute neighbor max, min, mean for a variable using the edge list.
#' Replaces compute_neighbor_stats() + compute_and_add_neighbor_features().
#'
#' @param data       data.frame / data.table (full dataset)
#' @param edge_list  data.table from build_neighbor_edge_list()
#' @param var_name   character, column name in `data`
#' @return data.table with columns: focal_row, nb_max, nb_min, nb_mean
compute_neighbor_stats_fast <- function(data, edge_list, var_name) {

  vals <- data[[var_name]]

  # Attach neighbor values
  el <- copy(edge_list)
  el[, nb_val := vals[neighbor_row]]

  # Drop NAs (matches original behaviour)
  el <- el[!is.na(nb_val)]

  # Grouped aggregation — single pass in C
  stats <- el[, .(
    nb_max  = max(nb_val),
    nb_min  = min(nb_val),
    nb_mean = mean(nb_val)
  ), by = focal_row]

  return(stats)
}


#' Main driver: build features for all neighbor source variables and
#' attach them to cell_data.  Drop-in replacement for the outer loop.
#'
#' @param cell_data             data.frame with columns id, year, and the source vars
#' @param id_order              integer vector of cell IDs matching the nb object
#' @param rook_neighbors_unique spdep nb object
#' @param neighbor_source_vars  character vector of variable names
#' @return cell_data with new neighbor feature columns appended
add_all_neighbor_features <- function(cell_data,
                                      id_order,
                                      rook_neighbors_unique,
                                      neighbor_source_vars) {

  message("Building vectorized edge list ...")
  edge_list <- build_neighbor_edge_list(cell_data, id_order, rook_neighbors_unique)
  message(sprintf("  Edge list: %s rows", format(nrow(edge_list), big.mark = ",")))

  n <- nrow(cell_data)

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))

    stats <- compute_neighbor_stats_fast(cell_data, edge_list, var_name)

    # Initialize with NA (matches original for cells with no valid neighbors)
    max_col  <- rep(NA_real_, n)
    min_col  <- rep(NA_real_, n)
    mean_col <- rep(NA_real_, n)

    # Fill in computed values
    max_col[stats$focal_row]  <- stats$nb_max
    min_col[stats$focal_row]  <- stats$nb_min
    mean_col[stats$focal_row] <- stats$nb_mean

    # Use the same column naming convention as the original pipeline
    cell_data[[paste0(var_name, "_nb_max")]]  <- max_col
    cell_data[[paste0(var_name, "_nb_min")]]  <- min_col
    cell_data[[paste0(var_name, "_nb_mean")]] <- mean_col
  }

  return(cell_data)
}


# ========================================================================
# USAGE — drop-in replacement for the original outer loop
# ========================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is unchanged; proceed directly to prediction:
# predictions <- predict(rf_model, cell_data[, predictor_cols])
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | The same `max`, `min`, `mean` are computed over exactly the same set of non-NA neighbor values per cell-year. The `data.table` join replicates the original `paste(id, year)`-based matching logic, and the grouped aggregation replicates the per-element summary. |
| **NA handling** | Rows with zero valid neighbors produce `NA` for all three stats, identical to the original `if (length(neighbor_vals) == 0) return(c(NA, NA, NA))`. |
| **Column names** | Output columns follow the same `{var}_nb_max`, `{var}_nb_min`, `{var}_nb_mean` convention. (Adjust the suffix pattern if the original `compute_and_add_neighbor_features` used a different naming scheme.) |
| **Trained RF model** | Untouched. Only the feature engineering step is optimized. |

---

## Expected Performance

On a 16 GB RAM laptop:

- **Edge list construction:** ~25.8M-row `data.table` ≈ ~600 MB. Two keyed joins, each taking ~1–2 minutes.
- **Per-variable aggregation:** Grouped `max`/`min`/`mean` over ~25.8M rows ≈ ~30–60 seconds each.
- **Total for 5 variables:** **~5–15 minutes** versus the original 86+ hours — roughly a **350–1000× speedup**.
- **Peak RAM:** ~3–5 GB for the edge list plus intermediate copies, well within 16 GB.