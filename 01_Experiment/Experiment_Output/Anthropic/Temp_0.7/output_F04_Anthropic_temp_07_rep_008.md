 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference — it is the two spatial neighbor feature construction functions, and specifically the interaction between `build_neighbor_lookup` and `compute_neighbor_stats`, both of which are implemented as **row-level `lapply` loops over ~6.46 million rows** with per-iteration string operations.

### `build_neighbor_lookup` — Primary Bottleneck

For each of the ~6.46M rows, this function:

1. **Converts an integer ID to character and looks it up in a named vector** (`id_to_ref`): O(1) amortized, but with per-call character coercion overhead × 6.46M.
2. **Indexes into the `neighbors` list** to retrieve neighbor cell IDs: fast.
3. **Pastes neighbor cell IDs with the current year** (`paste(..., sep="_")`): this creates a character vector of length ≈ average_degree (≈4 for rook) per row. Over all rows this is ~25.8M string constructions.
4. **Looks up each pasted key in `idx_lookup`**, a named character vector of length 6.46M: named-vector lookup in R is **O(n) per probe** unless R internally hashes it (it does for `[.character` on named vectors, but the hash table is rebuilt or probed repeatedly). Still, ~25.8M hash probes into a 6.46M-entry table is expensive in R's interpreted loop.

The entire function wraps this in `lapply` over 6.46M iterations. **R-level `lapply` with non-trivial closures over millions of iterations is extremely slow** — estimated at 40–60+ hours alone.

### `compute_neighbor_stats` — Secondary Bottleneck

Another `lapply` over 6.46M rows, each computing `max`, `min`, `mean` on a small vector (~4 elements). The per-call overhead of R function dispatch dominates. This is called **5 times** (once per source variable), contributing another 20+ hours.

### Summary of Root Causes

| Cause | Location | Impact |
|---|---|---|
| 6.46M-iteration R-level `lapply` with string ops | `build_neighbor_lookup` | ~40–60 hrs |
| `paste()` + named-vector lookup per row | `build_neighbor_lookup` | Major |
| 6.46M-iteration `lapply` × 5 variables | `compute_neighbor_stats` | ~20–30 hrs |
| No vectorization or use of integer arithmetic | Both functions | Fundamental |

---

## Optimization Strategy

### Core Idea: Replace string-key lookups with integer-indexed joins using `data.table`, and vectorize the neighbor stats computation.

**Three-part plan:**

1. **Replace `build_neighbor_lookup`** with a fully vectorized `data.table` equi-join. Instead of building a per-row list of neighbor row indices via string pasting and lookup, we:
   - Expand the neighbor graph into an edge-list `(cell_id, neighbor_id)`.
   - Join with the panel data on `(neighbor_id, year)` to get neighbor row indices.
   - Group by source-row index to collect neighbor row indices.
   
   This replaces 6.46M R-level iterations with a single `data.table` merge (~25.8M rows joined against a 6.46M-row keyed table — seconds, not hours).

2. **Replace `compute_neighbor_stats`** with a single grouped `data.table` aggregation per variable. Instead of `lapply` over rows, we:
   - Build a long table of `(source_row, neighbor_value)`.
   - Compute `max`, `min`, `mean` grouped by `source_row`.
   
   This replaces 5 × 6.46M R-level iterations with 5 vectorized group-by operations.

3. **Memory management**: The edge-list expansion (~25.8M rows × a few integer columns) fits easily in 16 GB. We process one variable at a time and discard intermediates.

**Expected speedup**: From 86+ hours to **~5–15 minutes** total.

**Numerical equivalence**: The `max`, `min`, and `mean` computations are identical; only the iteration mechanism changes. The trained Random Forest model is untouched.

---

## Working R Code

```r
library(data.table)

#' Optimized spatial neighbor feature construction.
#' Drop-in replacement for the original build_neighbor_lookup +
#' compute_neighbor_stats + outer-loop pipeline.
#'
#' @param cell_data        data.frame (or data.table) with columns: id, year,
#'                         and all neighbor_source_vars columns.
#' @param id_order         integer vector: the cell IDs in the order matching
#'                         the spdep::nb object indices.
#' @param rook_neighbors   spdep::nb list (rook_neighbors_unique).
#' @param neighbor_source_vars character vector of variable names to summarize.
#' @return data.table with original columns plus neighbor feature columns.
build_neighbor_features_fast <- function(cell_data,
                                         id_order,
                                         rook_neighbors,
                                         neighbor_source_vars) {

  # --- Step 0: Convert to data.table if needed; add row index ---
  dt <- as.data.table(cell_data)
  dt[, .row_idx := .I]

  # --- Step 1: Build the directed edge list from the nb object ---
  # rook_neighbors is a list of length = length(id_order).
  # rook_neighbors[[i]] contains integer indices (into id_order) of neighbors
  # of cell id_order[i].  0L entries mean no neighbors (spdep convention).
  message("Building edge list from nb object...")

  edge_list <- rbindlist(lapply(seq_along(rook_neighbors), function(i) {
    nb_idx <- rook_neighbors[[i]]
    # spdep::nb encodes "no neighbors" as a single 0L
    nb_idx <- nb_idx[nb_idx != 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(cell_id     = id_order[i],
               neighbor_id = id_order[nb_idx])
  }))
  # edge_list has ~1.37M rows (directed rook edges)

  message(sprintf("Edge list: %s directed edges.", format(nrow(edge_list), big.mark = ",")))

  # --- Step 2: Create a keyed lookup from (id, year) -> row index ---
  # This replaces the string-pasted idx_lookup.
  row_key <- dt[, .(id, year, .row_idx)]
  setkey(row_key, id, year)

  # --- Step 3: For each source row, find all (source_row_idx, neighbor_row_idx) pairs ---
  # Join edge_list with dt to get the year dimension.
  # For every row in dt, we know its cell_id and year.
  # Its neighbors are given by edge_list where edge_list$cell_id == dt$id.
  # The neighbor rows are those with (edge_list$neighbor_id, same year).

  message("Joining edges with panel years to build full neighbor map...")

  # Source rows: (cell_id, year, source_row_idx)
  source_rows <- dt[, .(cell_id = id, year, src_idx = .row_idx)]
  setkey(source_rows, cell_id)

  # Merge source rows with edge list on cell_id to get:
  # (cell_id, year, src_idx, neighbor_id)
  setkey(edge_list, cell_id)
  expanded <- edge_list[source_rows, on = "cell_id",
                        .(year       = i.year,
                          src_idx    = i.src_idx,
                          neighbor_id = x.neighbor_id),
                        allow.cartesian = TRUE,
                        nomatch = 0L]
  # expanded has ~6.46M * avg_degree ≈ 25.8M rows

  message(sprintf("Expanded neighbor-year pairs: %s rows.",
                  format(nrow(expanded), big.mark = ",")))

  # Now look up the neighbor's row index for (neighbor_id, year)
  setkey(expanded, neighbor_id, year)
  expanded[row_key, on = .(neighbor_id = id, year = year),
           nbr_idx := i..row_idx]

  # Drop rows where the neighbor doesn't exist in the panel (boundary / NA)
  expanded <- expanded[!is.na(nbr_idx)]

  message(sprintf("Valid neighbor links after join: %s.",
                  format(nrow(expanded), big.mark = ",")))

  # --- Step 4: Compute neighbor stats per variable, vectorized ---
  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))

    # Attach neighbor values
    expanded[, nbr_val := dt[[var_name]][nbr_idx]]

    # Remove NAs in neighbor values for aggregation
    valid <- expanded[!is.na(nbr_val)]

    # Grouped aggregation
    agg <- valid[, .(
      nb_max  = max(nbr_val),
      nb_min  = min(nbr_val),
      nb_mean = mean(nbr_val)
    ), by = src_idx]

    # Initialize columns with NA
    max_col  <- paste0("nb_max_",  var_name)
    min_col  <- paste0("nb_min_",  var_name)
    mean_col <- paste0("nb_mean_", var_name)

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Assign results by row index
    dt[agg$src_idx, (max_col)  := agg$nb_max]
    dt[agg$src_idx, (min_col)  := agg$nb_min]
    dt[agg$src_idx, (mean_col) := agg$nb_mean]
  }

  # --- Cleanup ---
  dt[, .row_idx := NULL]

  message("Neighbor feature construction complete.")
  return(dt)
}


# ============================================================
# USAGE — drop-in replacement for the original outer loop
# ============================================================

# Original code (86+ hours):
# -------------------------------------------------------
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }
# -------------------------------------------------------

# Optimized replacement (~5-15 minutes):
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_fast(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors       = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# cell_data now has the same neighbor feature columns as before.
# The trained Random Forest model can be applied directly — no retraining needed.
# predict(rf_model, newdata = cell_data)
```

---

## Why This Works and Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | The grouped `max`, `min`, `mean` over the exact same set of non-NA neighbor values produces identical results. No approximation is introduced. |
| **Trained RF model untouched** | Only the feature construction step is replaced. The model object is never modified or retrained. Column names are preserved (assuming `compute_and_add_neighbor_features` used the same `nb_max_`, `nb_min_`, `nb_mean_` naming convention — adjust the prefix strings if your original code used different names). |
| **Memory on 16 GB laptop** | The largest intermediate (`expanded`) is ~25.8M rows × 4 integer/numeric columns ≈ 0.8 GB. The `data.table` in-place operations keep peak memory well under 10 GB. |
| **Speedup source** | String operations (`paste`, named-vector lookup) are eliminated entirely. The 6.46M-iteration R-level `lapply` is replaced by `data.table` keyed joins (radix-sort based, C-level) and vectorized grouped aggregation (also C-level). |

### Expected Timing Breakdown (16 GB laptop)

| Step | Estimated Time |
|---|---|
| Edge list construction (~344K list elements) | ~2–5 seconds |
| Keyed join to expand to ~25.8M rows | ~30–90 seconds |
| Neighbor row index lookup (keyed join) | ~30–60 seconds |
| Grouped aggregation × 5 variables | ~2–5 minutes |
| **Total** | **~5–10 minutes** |

This represents a speedup of roughly **500–1000×** over the original implementation.