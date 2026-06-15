 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### Layer 1: String-key construction inside the row-level `lapply`
`build_neighbor_lookup` calls `paste(..., sep="_")` to build `idx_lookup` once (6.46M keys), but then **for every one of the 6.46M rows**, it calls `paste()` again to build `neighbor_keys` and does a named-vector lookup (`idx_lookup[neighbor_keys]`). Named-vector lookup in R is hash-table access, but creating ~4 temporary string vectors per row × 6.46M rows is extremely expensive.

### Layer 2: The lookup result is reused, but the architecture forces row-level iteration
`build_neighbor_lookup` returns a list of 6.46M integer vectors. This is then passed to `compute_neighbor_stats`, which iterates over the same 6.46M entries **once per variable**. With 5 variables, that's 5 × 6.46M = 32.3M list iterations, each extracting a small numeric vector and computing `max/min/mean`.

### Layer 3: The neighbor relationship is year-invariant but encoded year-by-year
Rook neighbors are purely spatial — they don't change across years. Yet the current code re-discovers "which rows are my neighbors in this year" for every cell-year, when the answer is structurally identical across all 28 years. This inflates the problem from ~344K spatial lookups to ~6.46M lookups.

### Estimated cost breakdown (current approach)
| Step | Operations | Bottleneck |
|---|---|---|
| String key construction | 6.46M × ~4 neighbors × `paste` | ~30-40% of time |
| Named-vector lookup | 6.46M hash lookups | ~20-30% |
| `compute_neighbor_stats` | 5 vars × 6.46M list iterations | ~30-40% |
| **Total estimated** | | **86+ hours** |

---

## Optimization Strategy

1. **Separate space from time.** Build the neighbor mapping once at the cell level (344K cells), then broadcast across years using integer arithmetic — no strings, no hashing.

2. **Vectorize the aggregation.** Replace 6.46M-element `lapply` with a single `data.table` grouped operation. Explode each cell-year into its neighbor rows (a long edge-table), join the variable values, and aggregate with `data.table`'s optimized `max/min/mean` by group.

3. **Process all 5 variables in one pass** over the edge-table rather than five separate passes.

This reduces the complexity from O(rows × neighbors × variables) with per-element R overhead to a handful of vectorized data.table joins and grouped aggregations.

### Expected speedup
- Eliminates all `paste`/string hashing: **100%** of string work removed.
- Replaces 32.3M R-level list iterations with ~2-3 vectorized `data.table` operations.
- Estimated runtime: **2–10 minutes** on the same laptop.

---

## Working R Code

```r
# =============================================================================
# Optimized neighbor feature construction
# Preserves the exact numerical estimand (max, min, mean of neighbor values)
# Preserves the trained Random Forest model (no retraining needed)
# =============================================================================

library(data.table)

build_and_apply_neighbor_features <- function(cell_data,
                                               id_order,
                                               rook_neighbors_unique,
                                               neighbor_source_vars) {
  # -------------------------------------------------------------------------
  # Step 1: Build a spatial-only edge table (year-invariant)
  #         This replaces the per-row string-key lookup entirely.
  # -------------------------------------------------------------------------
  # id_order is the vector of cell IDs in the order matching the nb object.
  # rook_neighbors_unique[[k]] gives the integer indices (into id_order) of

  # the neighbors of cell id_order[k].

  # Build edge list: focal_id -> neighbor_id  (cell IDs, not row indices)
  n_cells <- length(id_order)
  # Pre-allocate by counting total edges
  n_edges <- sum(lengths(rook_neighbors_unique))  # ~1.37M directed edges

  focal_ids    <- integer(n_edges)
  neighbor_ids <- integer(n_edges)
  pos <- 1L
  for (k in seq_len(n_cells)) {
    nb_idx <- rook_neighbors_unique[[k]]
    if (length(nb_idx) > 0L) {
      len <- length(nb_idx)
      focal_ids[pos:(pos + len - 1L)]    <- id_order[k]
      neighbor_ids[pos:(pos + len - 1L)] <- id_order[nb_idx]
      pos <- pos + len
    }
  }

  edges <- data.table(focal_id = focal_ids, neighbor_id = neighbor_ids)

  # -------------------------------------------------------------------------
  # Step 2: Convert cell_data to data.table and create a compact row key
  # -------------------------------------------------------------------------
  dt <- as.data.table(cell_data)

  # Create a row index for the focal data (to map results back)
  dt[, .row_idx := .I]

  # -------------------------------------------------------------------------
  # Step 3: For each variable, join edges to values and aggregate
  # -------------------------------------------------------------------------
  # We process all variables efficiently. For each variable:
  #   - Join the edge table with dt on (focal_id, year) to get focal row indices
  #   - Join again on (neighbor_id, year) to get neighbor values
  #   - Aggregate max, min, mean by focal row index

  # Create a keyed lookup for neighbor values: (id, year) -> variable values
  # We only need id, year, and the source variable columns
  val_cols <- intersect(neighbor_source_vars, names(dt))
  lookup_dt <- dt[, c("id", "year", val_cols, ".row_idx"), with = FALSE]

  # Key the lookup by id and year for fast joins
  setkey(lookup_dt, id, year)

  # Get unique years
  years <- sort(unique(dt$year))

  # Build the full edge-year table:
  # For each year, every spatial edge is an edge. Instead of a Cartesian

  # product (which would be 1.37M * 28 = 38.4M rows), we join through dt.

  # Focal side: map each row in dt to its row_idx and cell id
  focal_map <- dt[, .(focal_id = id, year = year, .row_idx)]
  setkey(focal_map, focal_id)

  # Join focal_map with edges to get: for each focal row, all neighbor cell IDs

  # This is the key step: we broadcast the spatial edges across years via join
  setkey(edges, focal_id)
  # Each focal row has edges to its spatial neighbors in the SAME year
  expanded <- edges[focal_map, on = "focal_id", allow.cartesian = TRUE,
                    nomatch = NULL]
  # expanded has columns: focal_id, neighbor_id, year, .row_idx
  # .row_idx is the focal cell-year's row in the original data

  # Now join to get neighbor values
  setkey(expanded, neighbor_id, year)

  for (var in val_cols) {
    # Extract just the column we need for the neighbor side
    nb_vals <- lookup_dt[, .(id, year, nb_val = get(var))]
    setkey(nb_vals, id, year)

    # Join neighbor values onto the expanded edge table
    merged <- nb_vals[expanded, on = c(id = "neighbor_id", "year"),
                      nomatch = NA]
    # merged has: id (=neighbor_id), year, nb_val, focal_id, .row_idx

    # Drop NAs in the neighbor value (mirrors original: neighbor_vals[!is.na()])
    merged_clean <- merged[!is.na(nb_val)]

    # Aggregate by focal row
    agg <- merged_clean[, .(
      nb_max  = max(nb_val),
      nb_min  = min(nb_val),
      nb_mean = mean(nb_val)
    ), by = .row_idx]

    # Initialize result columns with NA
    max_col  <- paste0(var, "_neighbor_max")
    min_col  <- paste0(var, "_neighbor_min")
    mean_col <- paste0(var, "_neighbor_mean")

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Fill in computed values
    dt[agg$.row_idx, (max_col)  := agg$nb_max]
    dt[agg$.row_idx, (min_col)  := agg$nb_min]
    dt[agg$.row_idx, (mean_col) := agg$nb_mean]

    message(sprintf("  Done: %s -> %s, %s, %s", var, max_col, min_col, mean_col))
  }

  # -------------------------------------------------------------------------
  # Step 4: Clean up and return as data.frame (to match downstream expectations)
  # -------------------------------------------------------------------------
  dt[, .row_idx := NULL]
  return(as.data.frame(dt))
}


# =============================================================================
# Usage (drop-in replacement for the original outer loop)
# =============================================================================
#
# # Original code:
# # neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# # for (var_name in neighbor_source_vars) {
# #   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# # }
#
# # Optimized replacement (single call):
# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#
# cell_data <- build_and_apply_neighbor_features(
#   cell_data,
#   id_order,
#   rook_neighbors_unique,
#   neighbor_source_vars
# )
#
# # The resulting cell_data has the same columns as before:
# #   {var}_neighbor_max, {var}_neighbor_min, {var}_neighbor_mean
# # with identical numerical values.
# #
# # The trained Random Forest model is unchanged and can be used directly
# # for prediction on the updated cell_data.
```

---

## Memory Considerations (16 GB laptop)

| Object | Estimated size |
|---|---|
| `dt` (6.46M × ~115 cols) | ~5.5 GB |
| `edges` (1.37M × 2 int cols) | ~11 MB |
| `expanded` (6.46M × ~4 neighbors × 4 cols) | ~800 MB |
| `merged` per variable | ~800 MB (reused) |
| **Peak** | **~8-9 GB** ✔ fits in 16 GB |

If memory is tight, the loop over `val_cols` already processes one variable at a time, and `merged`/`merged_clean`/`agg` are garbage-collected between iterations. You can add explicit `rm()` + `gc()` calls inside the loop if needed.

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| String key construction | 6.46M × per-row `paste` | **Zero** — integer joins only |
| Neighbor resolution | Per-row hash lookup | Vectorized `data.table` join |
| Variable aggregation | 5 × 6.46M `lapply` | 5 × one grouped `data.table` op |
| Estimated runtime | 86+ hours | **2–10 minutes** |
| Numerical output | max/min/mean per neighbor set | **Identical** |
| RF model impact | N/A | **None — preserved as-is** |