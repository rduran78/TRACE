 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is twofold:

### 1. `build_neighbor_lookup` — O(n) `lapply` over 6.46 million rows with per-row string pasting and named-vector lookups
- **String key construction:** `paste(id, year, sep="_")` is called millions of times, and named-vector lookup via character keys (`idx_lookup[neighbor_keys]`) is hash-table lookup repeated for every single row. With ~6.46M rows and an average of ~4 rook neighbors per cell, this is ~25.8 million string constructions and hash lookups just for the neighbor resolution step.
- **`lapply` over 6.46M elements:** Pure R loop overhead is enormous. Each iteration also allocates small vectors (paste results, index subsets), creating massive GC pressure.

### 2. `compute_neighbor_stats` — Another `lapply` over 6.46M rows computing max/min/mean by subsetting a numeric vector
- Each call to `vals[idx]` and the subsequent `max/min/mean` is fast individually, but repeated 6.46M × 5 variables = ~32.3 million times total, the R interpreter overhead dominates.
- `do.call(rbind, result)` on a 6.46M-element list of 3-vectors is itself slow and memory-hungry.

### Memory
With 6.46M rows × 110 columns, the data frame alone is ~5–6 GB. The `neighbor_lookup` list (6.46M elements, each a small integer vector) adds another ~1–2 GB. Headroom on a 16 GB machine is tight.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Per-row string pasting and hash lookup in `build_neighbor_lookup` | Replace with integer arithmetic: encode `(id, year)` as a single integer key, use `data.table` fast joins or direct integer indexing to resolve neighbor row indices in bulk (vectorized). |
| `lapply` over 6.46M rows for neighbor stats | Replace with a **flat edge-list approach**: expand all neighbor relationships into a long `data.table` of `(row_i, neighbor_row_j)`, join the variable values, then compute grouped `max/min/mean` by `row_i` using `data.table` aggregation — fully vectorized, no R-level loop. |
| `do.call(rbind, ...)` on millions of small vectors | Eliminated by the grouped `data.table` aggregation which returns a single `data.table` directly. |
| Memory pressure | The flat edge list for 6.46M rows × ~4 neighbors ≈ 25.8M rows × 2 integer columns ≈ 0.4 GB — manageable. We avoid duplicating the full data; we join only the single variable column needed. Process one variable at a time and discard intermediate objects. |
| 86+ hours runtime | Expected reduction to **minutes** (the vectorized `data.table` join + grouped aggregation on ~25.8M rows is very fast). |

**Key invariant preserved:** The numerical output (max, min, mean of rook-neighbor values per cell-year) is identical. The trained Random Forest model is untouched.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# Step 1: Build a flat edge list (row_i → neighbor_row_j) — VECTORIZED
#
# This replaces build_neighbor_lookup entirely.
# Instead of a 6.46M-element list, we produce a data.table with ~25.8M
# rows of (row_idx, neighbor_row_idx) that can be reused for every variable.
# ──────────────────────────────────────────────────────────────────────

build_neighbor_edge_list <- function(cell_dt, id_order, neighbors) {
  # cell_dt must be a data.table with columns 'id' and 'year'
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer neighbor indices into id_order)

  # --- Map each cell ID to its position in id_order (1-based ref index) ---
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # --- Build a cell-level edge list: (ref_idx → neighbor_ref_idx) ---
  #     Expand the nb list into two parallel integer vectors.
  n_neighbors <- lengths(neighbors)                 # integer vector, length = #cells
  from_ref    <- rep(seq_along(neighbors), n_neighbors)
  to_ref      <- unlist(neighbors, use.names = FALSE)

  cell_edges <- data.table(from_ref = from_ref, to_ref = to_ref)

  # --- Map ref indices back to actual cell IDs ---
  cell_edges[, from_id := id_order[from_ref]]
  cell_edges[, to_id   := id_order[to_ref]]

  # --- Create a row-index lookup table: (id, year) → row position in cell_dt ---
  cell_dt[, .row_idx := .I]
  row_lookup <- cell_dt[, .(.row_idx, id, year)]
  setkey(row_lookup, id, year)

  # --- Get the unique years present in the data ---
  years <- sort(unique(cell_dt$year))

  # --- Cross-join cell_edges × years, then resolve row indices for both
  #     the focal cell and the neighbor cell.
  #     To avoid a massive cross join in memory, we do two keyed joins. ---

  # Expand: for every (from_id, to_id) pair, replicate across all years.
  # But not every cell is present in every year, so we join rather than cross.

  # Approach: start from cell_dt rows, attach their ref_idx, then join neighbors.
  cell_dt[, ref_idx := id_to_ref[as.character(id)]]

  # Focal rows: (row_idx, ref_idx, year)
  focal <- cell_dt[, .(focal_row = .row_idx, ref_idx, year)]
  setkey(focal, ref_idx)

  # For each focal row, find its neighbor ref indices via cell_edges
  setkey(cell_edges, from_ref)
  # Join: for each focal row, get all neighbor ref indices
  edge_expanded <- cell_edges[focal, on = .(from_ref = ref_idx),
                              .(focal_row, to_ref, year),
                              allow.cartesian = TRUE,
                              nomatch = NULL]

  # Now resolve neighbor rows: need (to_ref → to_id), then join (to_id, year) → neighbor_row
  edge_expanded[, neighbor_id := id_order[to_ref]]

  # Join to get neighbor row index
  setkey(edge_expanded, neighbor_id, year)
  setkey(row_lookup, id, year)

  edge_expanded[row_lookup,
                neighbor_row := i..row_idx,
                on = .(neighbor_id = id, year = year)]

  # Drop rows where the neighbor wasn't found (boundary / missing year)
  edge_list <- edge_expanded[!is.na(neighbor_row),
                             .(focal_row, neighbor_row)]

  # Clean up temporary columns from cell_dt
  cell_dt[, c(".row_idx", "ref_idx") := NULL]

  return(edge_list)
}


# ──────────────────────────────────────────────────────────────────────
# Step 2: Compute neighbor stats for one variable — VECTORIZED
#
# This replaces compute_neighbor_stats.
# ──────────────────────────────────────────────────────────────────────

compute_neighbor_stats_fast <- function(cell_dt, edge_list, var_name) {
  # Extract the variable values for all neighbor rows
  vals <- cell_dt[[var_name]]

  # Build a working table: focal_row + neighbor's value
  work <- copy(edge_list)
  work[, nval := vals[neighbor_row]]

  # Remove NA neighbor values (matches original behavior)
  work <- work[!is.na(nval)]

  # Grouped aggregation — fully vectorized via data.table
  stats <- work[, .(nb_max  = max(nval),
                     nb_min  = min(nval),
                     nb_mean = mean(nval)),
                keyby = .(focal_row)]

  # Allocate output columns (NA by default, matching original behavior for
  # cells with no valid neighbors)
  n <- nrow(cell_dt)
  out_max  <- rep(NA_real_, n)
  out_min  <- rep(NA_real_, n)
  out_mean <- rep(NA_real_, n)

  out_max[stats$focal_row]  <- stats$nb_max
  out_min[stats$focal_row]  <- stats$nb_min
  out_mean[stats$focal_row] <- stats$nb_mean

  # Name the output columns consistently with the original pipeline
  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")

  cell_dt[, (col_max)  := out_max]
  cell_dt[, (col_min)  := out_min]
  cell_dt[, (col_mean) := out_mean]

  invisible(cell_dt)
}


# ──────────────────────────────────────────────────────────────────────
# Step 3: Main driver — drop-in replacement for the outer loop
# ──────────────────────────────────────────────────────────────────────

run_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique) {

  # Convert to data.table if needed (by reference — no copy)
  if (!is.data.table(cell_data)) {
    setDT(cell_data)
  }

  message("Building vectorized neighbor edge list ...")
  edge_list <- build_neighbor_edge_list(cell_data, id_order, rook_neighbors_unique)
  message(sprintf("  Edge list: %s rows (%.1f MB)",
                  formatC(nrow(edge_list), big.mark = ","),
                  object.size(edge_list) / 1e6))

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for '%s' ...", var_name))
    compute_neighbor_stats_fast(cell_data, edge_list, var_name)
  }

  message("Done.")
  return(cell_data)
}


# ──────────────────────────────────────────────────────────────────────
# Usage (drop-in replacement for the original outer loop)
# ──────────────────────────────────────────────────────────────────────
# cell_data <- run_neighbor_features(cell_data, id_order, rook_neighbors_unique)
#
# Then proceed to prediction with the already-trained Random Forest model
# exactly as before. The new columns have the same names and identical
# numerical values as the originals.
```

---

## Why This Works — Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor resolution** | 6.46M R-level iterations with string pasting + hash lookup | Single vectorized `data.table` keyed join (~25.8M rows) |
| **Stat computation** | 6.46M R-level iterations × 5 vars = 32.3M iterations | 5 × one grouped `data.table` aggregation over ~25.8M rows |
| **Memory for lookup** | List of 6.46M small integer vectors (~1–2 GB + GC overhead) | One 2-column integer `data.table` (~0.4 GB) |
| **Expected runtime** | 86+ hours | **~5–15 minutes** (dominated by the keyed joins and grouped aggregations) |
| **Numerical output** | max, min, mean of non-NA neighbor values | **Identical** — same subsetting logic, same aggregation functions |
| **RF model** | Trained model used downstream | **Untouched** — only the feature-engineering step is optimized |

The key insight is replacing millions of R-interpreter-level iterations with bulk vectorized operations via `data.table`'s radix-sort-based keyed joins and grouped aggregation, which are implemented in optimized C and handle this scale comfortably within 16 GB RAM.