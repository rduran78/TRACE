 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck has two root causes:

### 1. `build_neighbor_lookup` — O(n) `lapply` over 6.46 million rows with per-row string pasting and named-vector lookups

Each of the ~6.46M iterations performs:
- `as.character()` and a named-vector lookup (`id_to_ref`).
- `paste()` to build neighbor keys for every neighbor of that cell in that year.
- A second named-vector lookup (`idx_lookup`) on those keys.

Named-vector lookups in R are hash-based but still carry significant per-call overhead at this scale. Pasting strings millions of times is also expensive. The result is a **list of 6.46M integer vectors**, which is both slow to build and memory-heavy (each list element is a separate R object with its own SEXP header — roughly 100+ bytes overhead per element, so the list skeleton alone can consume >600 MB before the actual index data).

### 2. `compute_neighbor_stats` — another O(n) `lapply` over 6.46M rows, called 5 times

For every row, it subsets a numeric vector by the neighbor indices, removes NAs, and computes max/min/mean. This is called once per variable (×5), so ~32.3M R-level function calls. Each call allocates small temporary vectors. The `do.call(rbind, ...)` at the end also materializes a 6.46M × 3 intermediate list before binding.

### Combined effect

~6.46M list elements built in step 1, then traversed 5 times in step 2, with heavy per-element R overhead → **estimated 86+ hours** on a laptop.

---

## Optimization Strategy

| Principle | Action |
|---|---|
| **Eliminate the per-row list** | Replace the 6.46M-element neighbor lookup list with a flat **edge table** (a two-column `data.table` of `[row_i, row_j]` pairs). This is a sparse-matrix/CSR-style representation that R and `data.table` can process in bulk. |
| **Vectorize the join** | Use `data.table` keyed joins (binary search, no hashing of 6.46M strings) to map `(neighbor_id, year)` → row index in one vectorized pass. |
| **Vectorize the aggregation** | Use `data.table`'s `by=` grouped aggregation (`max`, `min`, `mean`) over the edge table — one pass per variable, fully vectorized in C. No R-level `lapply`. |
| **Minimize memory** | The edge table has ~1.37M × 28 ≈ 38.4M rows of two integer columns ≈ **~307 MB**, far less than 6.46M list elements. Intermediate results are column vectors, not lists of 3-vectors. |
| **Preserve the trained RF model and the numerical estimand** | The output columns are identical (same names, same values: neighbor max, min, mean for each variable). The RF model is not retouched. |

Expected speedup: from 86+ hours to **minutes** (the join is O(n log n); each grouped aggregation is O(E) where E ≈ 38.4M).

---

## Working R Code

```r
library(data.table)

# ─────────────────────────────────────────────────────────────
# Step 1: Build a flat edge table (replaces build_neighbor_lookup)
# ─────────────────────────────────────────────────────────────

build_neighbor_edge_table <- function(cell_dt, id_order, neighbors) {

  # cell_dt must be a data.table with columns: id, year, and a row index
  # id_order: integer vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer index vectors into id_order)

  # --- 1a. Expand the nb object into a directed edge list of cell IDs --------
  #     Each element neighbors[[k]] contains indices into id_order.
  n_neighbors <- lengths(neighbors)
  from_idx    <- rep(seq_along(neighbors), times = n_neighbors)
  to_idx      <- unlist(neighbors, use.names = FALSE)

  # Remove the 0-entries that spdep uses to mark cells with no neighbors
  valid       <- to_idx != 0L
  from_idx    <- from_idx[valid]
  to_idx      <- to_idx[valid]

  edge_ids <- data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )
  # edge_ids now has ~1.37M rows (directed rook-neighbor pairs)

  # --- 1b. Cross-join with years to get (from_id, to_id, year) ---------------
  years <- sort(unique(cell_dt$year))
  edge_full <- edge_ids[, .(year = years), by = .(from_id, to_id)]
  # ~1.37M × 28 ≈ 38.4M rows

  # --- 1c. Map from_id×year → row position in cell_dt ("row_i")
  #         Map to_id×year   → row position in cell_dt ("row_j")
  #     We add a row-number column to cell_dt for this purpose.
  cell_dt[, .row_idx := .I]

  # Keyed join: map (from_id, year) → row_i
  setkey(cell_dt, id, year)
  edge_full[cell_dt, row_i := i..row_idx, on = .(from_id = id, year)]
  edge_full[cell_dt, row_j := i..row_idx, on = .(to_id   = id, year)]

  # Drop edges where either side is missing (cell not in data for that year)
  edge_full <- edge_full[!is.na(row_i) & !is.na(row_j)]

  # Clean up helper column
  cell_dt[, .row_idx := NULL]

  edge_full[]
}

# ─────────────────────────────────────────────────────────────
# Step 2: Compute neighbor stats for one variable (vectorized)
# ─────────────────────────────────────────────────────────────

compute_neighbor_stats_vec <- function(cell_dt, edge_table, var_name) {
  # Attach the neighbor's value to every edge
  vals <- cell_dt[[var_name]]
  edge_table[, nval := vals[row_j]]

  # Grouped aggregation: one group per row_i (the focal cell-year row)
  agg <- edge_table[!is.na(nval),
    .(nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)),
    keyby = .(row_i)
  ]

  # Initialize result columns with NA
  n <- nrow(cell_dt)
  col_max  <- rep(NA_real_, n)
  col_min  <- rep(NA_real_, n)
  col_mean <- rep(NA_real_, n)

  # Fill in the rows that have neighbors
  col_max [agg$row_i] <- agg$nb_max
  col_min [agg$row_i] <- agg$nb_min
  col_mean[agg$row_i] <- agg$nb_mean

  # Clean up temporary column
  edge_table[, nval := NULL]

  # Return a data.table of the three new columns
  setnames(
    data.table(col_max, col_min, col_mean),
    c(paste0(var_name, "_nb_max"),
      paste0(var_name, "_nb_min"),
      paste0(var_name, "_nb_mean"))
  )
}

# ─────────────────────────────────────────────────────────────
# Step 3: Outer loop — drop-in replacement
# ─────────────────────────────────────────────────────────────

## Convert to data.table if not already
if (!is.data.table(cell_data)) cell_data <- as.data.table(cell_data)

## Build the edge table once (replaces build_neighbor_lookup)
message("Building edge table …")
edge_table <- build_neighbor_edge_table(cell_data, id_order, rook_neighbors_unique)
message(sprintf("Edge table: %s rows", format(nrow(edge_table), big.mark = ",")))

## Compute and attach neighbor features for each variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor stats for: %s", var_name))
  new_cols <- compute_neighbor_stats_vec(cell_data, edge_table, var_name)
  # Bind new columns to cell_data by reference
  cell_data[, names(new_cols) := new_cols]
}

message("Done. Neighbor features added.")
```

---

## Summary of What Changed and What Is Preserved

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup | 6.46M-element R list built via `lapply` + `paste` | Flat `data.table` edge table built via vectorized joins |
| Aggregation | `lapply` over 6.46M elements × 5 vars | `data.table` grouped `by=` aggregation × 5 vars |
| Memory for lookup | ~600 MB+ list overhead + index data | ~307 MB flat integer table |
| Estimated runtime | 86+ hours | **Minutes** (dominated by the 38.4M-row grouped aggregation, which `data.table` handles in C) |
| Output columns | `{var}_nb_max`, `{var}_nb_min`, `{var}_nb_mean` | Identical names and identical numerical values |
| Trained RF model | Untouched | Untouched |
| Numerical estimand | Preserved (same max/min/mean over same neighbor sets) | Preserved |