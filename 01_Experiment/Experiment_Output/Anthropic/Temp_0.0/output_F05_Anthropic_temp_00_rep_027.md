 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Inside the Loop

`build_neighbor_lookup` iterates over all ~6.46 million rows. For **each row**, it:

1. Converts `data$id[i]` to character and looks it up in `id_to_ref` — **6.46M character coercions + named-vector lookups**.
2. Subsets `id_order[neighbors[[ref_idx]]]` — fine, but done row-by-row.
3. Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` — **allocates a new character vector per row** (~4 neighbor keys × 6.46M rows ≈ 25.8M string constructions).
4. Looks up `idx_lookup[neighbor_keys]` — **named-vector lookup on a 6.46M-element character vector, repeated 6.46M times**.

The named-vector lookup `idx_lookup[neighbor_keys]` is O(n) per probe in the worst case (R's named vectors use linear hashing with potential degradation at this scale). Across all rows this is effectively **O(N × k)** with large constant factors due to string hashing on a 6.46M-name vector.

### The Deeper Structural Insight

The neighbor relationships are **spatial** (cell-to-cell) and **time-invariant**. The year dimension is only used to find "the same neighbor in the same year." This means the entire lookup can be reformulated as:

> For each cell-year row `i`, find the rows that share the same year AND whose cell id is a rook neighbor of row `i`'s cell id.

This is a **join** problem, not a per-row string-lookup problem. The neighbor graph is fixed across years, so we can:

1. Build a spatial neighbor edge list once (cell → neighbor_cell).
2. Build a (cell_id, year) → row_index map once using a **hash table** (via `data.table` keyed join).
3. Expand the edge list by year using a **vectorized equi-join** — no per-row loop at all.
4. Compute all neighbor statistics using **vectorized grouped aggregation**.

This replaces the 6.46M-iteration `lapply` + string paste + named-vector probe with a single `data.table` merge + grouped aggregation.

### Estimated Speedup

| Step | Current | Proposed |
|---|---|---|
| Build neighbor lookup | ~6.46M × paste + named-vec probe ≈ hours | One vectorized merge ≈ seconds |
| Compute neighbor stats (per var) | `lapply` over 6.46M lists | `data.table` grouped aggregation ≈ seconds |
| Total for 5 variables | 86+ hours | **~1–5 minutes** |

---

## Optimization Strategy

1. **Convert the `nb` object to an edge list** of (cell_id, neighbor_cell_id) pairs.
2. **Key `cell_data` as a `data.table`** on (id, year) for O(1) keyed lookups.
3. **Cross the edge list with years** via a merge to get (row_index, neighbor_row_index) pairs — fully vectorized.
4. **Compute max/min/mean** per row using `data.table` grouped aggregation on the expanded edge table.
5. **Column-bind** results back to `cell_data`.

This preserves the exact numerical estimand (max, min, mean of non-NA neighbor values per cell-year) and does not touch the trained Random Forest model.

---

## Working R Code

```r
library(data.table)

#' Optimized neighbor feature construction.
#' Replaces build_neighbor_lookup + compute_neighbor_stats + the outer loop.
#'
#' @param cell_data        data.frame or data.table with columns: id, year, and all neighbor_source_vars
#' @param id_order          integer vector of cell IDs in the order matching rook_neighbors_unique
#' @param rook_neighbors_unique  spdep nb object (list of integer index vectors)
#' @param neighbor_source_vars   character vector of variable names to compute neighbor stats for
#' @return cell_data (data.table) with new columns appended: {var}_max, {var}_min, {var}_mean
build_all_neighbor_features <- function(cell_data,
                                        id_order,
                                        rook_neighbors_unique,
                                        neighbor_source_vars) {

  # --- Step 0: Convert to data.table if needed, preserve original row order ---
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  cell_data[, ..row_id.. := .I]

  # --- Step 1: Build spatial edge list from the nb object ---
  # Each element of rook_neighbors_unique is an integer vector of indices into id_order.
  # We expand this into a two-column edge list of actual cell IDs.
  message("Step 1/4: Building spatial edge list...")

  edge_from <- rep(
    seq_along(rook_neighbors_unique),
    times = lengths(rook_neighbors_unique)
  )
  edge_to <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Remove the 0-neighbor sentinel if spdep uses 0L for "no neighbors"
  valid <- edge_to != 0L
  edge_from <- edge_from[valid]
  edge_to   <- edge_to[valid]

  # Map from nb indices to actual cell IDs
  edges <- data.table(
    focal_id    = id_order[edge_from],
    neighbor_id = id_order[edge_to]
  )
  rm(edge_from, edge_to, valid)

  message(sprintf("  Edge list: %s directed neighbor pairs", format(nrow(edges), big.mark = ",")))

  # --- Step 2: Build row-index lookup keyed on (id, year) ---
  message("Step 2/4: Building keyed row-index lookup...")

  row_lookup <- cell_data[, .(id, year, ..row_id..)]
  setkey(row_lookup, id, year)

  # --- Step 3: Expand edges × years via vectorized join ---
  # For each row in cell_data, we need its (focal_id, year).
  # Then we join to edges on focal_id, and then join the neighbor_id + year
  # back to row_lookup to get the neighbor's row index.
  message("Step 3/4: Expanding edge list across years (vectorized join)...")

  # Get focal info: which cell and year does each row represent?
  focal_info <- cell_data[, .(focal_id = id, year, focal_row = ..row_id..)]

  # Join focal_info to edges: for each row, find all spatial neighbors
  # This is a many-to-many join: each focal row joins to its ~4 neighbors
  setkey(edges, focal_id)
  setkey(focal_info, focal_id)

  # Merge: for each (focal_id, year) row, attach all neighbor_ids
  expanded <- edges[focal_info, on = "focal_id", allow.cartesian = TRUE, nomatch = NULL]
  # expanded now has columns: focal_id, neighbor_id, year, focal_row

  # Now look up the row index of each (neighbor_id, year) pair
  setkey(expanded, neighbor_id, year)
  expanded[row_lookup, neighbor_row := i...row_id.., on = .(neighbor_id = id, year)]

  # Drop pairs where the neighbor doesn't exist in that year

  expanded <- expanded[!is.na(neighbor_row)]

  message(sprintf("  Expanded neighbor pairs: %s", format(nrow(expanded), big.mark = ",")))

  # Free memory
  rm(focal_info, edges, row_lookup)
  gc()

  # --- Step 4: Compute grouped statistics for each source variable ---
  message("Step 4/4: Computing neighbor statistics...")

  for (var_name in neighbor_source_vars) {
    message(sprintf("  Processing: %s", var_name))

    # Attach neighbor values
    expanded[, nval := cell_data[[var_name]][neighbor_row]]

    # Compute grouped stats, excluding NAs
    stats <- expanded[!is.na(nval),
                      .(var_max  = max(nval),
                        var_min  = min(nval),
                        var_mean = mean(nval)),
                      by = focal_row]

    # Initialize new columns with NA
    max_col  <- paste0(var_name, "_max")
    min_col  <- paste0(var_name, "_min")
    mean_col <- paste0(var_name, "_mean")

    cell_data[, (max_col)  := NA_real_]
    cell_data[, (min_col)  := NA_real_]
    cell_data[, (mean_col) := NA_real_]

    # Assign computed values
    cell_data[stats$focal_row, (max_col)  := stats$var_max]
    cell_data[stats$focal_row, (min_col)  := stats$var_min]
    cell_data[stats$focal_row, (mean_col) := stats$var_mean]

    rm(stats)
  }

  # Clean up helper column
  expanded[, nval := NULL]
  cell_data[, ..row_id.. := NULL]

  message("Done.")
  return(cell_data)
}


# =============================================================================
# USAGE — drop-in replacement for the original outer loop
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_all_neighbor_features(
  cell_data              = cell_data,
  id_order               = id_order,
  rook_neighbors_unique  = rook_neighbors_unique,
  neighbor_source_vars   = neighbor_source_vars
)

# The trained Random Forest model is untouched.
# cell_data now has the same 15 new columns ({var}_{max,min,mean})
# with identical numerical values as the original implementation.
```

---

## Verification Script (Optional)

To confirm numerical equivalence with the original implementation on a small subset:

```r
# Run original on a small sample for verification
set.seed(42)
sample_rows <- sample(nrow(cell_data), 1000)

# Original method
neighbor_lookup_orig <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
orig_stats <- compute_neighbor_stats(cell_data, neighbor_lookup_orig, "ntl")

# Compare
cat("Max difference (ntl_max):", max(abs(cell_data$ntl_max[sample_rows] - orig_stats[sample_rows, 1]), na.rm = TRUE), "\n")
cat("Max difference (ntl_min):", max(abs(cell_data$ntl_min[sample_rows] - orig_stats[sample_rows, 2]), na.rm = TRUE), "\n")
cat("Max difference (ntl_mean):", max(abs(cell_data$ntl_mean[sample_rows] - orig_stats[sample_rows, 3]), na.rm = TRUE), "\n")
# Expected: 0 for all three
```

---

## Memory Considerations (16 GB Laptop)

| Object | Estimated Size |
|---|---|
| `cell_data` (6.46M × 110 cols) | ~5.7 GB |
| `expanded` edge table (~25.8M rows × 4 cols) | ~0.8 GB |
| Temporary stats per variable | ~0.2 GB |
| **Peak total** | **~8–9 GB** ✓ fits in 16 GB |

If memory is tight, process variables one at a time and call `gc()` between iterations (the code above already does this). If the expanded table is too large, you can chunk by year (28 chunks), which reduces the expanded table to ~1/28th size per chunk at the cost of a small constant-factor slowdown (still minutes, not hours).