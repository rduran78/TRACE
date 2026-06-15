 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The colleague's claim is **partially correct but misses the deeper bottleneck**. Let me audit both functions:

### `compute_neighbor_stats()` — Colleague's Claimed Bottleneck
- `do.call(rbind, result)`: Binding ~6.46M small 3-element vectors into a matrix is not cheap, but `do.call(rbind, ...)` on a list of uniform-length numeric vectors is actually reasonably efficient in R. This is a **minor** bottleneck.
- There is no "repeated list binding" inside `compute_neighbor_stats()` — it uses `lapply` and returns fixed-length vectors. The colleague's description of the inner workings is factually wrong. This function runs 5 times (once per variable), each time iterating over 6.46M entries with simple numeric subsetting. It's not trivial, but it's **not the dominant cost**.

### `build_neighbor_lookup()` — The True Deep Bottleneck
This function is called once but contains the **critical bottleneck**:

1. **`paste()` to build string keys and `idx_lookup` named vector**: It creates a named vector of length ~6.46M using `paste(data$id, data$year, sep = "_")`. Named vector lookup in R uses linear hashing that degrades at scale.

2. **Per-row `lapply` over 6.46M rows**: Inside the lambda:
   - `as.character(data$id[i])` — called 6.46M times.
   - `id_to_ref[as.character(...)]` — named vector lookup, 6.46M times.
   - `paste(neighbor_cell_ids, data$year[i], sep = "_")` — string construction for every neighbor of every row, totaling ~1.37M neighbor pairs × 28 years ≈ **38.4M string constructions** (and likely more, since each row's neighbors are looked up individually).
   - `idx_lookup[neighbor_keys]` — named vector lookup of ~38.4M keys against a 6.46M-element named vector.

The total work in `build_neighbor_lookup` is **O(N × avg_neighbors)** string concatenations and named-vector lookups, where N = 6.46M and avg_neighbors ≈ 4 (rook neighbors). That's ~25.8M string-match lookups against a 6.46M-entry named vector — **this is the 86-hour bottleneck**.

**String-keyed lookup in a named vector is O(n) per miss in the worst case in R's internal hashing, and even at O(1) amortized, the constant factor of `paste` + character matching at this scale is enormous.**

### Verdict: **Reject the colleague's diagnosis.** The dominant bottleneck is `build_neighbor_lookup()`, specifically the row-by-row `lapply` with repeated string construction and named-vector lookups over 6.46M rows.

---

## Optimization Strategy

1. **Eliminate all string-key lookups.** Replace `paste`-based keys with integer arithmetic: encode `(id, year)` as a single integer or use `data.table` keyed joins.

2. **Vectorize `build_neighbor_lookup` entirely.** Instead of iterating row-by-row, expand the neighbor relationships into a full edge table (cell_i, cell_j), join on year to get (row_i, row_j) pairs, then compute grouped statistics using `data.table`.

3. **Vectorize `compute_neighbor_stats`.** Instead of `lapply` over 6.46M elements, use `data.table` grouped aggregation on the edge table.

4. **Preserve the trained Random Forest model** — we only change feature engineering, producing identical numerical columns.

### Complexity Reduction
- Old: ~25.8M string constructions + named-vector lookups in an R-level loop = **hours**.
- New: One `data.table` merge + grouped aggregation = **minutes**.

---

## Working R Code

```r
library(data.table)

#' Optimized pipeline: replaces build_neighbor_lookup + compute_neighbor_stats
#' Produces identical numerical output (max, min, mean of neighbor values).
#'
#' @param cell_data        data.frame with columns: id, year, and all var columns
#' @param id_order         integer vector of cell IDs in the order matching the nb object
#' @param rook_neighbors   spdep::nb object (list of integer index vectors)
#' @param neighbor_source_vars character vector of variable names
#' @return cell_data with new neighbor feature columns appended

compute_all_neighbor_features_optimized <- function(cell_data,
                                                     id_order,
                                                     rook_neighbors,
                                                     neighbor_source_vars) {

  # ---- Step 1: Build directed edge list (in terms of cell IDs) ----
  # rook_neighbors[[i]] contains indices into id_order for the neighbors of id_order[i]
  from_idx <- rep(seq_along(rook_neighbors), lengths(rook_neighbors))
  to_idx   <- unlist(rook_neighbors)

  # Convert from nb indices to actual cell IDs
  edge_dt <- data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )

  # ---- Step 2: Convert cell_data to data.table and key it ----
  dt <- as.data.table(cell_data)

  # Create a row index to preserve original order
  dt[, .row_idx := .I]

  # We need to know which years exist
  years <- sort(unique(dt$year))

  # ---- Step 3: Cross edges with years to get (from_id, year) -> (to_id, year) ----
  # For each edge (from_id, to_id), the neighbor relationship holds across ALL years.
  # So we expand: for each year, (from_id, year) has neighbor row (to_id, year).

  # Build lookup: (id, year) -> row_idx
  # Use integer key for speed
  setkey(dt, id, year)

  # Expand edges × years using a merge approach:
  # "from" rows: merge edge_dt with dt on from_id = id to get (from_row, to_id, year)
  # Then merge on (to_id, year) to get to_row.

  # Step 3a: Get from-side row indices
  # For each edge, we need every year that from_id appears in.
  from_rows <- dt[, .(from_id = id, year, from_row = .row_idx)]
  setkey(from_rows, from_id)
  setkey(edge_dt, from_id)

  # Merge: for each edge, get all (from_row, year, to_id) combinations
  edge_year <- edge_dt[from_rows, on = "from_id", allow.cartesian = TRUE, nomatch = 0L]
  # Columns: from_id, to_id, year, from_row

  # Step 3b: Get to-side row indices
  to_rows <- dt[, .(to_id = id, year, to_row = .row_idx)]
  setkey(edge_year, to_id, year)
  setkey(to_rows, to_id, year)

  edge_full <- edge_year[to_rows, on = c("to_id", "year"), nomatch = 0L]
  # Columns: from_id, to_id, year, from_row, to_row

  # Free intermediate objects
  rm(from_rows, to_rows, edge_year)
  gc()

  # ---- Step 4: For each variable, compute grouped stats ----
  for (var_name in neighbor_source_vars) {

    # Extract neighbor values via to_row indexing (vectorized)
    edge_full[, neighbor_val := dt[[var_name]][to_row]]

    # Compute stats grouped by from_row, excluding NAs
    stats <- edge_full[!is.na(neighbor_val),
                       .(nmax  = max(neighbor_val),
                         nmin  = min(neighbor_val),
                         nmean = mean(neighbor_val)),
                       by = from_row]

    # Initialize new columns with NA
    max_col  <- paste0("n_max_", var_name)
    min_col  <- paste0("n_min_", var_name)
    mean_col <- paste0("n_mean_", var_name)

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Assign computed values back
    dt[stats$from_row, (max_col)  := stats$nmax]
    dt[stats$from_row, (min_col)  := stats$nmin]
    dt[stats$from_row, (mean_col) := stats$nmean]

    # Clean up
    edge_full[, neighbor_val := NULL]
    rm(stats)
  }

  # ---- Step 5: Restore original order and return as data.frame ----
  setorder(dt, .row_idx)
  dt[, .row_idx := NULL]

  as.data.frame(dt)
}

# ============================================================
# USAGE — drop-in replacement for the original outer loop
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features_optimized(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Predictions proceed as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Works and Performance Estimate

| Aspect | Original | Optimized |
|---|---|---|
| Lookup strategy | 6.46M `paste` + named-vector lookups in R loop | `data.table` keyed binary-search join |
| Neighbor expansion | Per-row `lapply` with string matching | Single vectorized merge (~38.4M rows) |
| Stat computation | `lapply` over 6.46M elements + `do.call(rbind)` | `data.table` grouped `by` aggregation (C-level) |
| Estimated time | **86+ hours** | **~5–15 minutes** on same hardware |
| Memory | Moderate (many small allocations) | ~2–4 GB peak for edge table (fits in 16 GB) |
| Numerical output | max, min, mean per row | **Identical** max, min, mean per row |
| RF model | Unchanged | **Unchanged** |

### Memory Note
The expanded edge table has ~38.4M rows × 5 columns of integers ≈ ~1.5 GB. With the `neighbor_val` double column added temporarily, peak usage is ~2–3 GB for the edge table plus the original data (~6.46M × 110 columns). This fits comfortably within 16 GB RAM.

**Summary:** The real bottleneck is `build_neighbor_lookup()` — specifically, 6.46 million iterations of string-key construction and named-vector lookups in an R-level loop. The fix is to replace all string-keyed row-level iteration with vectorized `data.table` joins and grouped aggregation, reducing runtime from 86+ hours to minutes while producing numerically identical features.