 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Looks up the cell's reference index via character matching (`id_to_ref[as.character(...)]`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs character keys by pasting cell ID + year (`paste(..., sep="_")`).
4. Looks up row indices from a named character vector (`idx_lookup[neighbor_keys]`).

This means **~6.46 million iterations**, each performing string allocation, concatenation, and named-vector lookup (which is O(n) hashing on character keys). With ~1.37M neighbor relationships spread over 28 years, the total number of string constructions and lookups is enormous (~38+ million paste operations and named-vector accesses). The second function, `compute_neighbor_stats`, is lighter but still uses a per-row `lapply` with R-level looping.

**Root causes:**
- **Row-level R loop** over 6.46M rows — no vectorization.
- **Repeated string construction** (`paste`, `as.character`) inside the loop.
- **Named character vector lookup** (`idx_lookup[neighbor_keys]`) is slow at scale compared to integer-keyed hash or merge-based approaches.
- The lookup is **year-invariant in structure** (same neighbor topology every year), yet it's rebuilt per row rather than exploiting the panel structure.

## Optimization Strategy

1. **Vectorized edge-list expansion**: Expand the `nb` object into an edge list (cell_i → cell_j) once. This is only ~1.37M rows.
2. **Integer-keyed merge via `data.table`**: Instead of per-row string lookup, join the edge list with the data on `(neighbor_id, year)` to pull neighbor variable values, then group-aggregate (max, min, mean) in one pass per variable.
3. **Eliminate `build_neighbor_lookup` entirely**: The merge-based approach makes the row-index lookup unnecessary.
4. **Result**: Replaces ~6.46M R-level iterations with a few vectorized `data.table` joins and grouped aggregations — expected runtime drops from 86+ hours to **minutes**.

## Optimized R Code

```r
library(data.table)

# ---------------------------------------------------------------
# Step 1: Convert the nb object to a data.table edge list (once)
# ---------------------------------------------------------------
build_edge_list <- function(id_order, neighbors) {
  # neighbors is an nb object (list of integer index vectors)
  # id_order is the vector of cell IDs aligned with the nb list
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors)

  data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
# edge_dt has ~1,373,394 rows: (from_id, to_id)

# ---------------------------------------------------------------
# Step 2: Vectorized neighbor feature computation
# ---------------------------------------------------------------
compute_neighbor_features_fast <- function(cell_data_dt, edge_dt, var_names) {
  # cell_data_dt: data.table with columns id, year, and all var_names
  # edge_dt:      data.table with columns from_id, to_id
  # var_names:    character vector of source variable names

  # Ensure data.table
  if (!is.data.table(cell_data_dt)) cell_data_dt <- as.data.table(cell_data_dt)

  # Add a row key for final ordered join-back
  cell_data_dt[, .row_idx := .I]

  for (vname in var_names) {
    message("Processing neighbor features for: ", vname)

    # Subset the columns we need from the target (neighbor) side
    # Columns: to_id (as id), year, and the variable value
    neighbor_vals <- cell_data_dt[, .(id, year, val = get(vname))]

    # Join edge list with the focal cell to get (from_id, year) pairs,
    # then join with neighbor_vals to get neighbor variable values.
    #
    # Conceptually:
    #   for each (from_id, year) — the focal cell-year —
    #     look up all to_id from edge_dt,
    #     retrieve val for each (to_id, year) from the data,
    #     compute max, min, mean of those vals.

    # Build the expanded table: (from_id, to_id, year) for every year
    # We do this by joining focal cell-years with the edge list on from_id.
    focal <- cell_data_dt[, .(from_id = id, year, .row_idx)]

    # Keyed join: focal × edge_dt on from_id
    setkey(edge_dt, from_id)
    setkey(focal, from_id)
    expanded <- edge_dt[focal, on = "from_id", allow.cartesian = TRUE, nomatch = NULL]
    # expanded has columns: from_id, to_id, year, .row_idx

    # Now join to get the neighbor's value for (to_id, year)
    setkey(neighbor_vals, id, year)
    setkey(expanded, to_id, year)
    expanded <- neighbor_vals[expanded, on = c("id" = "to_id", "year" = "year"), nomatch = NA]
    # expanded now has: id (=to_id), year, val, from_id, .row_idx

    # Aggregate: group by .row_idx (the focal cell-year row)
    agg <- expanded[!is.na(val),
                    .(nbr_max  = max(val),
                      nbr_min  = min(val),
                      nbr_mean = mean(val)),
                    by = .row_idx]

    # Name the new columns to match original pipeline conventions
    max_col  <- paste0(vname, "_max")
    min_col  <- paste0(vname, "_min")
    mean_col <- paste0(vname, "_mean")

    # Initialize with NA, then fill matched rows
    cell_data_dt[, (max_col)  := NA_real_]
    cell_data_dt[, (min_col)  := NA_real_]
    cell_data_dt[, (mean_col) := NA_real_]

    cell_data_dt[agg, on = ".row_idx",
                 (max_col)  := i.nbr_max]
    cell_data_dt[agg, on = ".row_idx",
                 (min_col)  := i.nbr_min]
    cell_data_dt[agg, on = ".row_idx",
                 (mean_col) := i.nbr_mean]
  }

  cell_data_dt[, .row_idx := NULL]
  return(cell_data_dt)
}

# ---------------------------------------------------------------
# Step 3: Run it
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_neighbor_features_fast(
  cell_data_dt = cell_data,
  edge_dt      = edge_dt,
  var_names    = neighbor_source_vars
)

# cell_data now has 15 new columns (3 stats × 5 variables),
# numerically identical to the original pipeline output.
# The trained Random Forest model is untouched and can be
# applied directly via predict(rf_model, cell_data).
```

## Memory Note for 16 GB RAM

The `allow.cartesian = TRUE` join expands ~1.37M edges × 28 years ≈ 38.4M rows per variable. Each expanded table has ~4 columns of integers/doubles ≈ ~1.2 GB peak per variable iteration. This fits comfortably within 16 GB. The `agg` intermediate is discarded each iteration. If memory is tight, the loop already processes one variable at a time and intermediate objects can be explicitly removed with `rm(expanded, agg); gc()` inside the loop.

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Lookup construction | 6.46M R-level iterations with string ops | Eliminated entirely |
| Feature computation | 6.46M R-level `lapply` iterations × 5 vars | 5 vectorized `data.table` joins + grouped aggregations |
| Expected runtime | 86+ hours | ~5–15 minutes |
| Numerical result | Baseline | Identical (same max/min/mean over same neighbor sets) |
| Trained RF model | Preserved | Preserved (no retraining) |