 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague is **partially right but mostly wrong** about the main bottleneck. Let me audit each claim:

**Claim 1: `do.call(rbind, result)` is a bottleneck.**
This is a minor concern. `do.call(rbind, ...)` on a list of ~6.46M fixed-length numeric vectors (each length 3) is not trivial, but it completes in seconds-to-minutes, not hours. It's a single matrix allocation and copy operation.

**Claim 2: "Repeated list binding inside `compute_neighbor_stats()`".**
There is **no repeated list binding** inside `compute_neighbor_stats()`. It uses `lapply` to produce a list in one pass, then binds once. This claim is factually wrong about the code.

**The actual deep bottleneck is `build_neighbor_lookup()`.**

Specifically, inside the `lapply` over **6.46 million rows**:

1. **`as.character(data$id[i])` and `id_to_ref[as.character(...)]`** — 6.46M individual character conversions and named-vector lookups.
2. **`paste(neighbor_cell_ids, data$year[i], sep = "_")`** — 6.46M calls to `paste()` constructing key vectors (each with ~4 neighbors on average for rook contiguity, so ~25.8M string constructions).
3. **`idx_lookup[neighbor_keys]`** — 6.46M named-vector lookups, where `idx_lookup` itself has **6.46M entries**. Named vector lookup in R is **O(n)** per query (linear scan or hash with overhead), making the total complexity approximately **O(n × k × m)** where n = 6.46M rows, k = avg neighbors (~4), and m = lookup overhead on a 6.46M-length named vector.

The named-vector `idx_lookup` with 6.46M elements is the **critical bottleneck**. Each lookup into this vector inside the row-level `lapply` is extremely expensive at scale. With ~6.46M iterations × ~4 lookups each, this is what produces the 86+ hour runtime.

`compute_neighbor_stats()`, by contrast, does only **integer indexing** (`vals[idx]`) which is O(1) per element — it is fast.

## Optimization Strategy

1. **Replace the row-level `lapply` in `build_neighbor_lookup()` with a vectorized approach** using `data.table` hash joins or environment-based hash lookups instead of named-vector lookups on a 6.46M-length vector.
2. **Pre-vectorize the entire neighbor-lookup construction** by expanding neighbors at the cell level (344K cells × ~4 neighbors = ~1.37M pairs), then joining with year in a single merge operation — eliminating the per-row loop entirely.
3. **Represent the lookup as a sparse adjacency structure** (list of integer row indices) built via `data.table` merge, not per-row string matching.
4. **Vectorize `compute_neighbor_stats()`** using the pre-built integer index lists, which is already reasonably fast but can be improved with `vapply` or direct matrix operations.

The key insight: instead of iterating 6.46M rows and doing string-based lookups each time, we:
- Build a `data.table` of all (cell_id, neighbor_id) pairs (~1.37M rows).
- Cross-join with years to get (cell_id, year, neighbor_id, year) → (row_index, neighbor_row_index) pairs (~1.37M × 28 ≈ 38.5M rows).
- Group by row_index to build the lookup list.

This replaces 6.46M sequential R-level iterations with a single vectorized `data.table` merge.

## Working R Code

```r
library(data.table)

# =============================================================================
# OPTIMIZED build_neighbor_lookup using data.table hash joins
# =============================================================================
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {

  # Convert data to data.table if not already (non-destructive)
  dt <- as.data.table(data)
  dt[, row_idx := .I]

  # Build a mapping from id to its position in id_order
  id_to_ref <- data.table(
    id  = id_order,
    ref = seq_along(id_order)
  )

  # Expand the nb object into a data.table of (focal_id, neighbor_id) pairs
  # neighbors is an nb object: a list of integer vectors (indices into id_order)
  edge_list <- rbindlist(lapply(seq_along(neighbors), function(i) {
    nb_idx <- neighbors[[i]]
    # nb objects use 0 for no-neighbor islands; filter those out
    nb_idx <- nb_idx[nb_idx > 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb_idx])
  }))

  # Build keyed lookup: (id, year) -> row_idx
  setkey(dt, id, year)

  # For each focal_id and year, find the row_idx of the focal cell
  # Then find row_idx of each neighbor_id in the same year

  # Step 1: Get all unique (focal_id, year) combinations present in data
  focal_dt <- dt[, .(id, year, focal_row_idx = row_idx)]

  # Step 2: Join edge_list with focal_dt to get (focal_row_idx, neighbor_id, year)
  # For each row in the data that has a focal_id in the edge list,
  # we expand to its neighbors
  setnames(focal_dt, "id", "focal_id")
  setkey(edge_list, focal_id)
  setkey(focal_dt, focal_id)

  # Merge: for each focal row, get all its neighbors
  expanded <- merge(focal_dt, edge_list, by = "focal_id", allow.cartesian = TRUE)
  # expanded has columns: focal_id, year, focal_row_idx, neighbor_id

  # Step 3: Look up the row index of each (neighbor_id, year) in the data
  neighbor_key <- dt[, .(neighbor_id = id, year, neighbor_row_idx = row_idx)]
  setkey(neighbor_key, neighbor_id, year)
  setkey(expanded, neighbor_id, year)

  matched <- merge(expanded, neighbor_key, by = c("neighbor_id", "year"), nomatch = 0L)
  # matched has: focal_row_idx, neighbor_row_idx (and other cols)

  # Step 4: Build the lookup list — group neighbor_row_idx by focal_row_idx
  setkey(matched, focal_row_idx)
  lookup_dt <- matched[, .(neighbors = list(neighbor_row_idx)), by = focal_row_idx]

  # Initialize result list (all rows, default to empty integer)
  n_rows <- nrow(dt)
  result <- vector("list", n_rows)
  for (i in seq_len(n_rows)) {
    result[[i]] <- integer(0L)
  }

  # Fill in the non-empty entries
  result[lookup_dt$focal_row_idx] <- lookup_dt$neighbors

  return(result)
}


# =============================================================================
# OPTIMIZED compute_neighbor_stats using vapply
# =============================================================================
compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  n <- length(neighbor_lookup)

  # Use vapply for type-stable output, returns 3 x n matrix
  stats_mat <- vapply(neighbor_lookup, function(idx) {
    if (length(idx) == 0L) return(c(NA_real_, NA_real_, NA_real_))
    neighbor_vals <- vals[idx]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0L) return(c(NA_real_, NA_real_, NA_real_))
    c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  }, numeric(3))

  # vapply returns 3 x n matrix; transpose to n x 3
  t(stats_mat)
}


# =============================================================================
# Wrapper: compute and add neighbor features (preserves original column naming)
# =============================================================================
compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  stats <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)
  data[[paste0(var_name, "_neighbor_max")]]  <- stats[, 1]
  data[[paste0(var_name, "_neighbor_min")]]  <- stats[, 2]
  data[[paste0(var_name, "_neighbor_mean")]] <- stats[, 3]
  data
}


# =============================================================================
# MAIN PIPELINE — drop-in replacement
# =============================================================================
# Build neighbor lookup ONCE (this is the step that was 86+ hours; now ~minutes)
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor features for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, var_name, neighbor_lookup)
}

# The trained Random Forest model is untouched.
# The numerical estimand is preserved: same max/min/mean neighbor statistics,
# same column names, same row order.
```

## Expected Performance

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~80–86 hours (6.46M row-level string lookups into 6.46M named vector) | **~2–5 minutes** (single `data.table` merge of ~38.5M rows) |
| `compute_neighbor_stats` (×5 vars) | ~minutes (integer indexing, already OK) | **~1–2 minutes** (`vapply` avoids `do.call(rbind,...)` overhead) |
| **Total** | **86+ hours** | **~5–10 minutes** |

## Summary

The colleague's diagnosis was **incorrect**. `do.call(rbind, result)` and the alleged "repeated list binding" (which doesn't exist in the code) are not the bottleneck. The true bottleneck is the **O(n²)-behaving named-vector lookup** in `build_neighbor_lookup()`, where 6.46M rows each perform string-key lookups into a 6.46M-element named vector. Replacing this with `data.table` hash-based equi-joins eliminates the bottleneck entirely while preserving the exact same numerical output and leaving the trained Random Forest model untouched.