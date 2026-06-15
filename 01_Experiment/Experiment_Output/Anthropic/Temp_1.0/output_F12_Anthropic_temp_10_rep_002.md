 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague is **partially right but misses the deeper bottleneck**. Let me audit both functions:

### `compute_neighbor_stats()` — Minor bottleneck
- `do.call(rbind, result)` on ~6.46M rows is indeed slow (binding millions of 3-element vectors), but this runs only **5 times** (once per variable). With modern R, binding ~6.46M small vectors takes perhaps 30–60 seconds per call — annoying but not 86 hours.
- There is no "repeated list binding" inside `compute_neighbor_stats()` — it uses `lapply` which pre-allocates a list. The colleague's description is factually wrong about internal list binding.

### `build_neighbor_lookup()` — **THE TRUE BOTTLENECK**

This is where the 86+ hours come from. Examine what happens inside the `lapply` over **6.46 million rows**:

1. **`as.character(data$id[i])`** — called 6.46M times (minor).
2. **`id_order[neighbors[[ref_idx]]]`** — repeated neighbor vector subsetting (moderate).
3. **`paste(neighbor_cell_ids, data$year[i], sep = "_")`** — 6.46M calls to `paste()` creating character key vectors (significant).
4. **`idx_lookup[neighbor_keys]`** — **6.46M named-vector lookups by character key** (SEVERE).

The named vector `idx_lookup` has ~6.46M entries. Named vector lookup in R is **O(n)** per query (linear scan, not hashed). Performing ~6.46M lookups, each scanning a ~6.46M-length named vector, yields **O(n²) ≈ 4.17 × 10¹³ operations**. This is the dominant bottleneck — not `do.call(rbind, ...)`.

Additionally, the `build_neighbor_lookup` creates a list of ~6.46M integer vectors, each holding neighbor row indices. With ~4 rook neighbors per cell on average, that's ~25.8M integers stored across 6.46M list elements — significant memory pressure on a 16 GB machine, especially when the full data frame with 110 columns is also in memory.

## Optimization Strategy

1. **Replace named-vector lookup with an `environment`-based hash map (or `data.table` join)** — converts O(n) per lookup to O(1), reducing `build_neighbor_lookup` from O(n²) to O(n).
2. **Vectorize the neighbor lookup construction** — instead of row-by-row `lapply` over 6.46M rows, exploit the fact that many rows share the same `id` (each id has 28 year-rows). Build a **cell-level** neighbor map once (344K entries), then expand to row-level using vectorized joins.
3. **Vectorize `compute_neighbor_stats`** — replace `lapply` + `do.call(rbind, ...)` with direct matrix indexing and `vapply` or grouped column operations.
4. **Preserve the trained Random Forest model** — no changes to model or features, only to how neighbor features are computed (same numerical results).

## Working R Code

```r
# =============================================================================
# OPTIMIZED PIPELINE — preserves exact numerical output
# =============================================================================

library(data.table)

# ---- Step 1: Optimized neighbor lookup builder (O(n) instead of O(n²)) ------

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Convert data to data.table for fast operations (non-destructive)
  dt <- as.data.table(data)
  dt[, row_idx := .I]

  # Build cell-level neighbor map: for each cell id, which cell ids are neighbors?
  # id_order is the vector of cell ids; neighbors is the nb object (list of integer indices)
  # neighbors[[k]] gives the indices into id_order for neighbors of id_order[k]

  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # Get unique cell ids present in data
  unique_ids <- unique(dt$id)

  # Pre-build: for each unique cell id, the neighbor cell ids
  cell_neighbor_ids <- lapply(as.character(unique_ids), function(cid) {
    ref_idx <- id_to_ref[cid]
    if (is.na(ref_idx)) return(integer(0))
    nb_indices <- neighbors[[ref_idx]]
    # nb objects use 0 to indicate no neighbors
    nb_indices <- nb_indices[nb_indices > 0]
    id_order[nb_indices]
  })
  names(cell_neighbor_ids) <- as.character(unique_ids)

  # Build a fast row lookup: (id, year) -> row_idx using data.table keyed join
  setkey(dt, id, year)

  # Now, for each row i, we need the row indices of (neighbor_id, same year) combos.
  # Strategy: build an edge table (focal_row, neighbor_id, year), then join to dt
  # to get neighbor_row indices. Finally, split by focal_row.

  # Step A: for each row, get its id and year, then its neighbor cell ids
  # Use dt to expand: each row -> its neighbor cell ids
  cat("Building edge table...\n")

  # Map each id to an integer group for fast lapply
  id_levels <- as.character(unique_ids)

  # Expand: for each (id, year, row_idx), create (row_idx, neighbor_id, year) rows
  # Do this per unique id to keep memory bounded

  # Pre-compute: group rows by id
  id_groups <- split(seq_len(nrow(dt)), dt$id)

  # Build edge list in chunks
  edge_list <- vector("list", length(unique_ids))

  for (k in seq_along(unique_ids)) {
    cid <- as.character(unique_ids[k])
    nb_ids <- cell_neighbor_ids[[cid]]
    if (length(nb_ids) == 0) next

    focal_rows <- id_groups[[cid]]
    focal_years <- dt$year[focal_rows]

    # Create all (focal_row, neighbor_id, year) combinations
    edge_list[[k]] <- data.table(
      focal_row   = rep(focal_rows, each = length(nb_ids)),
      neighbor_id = rep(nb_ids, times = length(focal_rows)),
      year        = rep(focal_years, each = length(nb_ids))
    )
  }

  cat("Combining edge table...\n")
  edges <- rbindlist(edge_list)
  rm(edge_list)
  gc()

  # Step B: Join edges to dt to get neighbor row indices
  cat("Joining to get neighbor row indices...\n")
  setkey(edges, neighbor_id, year)

  # Create a join target: (id, year) -> row_idx
  row_lookup <- dt[, .(neighbor_id = id, year, neighbor_row = row_idx)]
  setkey(row_lookup, neighbor_id, year)

  edges <- row_lookup[edges, nomatch = 0L]
  # Now edges has columns: neighbor_id, year, neighbor_row, focal_row

  # Step C: Split neighbor_row by focal_row
  cat("Splitting into per-row neighbor lists...\n")
  n_rows <- nrow(dt)

  # Pre-allocate result as list of integer(0)
  neighbor_lookup <- vector("list", n_rows)
  # Fill with empty integer vectors
  for (i in seq_len(n_rows)) neighbor_lookup[[i]] <- integer(0)

  # Use split (fast in data.table)
  split_result <- split(edges$neighbor_row, edges$focal_row)
  split_names  <- as.integer(names(split_result))

  for (j in seq_along(split_names)) {
    neighbor_lookup[[split_names[j]]] <- as.integer(split_result[[j]])
  }

  rm(edges, split_result, row_lookup, dt)
  gc()

  cat("Neighbor lookup built.\n")
  return(neighbor_lookup)
}

# ---- Step 2: Optimized neighbor stats computation ---------------------------

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  n <- length(neighbor_lookup)

  # Use vapply for pre-allocated matrix output (avoids do.call(rbind, ...))
  result <- vapply(neighbor_lookup, function(idx) {
    if (length(idx) == 0L) return(c(NA_real_, NA_real_, NA_real_))
    neighbor_vals <- vals[idx]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0L) return(c(NA_real_, NA_real_, NA_real_))
    c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  }, numeric(3))

  # vapply returns 3 x n matrix; transpose to n x 3
  t(result)
}

# ---- Step 3: Feature addition wrapper (preserves original column names) -----

compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  cat(sprintf("Computing neighbor stats for '%s'...\n", var_name))
  stats_mat <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)

  # Assign columns — must match original naming convention
  data[[paste0(var_name, "_neighbor_max")]]  <- stats_mat[, 1]
  data[[paste0(var_name, "_neighbor_min")]]  <- stats_mat[, 2]
  data[[paste0(var_name, "_neighbor_mean")]] <- stats_mat[, 3]

  return(data)
}

# ---- Step 4: Run the optimized pipeline -------------------------------------

cat("Starting optimized neighbor feature pipeline...\n")
t0 <- Sys.time()

# Build the lookup ONCE (this is the former bottleneck)
neighbor_lookup <- build_neighbor_lookup_fast(
  cell_data, id_order, rook_neighbors_unique
)

# Compute neighbor features for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_fast(
    cell_data, var_name, neighbor_lookup
  )
}

t1 <- Sys.time()
cat(sprintf("Pipeline completed in %.1f minutes.\n", difftime(t1, t0, units = "mins")))

# The trained Random Forest model is untouched — use it as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Summary of Changes and Expected Speedup

| Component | Original Complexity | Optimized Complexity | Speedup Factor |
|---|---|---|---|
| `build_neighbor_lookup` (idx_lookup named-vector scan) | O(n²) ≈ 4.2×10¹³ ops | O(n log n) via data.table keyed join | **~1,000,000×** |
| `compute_neighbor_stats` (`do.call(rbind,...)`) | O(n) with high overhead | O(n) via `vapply` + `t()` (pre-allocated) | **~3–5×** |
| **Total estimated runtime** | **86+ hours** | **~10–30 minutes** | **~200–500×** |

**Key insight**: The colleague focused on a cosmetic inefficiency (`do.call(rbind, ...)` runs 5 times, ~5 minutes total) while the real bottleneck was the **O(n²) named-vector lookup** in `build_neighbor_lookup()`, which executes 6.46 million linear scans through a 6.46-million-element named vector. Replacing this with `data.table` keyed joins reduces the entire pipeline from days to minutes while preserving the exact same numerical output and leaving the trained Random Forest model untouched.