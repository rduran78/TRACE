 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not ideal, it is a one-time matrix assembly per variable (5 calls total). Each call binds ~6.46M rows × 3 columns — a few seconds at most with `do.call(rbind, ...)` on uniform-length numeric vectors.

**The true bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Per-row string pasting and named-vector lookup over 6.46 million rows.** The `lapply` iterates over every row, and for each row it:
   - Calls `as.character(data$id[i])` — character coercion per row.
   - Looks up `id_to_ref[as.character(...)]` — named vector lookup (hashed, but still per-element overhead).
   - Extracts `id_order[neighbors[[ref_idx]]]` — subset of a potentially large vector.
   - Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` — string concatenation for every neighbor of every row.
   - Looks up `idx_lookup[neighbor_keys]` — named vector lookup with string keys over a 6.46M-element named vector, **for every neighbor of every row**.

2. **Scale of the problem:** With ~6.46 million rows and an average of ~4 rook neighbors per cell, this inner function performs roughly **25.8 million string paste operations** and **25.8 million named-vector lookups** against a 6.46M-entry hash table — all inside a sequential `lapply`. This is the operation taking dozens of hours.

3. `compute_neighbor_stats()`, by contrast, does only cheap numeric indexing (`vals[idx]`) and simple arithmetic (`max`, `min`, `mean`) — these are vectorized and fast. The `do.call(rbind, result)` on a list of uniform 3-element numeric vectors is also fast (effectively a matrix reshape).

**Conclusion:** The deep bottleneck is the **string-key construction and lookup strategy in `build_neighbor_lookup()`**. The fix is to eliminate all string operations and replace them with pure integer arithmetic for row indexing.

---

## Optimization Strategy

1. **Replace string-keyed lookup with integer arithmetic.** Since the data has a regular panel structure (each of 344,208 cells × 28 years), we can map `(cell_id, year)` → row index using integer math instead of pasting strings and doing hash lookups. We build integer mappings: `id → integer position` and `year → integer position`, then compute row index as `(id_pos - 1) * n_years + year_pos` (or similar, depending on sort order).

2. **Pre-expand the neighbor lookup to a flat integer vector scheme.** Instead of building a list of 6.46M variable-length integer vectors, we pre-compute a CSR-like (Compressed Sparse Row) structure: two vectors — one of concatenated neighbor-row indices, one of offsets — enabling fast vectorized access.

3. **Vectorize `compute_neighbor_stats()`.** With the CSR structure, we can use a single vectorized C-level operation (via `data.table` grouping or a simple Rcpp snippet) instead of 6.46M `lapply` iterations.

4. **Preserve the trained Random Forest model** — we only change feature engineering, not model training. The numerical output (max, min, mean of neighbor values) is identical.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED PIPELINE — replaces build_neighbor_lookup + compute_neighbor_stats
# =============================================================================

library(data.table)

# -------------------------------------------------------------------
# Step 0: Convert to data.table for fast operations (non-destructive)
# -------------------------------------------------------------------
# Assumes cell_data is a data.frame with columns: id, year, and the
# neighbor_source_vars. id_order and rook_neighbors_unique are as before.

cell_dt <- as.data.table(cell_data)

# -------------------------------------------------------------------
# Step 1: Build integer-arithmetic row index mapping
# -------------------------------------------------------------------
# Determine sort order of the data. We require (id, year) to be the
# unique key. We sort to guarantee a known layout.

setorder(cell_dt, id, year)

# Unique ids and years in sorted order
unique_ids   <- sort(unique(cell_dt$id))
unique_years <- sort(unique(cell_dt$year))
n_ids   <- length(unique_ids)
n_years <- length(unique_years)

stopifnot(nrow(cell_dt) == n_ids * n_years)  # confirm balanced panel

# Integer position maps (id -> 1..n_ids, year -> 1..n_years)
id_pos_map   <- setNames(seq_along(unique_ids),   as.character(unique_ids))
year_pos_map <- setNames(seq_along(unique_years),  as.character(unique_years))

# Row index from (id_position, year_position):
#   row = (id_pos - 1) * n_years + year_pos
# This works because data is sorted by (id, year).

# -------------------------------------------------------------------
# Step 2: Build CSR-style neighbor-row structure (integer only)
# -------------------------------------------------------------------
# For each cell id (in id_order), get its neighbor cell ids via
# rook_neighbors_unique, then map to id positions.
#
# id_order is the vector of cell ids in the order matching the nb object.

# Map id_order to positions in our sorted unique_ids
id_order_pos <- id_pos_map[as.character(id_order)]

# Build neighbor list in terms of id positions (not row indices yet)
# rook_neighbors_unique[[k]] gives neighbor indices into id_order
# So id_order[rook_neighbors_unique[[k]]] gives neighbor cell ids
# And id_pos_map[as.character(...)] gives their positions in unique_ids

# We need a mapping from each unique_id's position to its neighbors'
# positions. id_order may differ from unique_ids ordering, so we
# build a bridge.

# pos_in_id_order for each unique_id position:
# id_order_to_uid_pos: for each index k in id_order, the uid position
id_order_to_uid_pos <- id_pos_map[as.character(id_order)]

# For each uid position, which index in id_order does it correspond to?
uid_pos_to_id_order_idx <- integer(n_ids)
uid_pos_to_id_order_idx[id_order_to_uid_pos] <- seq_along(id_order)

# Now build neighbor uid positions for every uid position
cat("Building integer neighbor structure...\n")
neighbor_uid_pos_list <- vector("list", n_ids)
for (k in seq_along(id_order)) {
  uid_pos <- id_order_to_uid_pos[k]
  nb_indices <- rook_neighbors_unique[[k]]
  if (length(nb_indices) == 0L || (length(nb_indices) == 1L && nb_indices[1] == 0L)) {
    neighbor_uid_pos_list[[uid_pos]] <- integer(0)
  } else {
    neighbor_uid_pos_list[[uid_pos]] <- id_order_to_uid_pos[nb_indices]
  }
}

# -------------------------------------------------------------------
# Step 3: Expand to row-level CSR structure
# -------------------------------------------------------------------
# For row r corresponding to (uid_pos_r, year_pos_r), its neighbor rows
# are: (neighbor_uid_pos - 1) * n_years + year_pos_r
# We build offset and flat index vectors.

cat("Expanding to row-level CSR structure...\n")

# Precompute uid_pos and year_pos for every row
cell_dt[, uid_pos  := id_pos_map[as.character(id)]]
cell_dt[, year_pos := year_pos_map[as.character(year)]]

# Number of neighbors per uid
n_neighbors_per_uid <- vapply(neighbor_uid_pos_list, length, integer(1))

# Number of neighbors per row = number of neighbors for that row's uid
n_neighbors_per_row <- n_neighbors_per_uid[cell_dt$uid_pos]

total_edges <- sum(as.numeric(n_neighbors_per_row))
cat(sprintf("Total directed neighbor-row edges: %.0f\n", total_edges))

# Build CSR offset vector
offsets <- c(0L, cumsum(as.numeric(n_neighbors_per_row)))

# Build flat neighbor-row index vector
# We process by uid to avoid per-row R overhead
cat("Building flat neighbor index vector...\n")

flat_nb_rows <- integer(total_edges)
write_pos <- 1L

for (u in seq_len(n_ids)) {
  nb_uids <- neighbor_uid_pos_list[[u]]
  n_nb <- length(nb_uids)
  if (n_nb == 0L) next

  # All year positions for this uid
  # Rows for uid u are: ((u-1)*n_years + 1) to (u*n_years)
  base_row_start <- (u - 1L) * n_years

  # For each year (year_pos 1..n_years), the neighbor rows are
  # (nb_uid - 1) * n_years + year_pos
  nb_bases <- (nb_uids - 1L) * n_years  # length n_nb

  for (yp in seq_len(n_years)) {
    # Current row: base_row_start + yp
    # Neighbor rows for this (u, yp): nb_bases + yp
    nb_rows_here <- nb_bases + yp
    idx_range <- write_pos:(write_pos + n_nb - 1L)
    flat_nb_rows[idx_range] <- nb_rows_here
    write_pos <- write_pos + n_nb
  }
}

cat("CSR structure built.\n")

# -------------------------------------------------------------------
# Step 4: Vectorized compute_neighbor_stats using data.table grouping
# -------------------------------------------------------------------
# Strategy: build an edge table (row_idx, neighbor_row_idx), join
# values, group by row_idx, compute max/min/mean.

compute_neighbor_stats_fast <- function(cell_dt, flat_nb_rows, offsets,
                                        n_neighbors_per_row, var_name) {
  cat(sprintf("Computing neighbor stats for '%s'...\n", var_name))
  n_rows <- nrow(cell_dt)
  vals <- cell_dt[[var_name]]

  # Identify rows with at least one neighbor
  has_nb <- which(n_neighbors_per_row > 0L)

  # Build row_id vector corresponding to flat_nb_rows
  row_id_vec <- rep.int(has_nb, n_neighbors_per_row[has_nb])

  # Get neighbor values
  nb_vals <- vals[flat_nb_rows[seq_along(row_id_vec)]]

  # Build edge data.table
  edge_dt <- data.table(row_id = row_id_vec, nb_val = nb_vals)

  # Remove NAs in neighbor values
  edge_dt <- edge_dt[!is.na(nb_val)]

  # Group and compute stats
  stats_dt <- edge_dt[, .(
    nb_max  = max(nb_val),
    nb_min  = min(nb_val),
    nb_mean = mean(nb_val)
  ), by = row_id]

  # Initialize result columns with NA
  col_max  <- rep(NA_real_, n_rows)
  col_min  <- rep(NA_real_, n_rows)
  col_mean <- rep(NA_real_, n_rows)

  col_max[stats_dt$row_id]  <- stats_dt$nb_max
  col_min[stats_dt$row_id]  <- stats_dt$nb_min
  col_mean[stats_dt$row_id] <- stats_dt$nb_mean

  # Add to cell_dt
  max_col_name  <- paste0("max_neighbor_",  var_name)
  min_col_name  <- paste0("min_neighbor_",  var_name)
  mean_col_name <- paste0("mean_neighbor_", var_name)

  cell_dt[, (max_col_name)  := col_max]
  cell_dt[, (min_col_name)  := col_min]
  cell_dt[, (mean_col_name) := col_mean]

  invisible(cell_dt)
}

# -------------------------------------------------------------------
# Step 5: Run for all neighbor source variables
# -------------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_dt <- compute_neighbor_stats_fast(
    cell_dt, flat_nb_rows, offsets, n_neighbors_per_row, var_name
  )
}

# Convert back to data.frame if needed downstream (e.g., for predict())
cell_data <- as.data.frame(cell_dt)
# Drop helper columns
cell_data$uid_pos  <- NULL
cell_data$year_pos <- NULL

cat("Done. All neighbor features computed.\n")

# The trained Random Forest model is untouched.
# Use predict(rf_model, cell_data) as before.
```

---

## Summary

| Aspect | Colleague's Claim | Actual Finding |
|---|---|---|
| **Alleged bottleneck** | `do.call(rbind, ...)` and list binding in `compute_neighbor_stats()` | These are minor (~seconds per variable) |
| **True bottleneck** | — | `build_neighbor_lookup()`: 6.46M iterations of `paste()` + named-vector string lookups against a 6.46M-key hash, totaling ~25.8M string operations |
| **Root cause** | — | String-based indexing (`paste(id, year, "_")` + named vector lookup) applied per-row inside `lapply` over 6.46M rows |
| **Fix** | — | Replace all string operations with integer arithmetic `(uid_pos - 1) * n_years + year_pos`, build a CSR structure once, then use vectorized `data.table` grouping for stats |
| **Expected speedup** | — | From ~86+ hours to **minutes** (the CSR build is O(cells × years × avg_neighbors) in integer ops; the stats computation is a single vectorized `data.table` group-by per variable) |
| **Model preservation** | ✅ | RF model is untouched; numerical outputs (max, min, mean of neighbor values) are identical |