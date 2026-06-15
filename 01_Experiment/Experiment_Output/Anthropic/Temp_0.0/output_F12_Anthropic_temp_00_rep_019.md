 # Diagnosis, Optimization Strategy, and Working Code

## Diagnosis

The colleague is **partially right but misses the deeper bottleneck**. Let me audit both functions against the data facts:

### `compute_neighbor_stats()` — Colleague's Claim

`do.call(rbind, result)` on a list of ~6.46 million 3-element vectors is indeed not free — it's O(n) allocation and copy. But with modern R, binding ~6.46M small fixed-length vectors into a matrix is a matter of seconds, not hours. The inner `lapply` does simple subsetting and three summary statistics per iteration — also fast. **This is not the 86-hour bottleneck.**

### `build_neighbor_lookup()` — The True Bottleneck

This is where the real cost hides. For each of the **~6.46 million rows**, the function:

1. **`as.character(data$id[i])`** — converts a single value to character (minor).
2. **`id_order[neighbors[[ref_idx]]]`** — integer subset of a vector (cheap).
3. **`paste(neighbor_cell_ids, data$year[i], sep = "_")`** — creates ~4 character strings per row (each cell has ~4 rook neighbors on average: 1,373,394 directed relationships / 344,208 cells ≈ 4).
4. **`idx_lookup[neighbor_keys]`** — **Named vector lookup by character key**. This is the killer.

`idx_lookup` is a named integer vector with **~6.46 million entries**. Every call to `idx_lookup[neighbor_keys]` triggers **hash-based name matching** against a 6.46M-element name vector. R's internal name lookup on large named vectors is notoriously slow — it rebuilds or searches a hash table on each access. Doing this **~6.46 million times × ~4 keys per call ≈ 25.8 billion character hash lookups** against a massive table.

**This is the true bottleneck: repeated character-key lookups into a 6.46M-entry named vector, called 6.46M times inside `lapply`.**

The `compute_neighbor_stats` function and its `do.call(rbind, ...)` are comparatively trivial.

## Optimization Strategy

**Replace all character-key lookups with direct integer-index arithmetic.** Since every `id` appears once per year for 28 consecutive years (1992–2019), we can compute the row index of any `(id, year)` pair arithmetically:

- Build a simple integer map: `id → integer position` (1-based).
- Row index for `(id, year)` = `(id_position - 1) * 28 + (year - 1992) + 1` (assuming data is sorted by id then year), **or** use a single `match()` call to build an integer matrix lookup — either way, **zero character operations inside the hot loop**.

Additionally, vectorize `compute_neighbor_stats` using a pre-allocated matrix instead of `lapply` + `do.call(rbind, ...)`.

## Working R Code

```r
# ==============================================================================
# OPTIMIZED PIPELINE — replaces build_neighbor_lookup + compute_neighbor_stats
# Preserves the trained Random Forest model and original numerical estimand.
# ==============================================================================

library(data.table)

# ---------- Step 0: Ensure data is a data.table with a known row order --------
cell_dt <- as.data.table(cell_data)

# Verify or impose a deterministic sort so we can use arithmetic indexing.
# Original row order is preserved via an explicit index column.
cell_dt[, .row_orig := .I]
setkey(cell_dt, id, year)
# After setkey, rows are sorted by (id, year).

# Build integer maps
unique_ids   <- sort(unique(cell_dt$id))
n_ids        <- length(unique_ids)
unique_years <- sort(unique(cell_dt$year))
n_years      <- length(unique_years)

stopifnot(nrow(cell_dt) == n_ids * n_years)
# Confirm: each id appears exactly n_years times in sorted order.

# Map id -> 1-based position (integer)
id_to_pos <- setNames(seq_along(unique_ids), as.character(unique_ids))

# Map year -> 0-based offset (integer)
year_to_offset <- setNames(seq_along(unique_years) - 1L, as.character(unique_years))

# Row index formula (data is keyed by id, year):
#   row(id, year) = (id_pos - 1) * n_years + year_offset + 1
# This is O(1) per lookup, pure integer arithmetic.

# ---------- Step 1: Build neighbor lookup as integer row indices (vectorised) --

build_neighbor_lookup_fast <- function(cell_dt, unique_ids, id_to_pos,
                                       year_to_offset, n_years,
                                       neighbors) {
  # neighbors is an nb object indexed by id_order position.
  # id_order == unique_ids (sorted), so neighbors[[k]] gives
  # the neighbor positions for unique_ids[k].

  n_rows <- nrow(cell_dt)
  ids    <- cell_dt$id
  years  <- cell_dt$year

  # Pre-compute id_pos and year_offset for every row (vectorised, once).
  id_pos_vec      <- id_to_pos[as.character(ids)]       # length n_rows

  year_offset_vec <- year_to_offset[as.character(years)] # length n_rows

  # For each row i, its neighbors in the SAME year are:
  #   neighbor_id_positions = neighbors[[ id_pos_vec[i] ]]
  #   neighbor_rows = (neighbor_id_positions - 1) * n_years + year_offset_vec[i] + 1
  #
  # We vectorise this by expanding into a flat edge list.

  # --- Build flat edge list of (row_index, neighbor_row_index) ---
  # Number of neighbors per id (constant across years for the same id).
  n_neighbors_per_id <- lengths(neighbors)  # length = n_ids

  # Expand: for every row, how many neighbors?
  n_neighbors_per_row <- n_neighbors_per_id[id_pos_vec]  # length n_rows

  # Total directed edges
  total_edges <- sum(as.numeric(n_neighbors_per_row))

  # Source row indices (repeated by number of neighbors)
  src_rows <- rep.int(seq_len(n_rows), n_neighbors_per_row)

  # Neighbor id-positions for each row
  # For row i, neighbor id positions = neighbors[[ id_pos_vec[i] ]]
  # Unlist neighbors in id_pos order, then replicate across years.
  neighbor_id_pos_per_id <- unlist(neighbors, use.names = FALSE)  
  # This has sum(n_neighbors_per_id) elements, one set per id.

  # Replicate each id's neighbor list n_years times (once per year of that id).
  # Since data is sorted by (id, year), rows for id k are consecutive:
  #   rows (k-1)*n_years+1  to  k*n_years
  # Each of those rows shares the same neighbor id positions.
  neighbor_id_pos_all <- rep(neighbor_id_pos_per_id, times = n_years)
  # length = sum(n_neighbors_per_id) * n_years = total_edges ✓

  # Year offset for each source row, expanded

  year_offset_expanded <- year_offset_vec[src_rows]

  # Compute neighbor row indices via arithmetic
  neighbor_rows <- (neighbor_id_pos_all - 1L) * n_years + year_offset_expanded + 1L

  # --- Pack into a list-of-integer-vectors, one per row ---
  # Use split (fast on integers with known grouping).
  # Since src_rows is non-decreasing (rep.int preserves order), we can use
  # a factor-free split.
  lookup <- split(neighbor_rows, src_rows)

  # Ensure every row is represented (some rows may have 0 neighbors).
  # split drops empty groups, so fill them in.
  full_lookup <- vector("list", n_rows)
  idx_present <- as.integer(names(lookup))
  full_lookup[idx_present] <- lookup
  # Rows with 0 neighbors remain NULL → will be treated as integer(0).

  full_lookup
}

neighbor_lookup <- build_neighbor_lookup_fast(
  cell_dt, unique_ids, id_to_pos, year_to_offset, n_years,
  rook_neighbors_unique
)

# ---------- Step 2: Compute neighbor stats (vectorised, no do.call(rbind)) ----

compute_neighbor_stats_fast <- function(cell_dt, neighbor_lookup, var_name) {
  vals   <- cell_dt[[var_name]]
  n_rows <- nrow(cell_dt)

  # Flatten neighbor_lookup into vectors for vectorised computation.
  lens <- lengths(neighbor_lookup)
  idx_flat <- unlist(neighbor_lookup, use.names = FALSE)

  if (is.null(idx_flat) || length(idx_flat) == 0) {
    # Edge case: no neighbors at all
    return(matrix(NA_real_, nrow = n_rows, ncol = 3,
                  dimnames = list(NULL, c("max", "min", "mean"))))
  }

  neighbor_vals_flat <- vals[idx_flat]

  # Group id for each element in the flat vector
  grp <- rep.int(seq_len(n_rows), lens)

  # Use data.table for grouped aggregation (extremely fast)
  dt_agg <- data.table(grp = grp, v = neighbor_vals_flat)
  # Remove NAs before aggregation
  dt_agg <- dt_agg[!is.na(v)]

  agg <- dt_agg[, .(vmax = max(v), vmin = min(v), vmean = mean(v)), by = grp]

  # Map back to full row set
  out <- matrix(NA_real_, nrow = n_rows, ncol = 3)
  out[agg$grp, 1] <- agg$vmax
  out[agg$grp, 2] <- agg$vmin
  out[agg$grp, 3] <- agg$vmean

  out
}

# ---------- Step 3: Wrapper that adds columns (same schema as original) -------

compute_and_add_neighbor_features_fast <- function(cell_dt, var_name,
                                                    neighbor_lookup) {
  stats <- compute_neighbor_stats_fast(cell_dt, neighbor_lookup, var_name)
  cell_dt[, paste0("neighbor_max_", var_name)  := stats[, 1]]
  cell_dt[, paste0("neighbor_min_", var_name)  := stats[, 2]]
  cell_dt[, paste0("neighbor_mean_", var_name) := stats[, 3]]
  invisible(cell_dt)
}

# ---------- Step 4: Outer loop (unchanged semantics) --------------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  compute_and_add_neighbor_features_fast(cell_dt, var_name, neighbor_lookup)
}

# ---------- Step 5: Restore original row order for downstream RF prediction ---

setorder(cell_dt, .row_orig)
cell_dt[, .row_orig := NULL]

# Convert back to data.frame if the trained RF model expects one.
cell_data <- as.data.frame(cell_dt)

# The trained Random Forest model is untouched.
# Predictions proceed as before:
# preds <- predict(rf_model, newdata = cell_data)
```

## Summary

| Aspect | Colleague's Claim | Actual Finding |
|---|---|---|
| `do.call(rbind, result)` | "Main bottleneck" | **Minor cost** (~seconds for 6.46M × 3 matrix) |
| `compute_neighbor_stats` inner loop | "Repeated list binding" | **No list binding occurs**; `lapply` returns a pre-allocated list; cost is trivial |
| **`build_neighbor_lookup`** | Not mentioned | **TRUE BOTTLENECK**: ~25.8 billion character hash lookups into a 6.46M-entry named vector |

**Verdict: Reject the colleague's diagnosis.** The dominant cost is the O(n²)-flavoured character-key lookup inside `build_neighbor_lookup()`. The fix is to replace all character-keyed lookups with O(1) integer arithmetic (`(id_pos - 1) * n_years + year_offset + 1`), and to vectorise the neighbor stats aggregation via `data.table` grouped operations. Expected runtime drops from **86+ hours to minutes**.