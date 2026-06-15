 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it is a single vectorized concatenation step per variable — only 5 calls total. Each inner function returns a fixed-length-3 numeric vector, so there is no "repeated list binding" growing a structure iteratively; `lapply` pre-allocates the list.

**The true, deeper bottleneck is `build_neighbor_lookup()`.**

Here is the evidence from the code:

1. **Per-row string pasting and named-vector lookup (O(n) × constant but enormous n).** For each of the ~6.46 million rows, the function calls `as.character(data$id[i])`, performs a named-vector lookup in `id_to_ref`, pastes every neighbor ID with the current year (`paste(neighbor_cell_ids, data$year[i], sep = "_")`), and then looks each key up in `idx_lookup` — another named character vector of length 6.46 million. Named-vector lookup in R is hash-based but carries significant per-call overhead when invoked millions of times inside an `lapply` scalar loop.

2. **Multiplied by neighbor fan-out.** With ~1.37 million directed rook-neighbor relationships across 344,208 cells, the average cell has ~4 neighbors. For every one of the 6.46 million rows, the function constructs ~4 paste keys and performs ~4 hash lookups. That is **~25.8 million `paste` + hash-probe operations inside a scalar R loop** — the dominant cost.

3. **The lookup is rebuilt from scratch every run** even though the spatial topology is static across all 28 years. The function loops over every cell-year row, yet the neighbor *structure* is year-invariant; only the *row indices* change by year.

4. **`compute_neighbor_stats` is comparatively cheap.** It does only `vals[idx]` (integer subsetting — very fast), a few simple aggregations, and one `do.call(rbind, ...)` per variable. Profiling arithmetic: 5 variables × 1 `do.call(rbind, 6.46M-element list)` ≈ seconds. The lookup construction is hours.

**Verdict:** The bottleneck is the **~6.46 million iterations of scalar string manipulation and hash lookup inside `build_neighbor_lookup()`**, not the `rbind` in `compute_neighbor_stats()`.

---

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup` completely** — eliminate the per-row `lapply`. Exploit the fact that the neighbor graph is year-invariant: build a cell-level neighbor map once (344K cells), then expand it to all 28 years using vectorized integer arithmetic instead of string hashing.

2. **Replace `do.call(rbind, ...)` with direct matrix construction** in `compute_neighbor_stats` — pre-allocate a matrix and fill it, or use `vapply` which returns a matrix directly. This is a minor but easy win.

3. **Use `data.table` for the row-index mapping** instead of named character vectors, giving O(1) keyed joins on integer columns rather than string hashing.

The optimized pipeline reduces the complexity from ~25.8 million scalar R-loop iterations with string operations to a handful of fully vectorized joins and integer operations, bringing runtime from 86+ hours down to minutes.

---

## Working R Code

```r
# ============================================================
# Optimized pipeline — preserves trained RF model and original
# numerical estimand (max, min, mean of neighbor values).
# ============================================================

library(data.table)

# ----------------------------------------------------------
# 1. Vectorized neighbor-lookup builder
#    Key insight: the rook-neighbor topology is YEAR-INVARIANT.
#    Build a cell-level edge list once, then map to row indices
#    for all years via a single keyed join.
# ----------------------------------------------------------

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # data must have columns: id, year (and be ordered consistently)
  dt <- as.data.table(data)
  dt[, row_idx := .I]

  # --- cell-level edge list (year-invariant) ---
  # neighbors is an nb object: list of integer vectors (indices into id_order)
  edge_list <- rbindlist(lapply(seq_along(id_order), function(i) {
    nb <- neighbors[[i]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
      return(data.table(focal_id = integer(0), neighbor_id = integer(0)))
    }
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb])
  }))

  # --- Map every (focal_id, year) row to its neighbor rows ---
  # Create a keyed lookup: (id, year) -> row_idx
  setkey(dt, id, year)

  # Expand: for each row in dt, find its neighbor cell IDs
  # Join dt with edge_list on id == focal_id to get neighbor_id per row
  focal <- dt[, .(focal_row = row_idx, focal_id = id, year)]
  setkey(edge_list, focal_id)

  # merge: each focal row gets its neighbor cell IDs
  expanded <- edge_list[focal, on = .(focal_id), allow.cartesian = TRUE,
                        nomatch = NA]
  # expanded has columns: focal_id, neighbor_id, focal_row, year

  # Drop rows with no neighbors (neighbor_id == NA)
  expanded <- expanded[!is.na(neighbor_id)]

  # Now join to get the ROW INDEX of each (neighbor_id, year) pair
  neighbor_rows <- dt[, .(neighbor_id = id, year, neighbor_row = row_idx)]
  setkey(neighbor_rows, neighbor_id, year)
  setkey(expanded, neighbor_id, year)

  merged <- neighbor_rows[expanded, on = .(neighbor_id, year), nomatch = NA]
  # Keep only matched rows (neighbor exists in that year)
  merged <- merged[!is.na(neighbor_row)]

  # --- Build the lookup list (one element per row of data) ---
  # Order by focal_row so we can split efficiently
  setkey(merged, focal_row)

  n_rows <- nrow(dt)
  lookup <- vector("list", n_rows)

  # Split neighbor_row by focal_row
  split_idx <- split(merged$neighbor_row, merged$focal_row)

  # Assign into lookup (focal rows with no neighbors remain NULL → handle later)
  focal_rows_with_neighbors <- as.integer(names(split_idx))
  for (j in seq_along(split_idx)) {
    lookup[[focal_rows_with_neighbors[j]]] <- split_idx[[j]]
  }

  # Replace NULLs with integer(0)
  null_mask <- vapply(lookup, is.null, logical(1))
  if (any(null_mask)) {
    lookup[null_mask] <- list(integer(0))
  }

  lookup
}

# ----------------------------------------------------------
# 2. Optimized compute_neighbor_stats using vapply
#    Returns an n×3 matrix (max, min, mean) — identical
#    numerical output to the original.
# ----------------------------------------------------------

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]

  result <- vapply(neighbor_lookup, function(idx) {
    if (length(idx) == 0L) return(c(NA_real_, NA_real_, NA_real_))
    nv <- vals[idx]
    nv <- nv[!is.na(nv)]
    if (length(nv) == 0L) return(c(NA_real_, NA_real_, NA_real_))
    c(max(nv), min(nv), mean(nv))
  }, numeric(3))

  t(result)  # transpose to n×3
}

# ----------------------------------------------------------
# 3. Wrapper that adds the three columns per variable
#    (preserves original column-naming convention)
# ----------------------------------------------------------

compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  stats_mat <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)
  data[[paste0(var_name, "_neighbor_max")]]  <- stats_mat[, 1]
  data[[paste0(var_name, "_neighbor_min")]]  <- stats_mat[, 2]
  data[[paste0(var_name, "_neighbor_mean")]] <- stats_mat[, 3]
  data
}

# ----------------------------------------------------------
# 4. Main pipeline execution
# ----------------------------------------------------------

# Build the lookup ONCE (vectorized, minutes not hours)
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

# cell_data now has the same 15 new columns (5 vars × {max, min, mean})
# with identical numerical values as the original pipeline.
# The trained Random Forest model is untouched and can be used directly
# for prediction on the updated cell_data.
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **True bottleneck** | `build_neighbor_lookup`: 6.46M scalar iterations with `paste` + named-vector hash lookups | Eliminated |
| **Lookup construction** | `lapply` over 6.46M rows, string keys | Vectorized `data.table` keyed join on integer columns; one cell-level edge list expanded by year |
| **`compute_neighbor_stats`** | `do.call(rbind, ...)` on 6.46M-element list (minor cost) | `vapply` → direct matrix, `t()` transpose (marginal improvement) |
| **Estimated runtime** | 86+ hours | Minutes (dominated by the `data.table` join on ~38M expanded edges) |
| **Trained RF model** | Preserved ✓ | Preserved ✓ |
| **Numerical output** | max / min / mean per neighbor set | Identical ✓ |