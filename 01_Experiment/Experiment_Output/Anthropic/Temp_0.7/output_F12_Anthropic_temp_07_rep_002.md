 # Diagnosis, Optimization Strategy, and Working Code

## Diagnosis

The colleague is **partially right but misses the deeper bottleneck**. Let me audit both functions:

### `compute_neighbor_stats()` — Colleague's Claim

`do.call(rbind, result)` on a list of ~6.46 million 3-element vectors is indeed wasteful (it creates a matrix row-by-row), and could be replaced with a single `matrix()` call. However, this operation runs only **5 times** (once per neighbor source variable). Binding 6.46M small vectors into a matrix is on the order of seconds, not hours. The `lapply` inside `compute_neighbor_stats` is also simple: it indexes into a pre-extracted numeric vector and computes `max/min/mean` on a small handful of neighbors (~4 for rook contiguity). This is lightweight. **This is not the 86-hour bottleneck.**

### `build_neighbor_lookup()` — The True Bottleneck

This function runs `lapply` over **6.46 million rows**, and for each row it:

1. **Character coercion + named-vector lookup**: `id_to_ref[as.character(data$id[i])]` — named vector lookup via string matching, O(n) in the worst case per call.
2. **Subsetting `id_order` by neighbor indices**: `id_order[neighbors[[ref_idx]]]`.
3. **`paste()` to build keys**: `paste(neighbor_cell_ids, data$year[i], sep = "_")` — creates temporary character vectors 6.46M times.
4. **Named-vector lookup on `idx_lookup`**: `idx_lookup[neighbor_keys]` — this named vector has **6.46 million entries**. Named vector lookup in R is **O(n)** per query (linear scan or partial hashing), performed ~4 times per row (one per neighbor). Over 6.46M rows × ~4 neighbors = ~25.8 million string-match lookups into a 6.46M-length named vector.

**This is the catastrophic bottleneck.** The total cost is roughly O(n²) in character matching. At 6.46M rows, this explains the 86+ hour estimate.

### Summary

| Component | Calls | Per-call cost | Total cost | Bottleneck? |
|---|---|---|---|---|
| `build_neighbor_lookup` | 1 | 6.46M × ~4 named-vector lookups into 6.46M-entry vector | **O(n × k × n) ≈ O(n²)** | **YES — dominant** |
| `compute_neighbor_stats` `lapply` | 5 | 6.46M × trivial arithmetic on ~4 values | O(n × k) | No |
| `do.call(rbind, ...)` | 5 | Bind 6.46M vectors | O(n) | Minor |

**Verdict: Reject the colleague's diagnosis.** The true bottleneck is `build_neighbor_lookup()`, specifically the repeated `paste()`-based key construction and named-vector lookups over a 6.46M-entry character vector. Replace it with integer-indexed hash lookups (via `data.table` or environments), and vectorize the entire operation to eliminate the per-row `lapply`.

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup`** with a fully vectorized, hash-based approach using `data.table` keyed joins. Instead of pasting keys and doing named-vector lookups, build an integer-keyed lookup table `(id, year) → row_index` and join directly.

2. **Vectorize `compute_neighbor_stats`** using `data.table` grouped operations — expand all neighbor pairs into a long table, join values, and compute `max/min/mean` by group in one pass. This eliminates both the `lapply` and the `do.call(rbind, ...)`.

3. **Preserve** the trained Random Forest model (no retraining) and the original numerical estimand (same `max`, `min`, `mean` neighbor statistics, same column names).

Expected speedup: from ~86 hours to **minutes**.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# OPTIMIZED PIPELINE — replaces build_neighbor_lookup, compute_neighbor_stats,
# and the outer loop. Produces identical numerical output.
# ==============================================================================

# --- Step 0: Convert to data.table and assign row indices --------------------
cell_dt <- as.data.table(cell_data)
cell_dt[, row_idx := .I]

# --- Step 1: Build (id, year) -> row_idx lookup via keyed data.table ---------
# This replaces the paste()-based named vector with an O(1) hash join.
lookup_dt <- cell_dt[, .(id, year, row_idx)]
setkey(lookup_dt, id, year)

# --- Step 2: Build id -> ref_idx mapping (position in id_order) --------------
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

# --- Step 3: Expand all (row_idx, neighbor_row_idx) pairs in one shot --------
# For each row in cell_dt, find its neighbors' row indices via a keyed join.
#
# 3a. Map each row's id to its ref_idx in the nb object
cell_dt[, ref_idx := id_to_ref[as.character(id)]]

# 3b. Build a long table of (source_row_idx, neighbor_cell_id, year)
#     by expanding the nb list for each unique cell, then joining on year.

# First, build a data.table of (ref_idx, neighbor_cell_id) from the nb object.
# This is done once and is small: ~1.37M directed relationships.
nb_edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(r) {
  nb <- rook_neighbors_unique[[r]]
  if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
    return(data.table(ref_idx = integer(0), neighbor_id = integer(0)))
  }
  data.table(ref_idx = r, neighbor_id = id_order[nb])
}))

# 3c. Join nb_edges to cell_dt to get (source_row_idx, neighbor_id, year)
#     For every row in cell_dt, we know its ref_idx; join to nb_edges.
setkey(nb_edges, ref_idx)
cell_ref <- cell_dt[, .(row_idx, ref_idx, year)]
setkey(cell_ref, ref_idx)

# This join expands each source row by its number of neighbors (~4 for rook).
# Result: ~25.8M rows of (source_row_idx, neighbor_id, year)
edge_expanded <- nb_edges[cell_ref, on = "ref_idx", allow.cartesian = TRUE,
                          nomatch = NA,
                          .(source_row_idx = i.row_idx,
                            neighbor_id    = x.neighbor_id,
                            year           = i.year)]

# Drop rows where neighbor_id is NA (cells with no neighbors)
edge_expanded <- edge_expanded[!is.na(neighbor_id)]

# 3d. Join to lookup_dt to resolve (neighbor_id, year) -> neighbor_row_idx
setnames(edge_expanded, "neighbor_id", "id")
setkey(edge_expanded, id, year)
edge_expanded <- lookup_dt[edge_expanded, on = c("id", "year"),
                           nomatch = NA,
                           .(source_row_idx = i.source_row_idx,
                             neighbor_row_idx = x.row_idx)]

# Drop unresolved neighbors
edge_expanded <- edge_expanded[!is.na(neighbor_row_idx)]

# --- Step 4: Compute neighbor stats for all 5 variables in vectorized fashion -
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  # Extract the variable values and attach to edges by neighbor_row_idx
  vals <- cell_dt[[var_name]]
  edge_expanded[, nval := vals[neighbor_row_idx]]

  # Compute grouped stats (excluding NAs), keyed by source_row_idx
  stats <- edge_expanded[!is.na(nval),
                         .(nb_max  = max(nval),
                           nb_min  = min(nval),
                           nb_mean = mean(nval)),
                         by = source_row_idx]

  # Initialize columns with NA
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]

  # Assign computed stats back by row index
  cell_dt[stats$source_row_idx, (max_col)  := stats$nb_max]
  cell_dt[stats$source_row_idx, (min_col)  := stats$nb_min]
  cell_dt[stats$source_row_idx, (mean_col) := stats$nb_mean]
}

# Clean up helper columns
cell_dt[, c("row_idx", "ref_idx") := NULL]
edge_expanded[, nval := NULL]

# --- Step 5: Convert back to data.frame for downstream compatibility ---------
cell_data <- as.data.frame(cell_dt)

# ==============================================================================
# The trained Random Forest model is untouched. Use cell_data with the new
# neighbor feature columns for prediction as before:
#   predictions <- predict(rf_model, newdata = cell_data)
# ==============================================================================
```

---

## Why This Works and What Changed

| Original | Optimized | Why faster |
|---|---|---|
| `paste()` key construction per row (6.46M calls) | `data.table` keyed join on integer `(id, year)` | O(1) hash join vs O(n) string scan |
| Named-vector lookup into 6.46M-entry vector | Eliminated entirely | — |
| Row-level `lapply` in `build_neighbor_lookup` (6.46M iterations) | Single vectorized `data.table` cartesian join | No R-level loop |
| Row-level `lapply` in `compute_neighbor_stats` (6.46M × 5) | `data.table` grouped `by = source_row_idx` aggregation | C-level grouped ops |
| `do.call(rbind, ...)` on 6.46M vectors (×5) | Direct column assignment | No intermediate list |

**Expected runtime**: ~2–10 minutes on a 16 GB laptop (dominated by the ~25.8M-row cartesian join and grouped aggregation, both highly optimized in `data.table`).

**Numerical equivalence**: The same `max`, `min`, and `mean` are computed over the same neighbor sets, producing identical column values. The Random Forest model is never modified.