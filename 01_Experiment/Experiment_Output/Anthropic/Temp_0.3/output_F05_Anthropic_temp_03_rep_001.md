 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Inside the Loop

`build_neighbor_lookup` iterates over all ~6.46 million rows. For **each row**, it:

1. Converts the cell ID to a character and looks it up in `id_to_ref` — **6.46M character coercions + named-vector lookups**.
2. Extracts neighbor cell IDs from the `nb` object.
3. **Pastes** each neighbor ID with the current year (`paste(..., sep="_")`) — this creates ~1.37M × 28 ≈ **tens of millions of temporary strings**.
4. Looks up each string key in `idx_lookup` (a named vector of length 6.46M) — **named-vector lookup is O(N) per probe** in base R because it uses linear hashing on a massive names attribute.

The string-key named vector `idx_lookup` has 6.46 million entries. Each lookup into it is expensive. Across all rows and their neighbors, you're doing roughly **38 million string-key lookups into a 6.46M-entry named vector**. This is the dominant cost.

Then `compute_neighbor_stats` is called 5 times (once per variable), but since it reuses the integer `neighbor_lookup`, it's comparatively cheap. The bottleneck is `build_neighbor_lookup`.

### Why This Is a Broader Algorithmic Issue

The fundamental insight is: **the neighbor topology is year-invariant**. Cell *i*'s rook neighbors are the same cells every year. The only thing that changes across years is *which row* in the data corresponds to a given (cell, year) pair. So the entire string-keying approach is an unnecessary indirection. You can:

1. Build a **cell-to-rows** mapping once (integer-indexed).
2. Build the **neighbor graph** once at the cell level (not the row level).
3. For each row, find its neighbors' rows by a direct integer join — no strings at all.

Furthermore, the per-row `lapply` can be **fully vectorized** using `data.table` grouped operations or a sparse-matrix multiplication, eliminating the R-level loop entirely.

---

## Optimization Strategy

| Step | Current | Proposed |
|---|---|---|
| Neighbor topology | Rebuilt per-row via string keys | Built once at cell level, reused |
| Row lookup | Named character vector (6.46M entries) | Integer-indexed `data.table` join |
| Per-row iteration | `lapply` over 6.46M rows | Vectorized edge-list join + `data.table` grouped aggregation |
| Stat computation | 5 separate `lapply` passes | Single grouped aggregation over all 5 variables |
| Complexity | ~38M string lookups into 6.46M named vec | ~38M integer-indexed joins (vectorized) |

**Expected speedup**: from ~86+ hours to **minutes** (likely 2–10 minutes depending on RAM pressure).

---

## Working R Code

```r
library(data.table)

#
# ── 0. Assumptions ──────────────────────────────────────────────────────────
#
# cell_data        : data.frame/data.table with columns id, year, ntl, ec,
#                    pop_density, def, usd_est_n2, plus other columns.
# id_order         : integer/numeric vector of cell IDs in the order matching
#                    rook_neighbors_unique (i.e., id_order[k] is the cell ID
#                    for the k-th element of the nb object).
# rook_neighbors_unique : an nb object (list of integer vectors of neighbor
#                         indices, referencing positions in id_order).
# rf_model         : the already-trained Random Forest model (untouched).
#

#
# ── 1. Build a vectorized edge list from the nb object ──────────────────────
#
# This replaces the per-row string-key lookup entirely.
# Each element rook_neighbors_unique[[k]] contains the *positional indices*
# (into id_order) of the neighbors of cell id_order[k].
#

build_edge_list <- function(id_order, nb_obj) {
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(nb_obj))
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (k in seq_along(nb_obj)) {
    nbrs <- nb_obj[[k]]
    # spdep nb objects use 0L to denote "no neighbors"
    nbrs <- nbrs[nbrs != 0L]
    n <- length(nbrs)
    if (n > 0L) {
      from_id[pos:(pos + n - 1L)] <- id_order[k]
      to_id[pos:(pos + n - 1L)]   <- id_order[nbrs]
      pos <- pos + n
    }
  }
  
  # Trim if any 0-neighbor cells caused over-allocation
  if (pos <= n_edges) {
    from_id <- from_id[1:(pos - 1L)]
    to_id   <- to_id[1:(pos - 1L)]
  }
  
  data.table(from_id = from_id, to_id = to_id)
}

cat("Building edge list from nb object...\n")
edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
cat(sprintf("  Edge list: %d directed edges\n", nrow(edge_dt)))

#
# ── 2. Convert cell_data to data.table and add a row index ──────────────────
#
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Preserve original row order for downstream compatibility
cell_data[, .row_idx := .I]

#
# ── 3. Vectorized neighbor-stat computation ─────────────────────────────────
#
# Strategy:
#   For each row (id_i, year_t), its neighbors are the set of rows
#   (id_j, year_t) where (id_i -> id_j) is in the edge list.
#
#   We achieve this by joining:
#     cell_data[, .(id, year, var1, ..., var5)]
#       ⟶ edge_dt on id == from_id
#       ⟶ cell_data on to_id == id AND same year
#   Then group by the focal row and compute max/min/mean.
#

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_stats <- function(cell_data, edge_dt, source_vars) {
  
  # Subset to only the columns we need for the join
  # focal side: we need id, year, .row_idx
  # neighbor side: we need id, year, and the source variables
  
  focal_cols    <- c("id", "year", ".row_idx")
  neighbor_cols <- c("id", "year", source_vars)
  
  focal_dt    <- cell_data[, ..focal_cols]
  neighbor_dt <- cell_data[, ..neighbor_cols]
  
  # Step A: Join focal rows to edge list to get (focal_row, neighbor_cell_id, year)
  # focal_dt joins edge_dt on focal_dt$id == edge_dt$from_id
  setkey(edge_dt, from_id)
  setkey(focal_dt, id)
  
  cat("  Join focal rows to edge list...\n")
  # Each focal row fans out to its neighbors
  # Result: one row per (focal_row, neighbor_cell, year)
  joined <- edge_dt[focal_dt,
                    .(focal_row = .row_idx, to_id, year),
                    on = .(from_id = id),
                    allow.cartesian = TRUE,
                    nomatch = NULL]
  
  cat(sprintf("  Joined table: %d rows (focal × neighbors × years)\n", nrow(joined)))
  
  # Step B: Join to neighbor_dt to get the neighbor variable values
  # Match on to_id == neighbor_dt$id AND same year
  setkey(neighbor_dt, id, year)
  setkey(joined, to_id, year)
  
  cat("  Join to neighbor values...\n")
  joined2 <- neighbor_dt[joined,
                         on = .(id = to_id, year),
                         nomatch = NULL]
  
  # joined2 now has columns: id (neighbor), year, source_vars..., focal_row
  # We need to group by focal_row and compute stats for each source var.
  
  cat("  Computing grouped statistics...\n")
  
  # Build the aggregation expressions dynamically
  agg_exprs <- list()
  for (v in source_vars) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("nb_max_", v)]]  <- bquote(
      as.numeric(ifelse(all(is.na(.(v_sym))), NA_real_, max(.(v_sym), na.rm = TRUE)))
    )
    agg_exprs[[paste0("nb_min_", v)]]  <- bquote(
      as.numeric(ifelse(all(is.na(.(v_sym))), NA_real_, min(.(v_sym), na.rm = TRUE)))
    )
    agg_exprs[[paste0("nb_mean_", v)]] <- bquote(
      as.numeric(ifelse(all(is.na(.(v_sym))), NA_real_, mean(.(v_sym), na.rm = TRUE)))
    )
  }
  
  agg_call <- as.call(c(as.name("list"), agg_exprs))
  
  stats_dt <- joined2[, eval(agg_call), by = .(focal_row)]
  
  cat(sprintf("  Stats computed for %d focal rows with neighbors\n", nrow(stats_dt)))
  
  return(stats_dt)
}

cat("Computing all neighbor statistics (vectorized)...\n")
t0 <- proc.time()

stats_dt <- compute_all_neighbor_stats(cell_data, edge_dt, neighbor_source_vars)

# ── 4. Merge stats back into cell_data ──────────────────────────────────────
# Rows with no neighbors (e.g., boundary cells with no valid neighbor in data)
# will get NA, which matches the original behavior.

stat_cols <- setdiff(names(stats_dt), "focal_row")

# Initialize new columns to NA
for (col in stat_cols) {
  set(cell_data, j = col, value = NA_real_)
}

# Assign by reference using focal_row as the row index
cell_data[stats_dt$focal_row, (stat_cols) := stats_dt[, ..stat_cols]]

elapsed <- (proc.time() - t0)["elapsed"]
cat(sprintf("Done. Elapsed: %.1f seconds\n", elapsed))

#
# ── 5. Clean up helper column ───────────────────────────────────────────────
#
cell_data[, .row_idx := NULL]

#
# ── 6. Verify column naming matches original pipeline ───────────────────────
#
# The original code (via compute_and_add_neighbor_features) likely produced
# columns named like: nb_max_ntl, nb_min_ntl, nb_mean_ntl, etc.
# Adjust the naming below if your original pipeline used different names.
# The names above already follow this convention.
#
cat("New neighbor-stat columns added:\n")
print(stat_cols)

#
# ── 7. The trained RF model is untouched ────────────────────────────────────
#
# rf_model remains as-is. Use it for prediction on the updated cell_data:
# predictions <- predict(rf_model, newdata = cell_data)
#
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same numerical estimand** | The vectorized joins match exactly the same (cell_i, cell_j, year_t) triples as the original. `max`, `min`, `mean` with `na.rm=TRUE` and the all-NA → NA fallback replicate the original `compute_neighbor_stats` logic exactly. |
| **RF model untouched** | No retraining. The code only constructs the same feature columns the model expects. |
| **Column names** | Follow the `nb_{max,min,mean}_{varname}` convention. Adjust the prefix if your `compute_and_add_neighbor_features` used different names. |
| **NA handling** | Cells with zero valid neighbors get `NA` for all three stats, identical to the original `return(c(NA, NA, NA))`. |

---

## Memory Considerations (16 GB laptop)

The largest intermediate object is `joined2`, which has ~38M rows × ~8 columns ≈ **2.3 GB**. Combined with `cell_data` (~6.46M × 110 cols ≈ 5.7 GB), peak usage will be around **10–12 GB**, which fits in 16 GB. If memory is tight, you can process the 5 variables in batches of 2–3 instead of all at once (split `source_vars`, run twice, and `cbind` the results).

**Expected runtime: 2–10 minutes** vs. the original 86+ hours — a roughly **500–2500× speedup**.