 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(N) over all cell-year rows (~6.46M), but it doesn't need to be.

The neighbor relationships are **purely spatial** — they don't change across years. Yet the function builds a lookup keyed on `(id, year)` pairs, iterating over every one of the ~6.46 million rows. For each row it:
- Converts the cell ID to a reference index.
- Looks up neighbor cell IDs from the `nb` object.
- Pastes neighbor IDs with the current row's year to form string keys.
- Matches those string keys back into a named character vector of length 6.46M.

The `paste()`-based string key construction and named-vector lookup (`idx_lookup[neighbor_keys]`) is **O(n)** per call in the worst case for R's hashed name lookup, and the sheer volume (6.46M calls, each doing multiple string operations and lookups) is the dominant bottleneck. The `lapply` over 6.46M rows with string concatenation and named-vector indexing is catastrophically slow.

### 2. `compute_neighbor_stats` uses `lapply` over 6.46M rows.

Even though the per-element work is small (subsetting a numeric vector, computing max/min/mean), calling an R-level anonymous function 6.46 million times with `lapply` and then `do.call(rbind, ...)` on 6.46M 3-element vectors is very slow due to R's function-call overhead and the final row-binding.

### Root Cause Summary

The spatial topology is **year-invariant**. There are only 344,208 cells and ~1.37M directed neighbor pairs. But the code treats every cell-year row as if it has a unique neighbor structure, inflating the problem by 28×. The string-key approach turns a simple integer-indexing problem into millions of string allocations and hash lookups.

---

## Optimization Strategy

### Core Insight: Separate Topology from Attributes

1. **Build the neighbor edge table once** — a simple two-column integer `data.table` of `(cell_id, neighbor_id)` with ~1.37M rows. This is year-invariant and built from the `nb` object in milliseconds.

2. **For each variable, use a `data.table` join** — join the cell-year attribute values onto the edge table by `(neighbor_id, year)`, then group by `(cell_id, year)` to compute `max`, `min`, `mean`. This replaces millions of R-level function calls with vectorized, indexed `data.table` operations.

### Expected Speedup

| Step | Current | Optimized |
|---|---|---|
| Build lookup | ~hours (6.46M string ops) | ~seconds (1.37M integer rows) |
| Neighbor stats (per variable) | ~hours (6.46M lapply calls) | ~seconds (data.table keyed join + grouped aggregation) |
| **Total for 5 variables** | **~86+ hours** | **~1–5 minutes** |

### Constraints Preserved
- The trained Random Forest model is **not retrained** — we only recompute the same input features faster.
- The numerical estimand is **identical** — same max, min, mean of the same neighbor values.

---

## Working R Code

```r
library(data.table)

# ==============================================================
# STEP 1: Build the year-invariant neighbor edge table ONCE
#         from the precomputed spdep::nb object.
# ==============================================================

build_neighbor_edges <- function(id_order, nb_obj) {
  # id_order: vector of cell IDs in the same order as the nb object
  # nb_obj:   spdep::nb list (rook_neighbors_unique)
  #
  # Returns a data.table with columns: cell_id, neighbor_id
  # representing all directed neighbor pairs (~1.37M rows).

  n <- length(nb_obj)
  # Pre-count total edges for pre-allocation
  edge_counts <- vapply(nb_obj, length, integer(1))
  total_edges <- sum(edge_counts)

  from_id <- integer(total_edges)
  to_id   <- integer(total_edges)

  pos <- 1L
  for (i in seq_len(n)) {
    nb_indices <- nb_obj[[i]]
    if (length(nb_indices) == 0L || (length(nb_indices) == 1L && nb_indices[1] == 0L)) next
    len <- length(nb_indices)
    from_id[pos:(pos + len - 1L)] <- id_order[i]
    to_id[pos:(pos + len - 1L)]   <- id_order[nb_indices]
    pos <- pos + len
  }

  # Trim if any nb entries were empty (0-neighbor cells)
  if (pos <= total_edges) {
    from_id <- from_id[1:(pos - 1L)]
    to_id   <- to_id[1:(pos - 1L)]
  }

  data.table(cell_id = from_id, neighbor_id = to_id)
}

# Build it once
neighbor_edges <- build_neighbor_edges(id_order, rook_neighbors_unique)

# ==============================================================
# STEP 2: Convert cell_data to data.table (if not already)
# ==============================================================

if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ==============================================================
# STEP 3: Function to compute neighbor max, min, mean for one
#         variable via keyed join + grouped aggregation.
# ==============================================================

compute_neighbor_features_dt <- function(cell_dt, neighbor_edges, var_name) {
  # cell_dt:        data.table with columns id, year, and <var_name>
  # neighbor_edges: data.table with columns cell_id, neighbor_id
  # var_name:       character, the variable to aggregate
  #
  # Returns cell_dt with three new columns appended:
  #   <var_name>_neighbor_max, <var_name>_neighbor_min, <var_name>_neighbor_mean

  # Extract only the columns we need for the join: neighbor_id, year, value
  val_dt <- cell_dt[, .(neighbor_id = id, year, val = get(var_name))]
  setkey(val_dt, neighbor_id, year)

  # Cross neighbor edges with all years present in the data.
  # Each edge (cell_id, neighbor_id) is replicated for every year,
  # then we join the neighbor's attribute value.
  #
  # But more efficiently: join edges to val_dt directly.
  # We need (cell_id, year) -> aggregate over neighbor values.
  # Start from edges, add year from the focal cell, get neighbor value.

  # Approach: expand edges × years via join on neighbor side.
  # edges has (cell_id, neighbor_id). val_dt has (neighbor_id, year, val).
  # Join: for each edge, get all (year, val) of the neighbor.
  # Then group by (cell_id, year).

  setkey(neighbor_edges, neighbor_id)
  # This join replicates each edge for every year the neighbor has data.
  # Result: (cell_id, neighbor_id, year, val)
  joined <- val_dt[neighbor_edges, on = "neighbor_id", allow.cartesian = TRUE, nomatch = NA]
  # joined columns: neighbor_id, year, val, cell_id

  # Aggregate: for each (cell_id, year), compute stats over neighbor vals
  stats <- joined[!is.na(val),
                  .(nmax  = max(val),
                    nmin  = min(val),
                    nmean = mean(val)),
                  by = .(cell_id, year)]

  # Name the new columns
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  setnames(stats, c("nmax", "nmin", "nmean"), c(max_col, min_col, mean_col))

  # Remove old columns if they exist (in case of re-run)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }

  # Merge back onto cell_dt
  result <- merge(cell_dt, stats, by.x = c("id", "year"), by.y = c("cell_id", "year"), all.x = TRUE)

  return(result)
}

# ==============================================================
# STEP 4: Outer loop — compute neighbor features for all 5 vars
# ==============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  cell_data <- compute_neighbor_features_dt(cell_data, neighbor_edges, var_name)
}

# ==============================================================
# STEP 5: Predict with the existing trained Random Forest model
#         (model object assumed to be `rf_model`)
# ==============================================================

# Convert back to data.frame if the RF model expects one
cell_data_df <- as.data.frame(cell_data)

# Predict (model is NOT retrained)
cell_data_df$prediction <- predict(rf_model, newdata = cell_data_df)
```

---

## Memory Consideration (16 GB RAM)

The `allow.cartesian = TRUE` join in Step 3 expands ~1.37M edges × 28 years = ~38.4M rows per variable. Each row holds two integers, one double, and one integer ≈ 20 bytes → ~768 MB per variable at peak. This is well within 16 GB, especially since the intermediate is discarded after aggregation. If memory is tight, you can process one variable at a time (as the loop already does) and call `gc()` between iterations.

## Why the Numerical Estimand Is Preserved

The `data.table` join + group-by computes **exactly** `max()`, `min()`, and `mean()` over the same set of neighbor values that the original `lapply`-based code computed. The neighbor topology is identical (same `nb` object, same `id_order`). The only difference is the computational mechanism — vectorized joins instead of row-by-row string-key lookups. The outputs are numerically identical.