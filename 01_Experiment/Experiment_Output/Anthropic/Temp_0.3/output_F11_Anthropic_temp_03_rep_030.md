 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` — as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to construct neighbor keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. Named-vector lookups in R are **O(n) hash probes per call but with significant per-call overhead** when done millions of times in an interpreted loop. With ~6.46M rows and an average of ~4 rook neighbors per cell, this creates tens of millions of string operations and named-vector lookups.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46M rows via `lapply`, subsetting a numeric vector, removing NAs, and computing `max`, `min`, `mean`. The final `do.call(rbind, result)` on a 6.46M-element list of 3-element vectors is itself expensive.

3. **Random Forest prediction**, by contrast, is a single call to `predict()` on a pre-loaded model object. Even with 110 predictors and 6.46M rows, modern RF implementations (e.g., `ranger`) perform prediction in minutes, not hours. Loading a serialized model from disk is a one-time I/O cost measured in seconds.

**The 86+ hour runtime is dominated by the R-level interpreted loops over 6.46M rows with expensive string operations, repeated 1 + 5 = 6 times (once for the lookup build, five times for the stats computation).**

---

## Optimization Strategy

The key insight is to **vectorize everything** — eliminate the per-row `lapply` loops entirely and replace them with bulk `data.table` merge/join operations and grouped aggregations.

**Specific steps:**

1. **Replace `build_neighbor_lookup()`** with a flat `data.table` edge list that maps each `(id, year)` row index to its neighbor `(neighbor_id, year)` row index via a keyed join — no per-row loop, no string pasting for lookup.

2. **Replace `compute_neighbor_stats()`** with a single vectorized `data.table` grouped aggregation: join the edge table to the data, then compute `max`, `min`, `mean` grouped by the origin row index.

3. **Process all 5 variables in one pass** over the joined edge table rather than rebuilding intermediate structures 5 times.

This reduces the complexity from ~6.46M × 5 interpreted R iterations to a handful of vectorized C-level `data.table` operations.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 0.  Assume these objects already exist in the workspace:
#       cell_data              — data.frame/data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
#       id_order               — integer vector of cell IDs in the order used by the nb object
#       rook_neighbors_unique  — spdep::nb object (list of integer index vectors)
#       rf_model               — the pre-trained Random Forest model (untouched)
# ──────────────────────────────────────────────────────────────────────

# Convert to data.table if not already (non-destructive copy)
if (!is.data.table(cell_data)) cell_data <- as.data.table(cell_data)

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build a flat edge list from the nb object (vectorized)
#
#   Each entry rook_neighbors_unique[[i]] contains the neighbor indices
#   (into id_order) for the i-th cell in id_order.
# ──────────────────────────────────────────────────────────────────────

message("Building flat edge list from nb object...")

# Number of neighbors per cell in the nb object
n_neighbors <- lengths(rook_neighbors_unique)

# Origin cell IDs and neighbor cell IDs (vectorized unlisting)
edge_dt <- data.table(
  id          = rep(id_order, times = n_neighbors),
  neighbor_id = id_order[unlist(rook_neighbors_unique, use.names = FALSE)]
)

# ──────────────────────────────────────────────────────────────────────
# STEP 2: Expand edges across all years (vectorized cross-join)
#
#   Every (id -> neighbor_id) relationship exists for every year.
#   Instead of a full cross-join (which would be huge), we merge
#   through the data itself.
# ──────────────────────────────────────────────────────────────────────

message("Assigning row indices and building lookup join...")

# Add a row index to cell_data
cell_data[, row_idx := .I]

# Create a keyed lookup: (id, year) -> row_idx
# This replaces the string-pasted idx_lookup named vector
key_dt <- cell_data[, .(id, year, row_idx)]
setkey(key_dt, id, year)

# For the neighbor side, we need: (neighbor_id, year) -> neighbor_row_idx
neighbor_key_dt <- copy(key_dt)
setnames(neighbor_key_dt, c("id", "year", "row_idx"),
                          c("neighbor_id", "year", "neighbor_row_idx"))
setkey(neighbor_key_dt, neighbor_id, year)

# ──────────────────────────────────────────────────────────────────────
# STEP 3: Build the full origin-row to neighbor-row mapping
#
#   Join edge_dt to key_dt to get origin row indices,
#   then join to neighbor_key_dt to get neighbor row indices.
#   This replaces build_neighbor_lookup() entirely.
# ──────────────────────────────────────────────────────────────────────

message("Joining edges to row indices (replaces build_neighbor_lookup)...")

# Join edges with all years from the origin cell
# For each (id, neighbor_id) pair, we need all years that 'id' appears in.
# Merge edge_dt with key_dt on 'id' to get (id, year, row_idx, neighbor_id)
setkey(edge_dt, id)
setkey(key_dt, id)

edge_year_dt <- key_dt[edge_dt, on = "id", allow.cartesian = TRUE,
                       nomatch = NULL]
# Columns: id, year, row_idx, neighbor_id

# Now join to get the neighbor's row index for the same year
setkey(edge_year_dt, neighbor_id, year)
edge_full <- neighbor_key_dt[edge_year_dt, on = c("neighbor_id", "year"),
                             nomatch = NA]
# Columns: neighbor_id, year, neighbor_row_idx, id, row_idx

# Drop rows where the neighbor doesn't exist in that year
edge_full <- edge_full[!is.na(neighbor_row_idx)]

# Keep only what we need
edge_full <- edge_full[, .(row_idx, neighbor_row_idx)]
setkey(edge_full, row_idx)

message(sprintf("Edge table: %s origin-neighbor-year pairs", format(nrow(edge_full), big.mark = ",")))

# Free intermediate objects
rm(edge_dt, edge_year_dt, key_dt, neighbor_key_dt)
gc()

# ──────────────────────────────────────────────────────────────────────
# STEP 4: Compute neighbor stats for all 5 variables (vectorized)
#
#   This replaces compute_neighbor_stats() and the outer for-loop.
#   Instead of 5 separate lapply passes over 6.46M rows, we do
#   one vectorized extraction + one grouped aggregation.
# ──────────────────────────────────────────────────────────────────────

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

message("Computing neighbor statistics for all variables (vectorized)...")

# Extract neighbor values for all variables at once using the neighbor row indices
# This is a single vectorized column-subset operation
neighbor_vals_dt <- cell_data[edge_full$neighbor_row_idx,
                              ..neighbor_source_vars]
neighbor_vals_dt[, row_idx := edge_full$row_idx]

# Grouped aggregation: max, min, mean per row_idx, per variable
# We melt to long form, aggregate, then cast back to wide
# But for performance, it's faster to aggregate each variable directly.

for (var in neighbor_source_vars) {
  message(sprintf("  Aggregating neighbor stats for: %s", var))
  
  # Build a small DT with just the origin row index and the neighbor's value
  agg_dt <- data.table(
    row_idx = edge_full$row_idx,
    val     = neighbor_vals_dt[[var]]
  )
  
  # Remove NA neighbor values before aggregation
  agg_dt <- agg_dt[!is.na(val)]
  
  # Grouped aggregation (C-level, vectorized)
  stats <- agg_dt[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), keyby = row_idx]
  
  # Assign back to cell_data by row index
  max_col  <- paste0("nb_max_", var)
  min_col  <- paste0("nb_min_", var)
  mean_col <- paste0("nb_mean_", var)
  
  # Initialize with NA
  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]
  
  # Fill in computed values
  cell_data[stats$row_idx, (max_col)  := stats$nb_max]
  cell_data[stats$row_idx, (min_col)  := stats$nb_min]
  cell_data[stats$row_idx, (mean_col) := stats$nb_mean]
  
  rm(agg_dt, stats)
}

rm(neighbor_vals_dt, edge_full)
gc()

message("Neighbor feature engineering complete.")

# ──────────────────────────────────────────────────────────────────────
# STEP 5: Random Forest prediction (unchanged — not the bottleneck)
# ──────────────────────────────────────────────────────────────────────

message("Running Random Forest prediction...")

# Remove the helper column before prediction
cell_data[, row_idx := NULL]

# Predict using the pre-trained model (preserved exactly as-is)
predictions <- predict(rf_model, data = cell_data)

# If using ranger:
# cell_data[, prediction := predictions$predictions]
# If using randomForest:
# cell_data[, prediction := as.numeric(predictions)]

message("Pipeline complete.")
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **`build_neighbor_lookup`** | `lapply` over 6.46M rows with `paste()` + named-vector lookups | Vectorized `data.table` keyed joins — no per-row loop |
| **`compute_neighbor_stats`** | 5 × `lapply` over 6.46M rows + `do.call(rbind, ...)` on 6.46M-element list | 5 × `data.table` grouped `[, .(max, min, mean), by = row_idx]` — C-level |
| **String operations** | ~32M+ `paste()` and named-vector character lookups | Zero — integer key joins only |
| **RF inference** | Single `predict()` call (minutes) | Unchanged — it was never the bottleneck |
| **Expected runtime** | 86+ hours | Minutes to low tens of minutes on the same laptop |
| **Trained RF model** | Preserved | Preserved (no retraining) |
| **Numerical estimand** | Preserved | Preserved (identical max/min/mean logic, same neighbor definitions) |

**Verdict:** The bottleneck is the interpreted-loop spatial feature engineering, not Random Forest inference. The fix is vectorization via `data.table` joins and grouped aggregations.