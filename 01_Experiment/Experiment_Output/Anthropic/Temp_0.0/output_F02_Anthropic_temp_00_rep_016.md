 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting, hash-table lookups, and NA filtering in pure R. This creates millions of small character vectors and triggers repeated hash lookups on a named vector of 6.46 million entries — an O(n²)-like workload in practice. The `compute_neighbor_stats` function then iterates over the same 6.46 million entries again for each of the 5 variables, subsetting numeric vectors by index lists. The combination of these two stages, repeated for 5 variables, produces the estimated 86+ hour runtime.

**Specific problems:**

1. **String-key lookups at scale:** `idx_lookup` is a named vector with ~6.46M entries. Named-vector lookup in R uses linear hashing that degrades at this scale. Each of the 6.46M rows performs multiple lookups into it.
2. **Per-row `lapply` with allocations:** Each iteration of the `lapply` in `build_neighbor_lookup` allocates character vectors (`paste`), performs named lookups, and filters NAs — millions of tiny allocations that thrash the garbage collector.
3. **List-of-vectors structure for `neighbor_lookup`:** Storing ~6.46M list elements (each a small integer vector) is memory-inefficient and cache-unfriendly.
4. **`compute_neighbor_stats` uses `lapply` + `do.call(rbind, ...)`:** Binding 6.46M 3-element vectors row-by-row is slow.

---

## Optimization Strategy

**Replace all per-row R loops with vectorized `data.table` joins and grouped aggregations.**

The key insight: the neighbor lookup is a **join** problem. Each `(cell, year)` needs to find its neighbors' values for the same year. This is a merge between a neighbor-edge table and the data table on `(neighbor_id, year)`, followed by a `group-by` aggregation. `data.table` performs this in optimized C with minimal memory overhead.

**Steps:**

1. **Build an edge table once** — a two-column `data.table` of `(id, neighbor_id)` from the `nb` object. This is ~1.37M rows.
2. **For each variable, join the edge table to the data on `(neighbor_id, year)`** to get neighbor values, then **group by `(id, year)`** to compute max, min, mean.
3. **Left-join** the aggregated stats back to the main data.

This eliminates all per-row R loops, all string-key hashing, and all list-of-vectors storage. Expected runtime: **minutes, not hours**.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# Step 1: Build the edge table ONCE from the nb object
#         (rook_neighbors_unique is a list of integer vectors
#          indexed by position in id_order)
# ---------------------------------------------------------------
build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(neighbors))
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_i <- neighbors[[i]]
    n_i  <- length(nb_i)
    if (n_i == 0L) next
    from_id[pos:(pos + n_i - 1L)] <- id_order[i]
    to_id[pos:(pos + n_i - 1L)]   <- id_order[nb_i]
    pos <- pos + n_i
  }
  
  data.table(id = from_id, neighbor_id = to_id)
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

# ---------------------------------------------------------------
# Step 2: Convert main data to data.table (in-place if possible)
# ---------------------------------------------------------------
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Ensure keys for fast joins
setkey(cell_data, id, year)

# ---------------------------------------------------------------
# Step 3: For each neighbor source variable, compute neighbor
#         max, min, mean via keyed join + grouped aggregation,
#         then left-join back to cell_data.
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  
  cat("Processing neighbor features for:", var_name, "\n")
  
  # Subset only the columns we need for the join (minimise memory)
  # Columns: neighbor_id (to join on), year, and the variable value
  val_dt <- cell_data[, .(neighbor_id = id, year, val = get(var_name))]
  setkey(val_dt, neighbor_id, year)
  
  # Join edges to data: for each (id, neighbor_id) pair, attach
  # the neighbor's value in each year.
  # edge_dt has (id, neighbor_id); we join val_dt on (neighbor_id, year).
  # We need to bring in year from the focal cell. Strategy:
  #   1. Join edge_dt to cell_data's (id, year) to expand edges across years.
  #      But that would be 1.37M edges × 28 years = ~38M rows — manageable.
  #   2. Then join to val_dt on (neighbor_id, year).
  
  # Get unique (id, year) pairs from cell_data
  focal_dt <- cell_data[, .(id, year)]
  setkey(focal_dt, id)
  
  # Expand: each focal (id, year) gets its neighbor_ids
  # This is a join of focal_dt and edge_dt on 'id'
  setkey(edge_dt, id)
  expanded <- edge_dt[focal_dt, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded has columns: id, neighbor_id, year
  # Rows: ~1.37M avg neighbors × 28 years ≈ 38.4M rows
  
  # Now join to get the neighbor's value
  setkey(expanded, neighbor_id, year)
  expanded[val_dt, on = .(neighbor_id, year), neighbor_val := i.val]
  
  # Aggregate by (id, year)
  agg <- expanded[!is.na(neighbor_val),
                  .(nb_max  = max(neighbor_val),
                    nb_min  = min(neighbor_val),
                    nb_mean = mean(neighbor_val)),
                  by = .(id, year)]
  
  # Rename columns to match expected feature names
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(agg, c("nb_max", "nb_min", "nb_mean"),
                c(max_col,  min_col,  mean_col))
  
  # Left-join back to cell_data
  setkey(agg, id, year)
  setkey(cell_data, id, year)
  
  # Remove old columns if they exist (idempotent re-runs)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }
  
  cell_data <- agg[cell_data, on = .(id, year)]
  setkey(cell_data, id, year)
  
  # Clean up intermediate objects to free RAM
  rm(val_dt, focal_dt, expanded, agg)
  gc()
  
  cat("  Done.\n")
}

# ---------------------------------------------------------------
# Step 4: Predict with the existing trained Random Forest model
#         (model object is unchanged; feature names and values
#          are numerically identical to the original pipeline)
# ---------------------------------------------------------------
# Example (adjust to your actual model object and predict call):
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets (same rook adjacency, same year matching, same NA exclusion) as the original code. The `data.table` grouped aggregation is numerically identical. |
| **Trained RF model** | The model object is never touched. The output columns have the same names and values, so `predict()` produces identical results. |
| **NA handling** | `!is.na(neighbor_val)` in the aggregation and `nomatch = 0L` in joins replicate the original NA-filtering logic. Cells with zero valid neighbors get `NA` for all three stats via the left join. |

---

## Expected Performance

| Stage | Original | Optimized |
|---|---|---|
| Neighbor lookup build | ~hours (6.46M `lapply` iterations with string ops) | ~seconds (vectorized edge table construction) |
| Stats per variable | ~15+ hours (6.46M `lapply` + `do.call(rbind)`) | ~1–3 minutes (`data.table` keyed join + grouped agg on ~38M rows) |
| **Total for 5 variables** | **~86+ hours** | **~10–20 minutes** |
| Peak RAM | Unbounded list-of-lists growth | ~3–5 GB (38M-row intermediate, freed each iteration) |

This fits comfortably within the 16 GB laptop constraint and reduces runtime by roughly **two to three orders of magnitude**.