 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three interconnected reasons:

### 1. Row-level `lapply` over 6.46 million rows in `build_neighbor_lookup`
For every single cell-year row, the function performs character coercion, `paste`, and named-vector lookups. With ~6.46M rows, this means ~6.46M iterations of string operations and hash lookups. Named vector lookups in R via `[` on character keys are O(n) in the worst case per call due to hashing overhead at scale.

### 2. Redundant recomputation of the same spatial topology per year
The rook-neighbor structure is **purely spatial**—it is identical across all 28 years. Yet `build_neighbor_lookup` rebuilds the mapping for every cell-year combination, effectively repeating the same spatial adjacency resolution 28 times (once per year embedded in each row). This inflates the work by a factor of 28.

### 3. Row-level `lapply` over 6.46 million rows in `compute_neighbor_stats`
Even after the lookup is built, computing max/min/mean via `lapply` over 6.46M list elements, each containing a small numeric vector, is dominated by R-level loop overhead and repeated memory allocations.

**Summary:** The core inefficiency is treating a **spatial** problem as a **cell-year** problem, and using R-level loops instead of vectorized/join-based operations.

---

## Optimization Strategy

The key insight: **build the neighbor table once at the cell level (344K cells × ~4 neighbors each ≈ 1.37M rows), then join yearly attributes onto it.**

### Step-by-step plan:

1. **Build a static edge table once:** Convert `rook_neighbors_unique` (the `nb` object) into a two-column `data.table` of `(cell_id, neighbor_id)` — ~1.37M rows. This is year-independent.

2. **For each year, join cell attributes onto the edge table:** For a given variable (e.g., `ntl`), join the variable's value for each `neighbor_id` in that year onto the edge table. This is a keyed `data.table` join — extremely fast.

3. **Aggregate by `(cell_id, year)`:** Compute `max`, `min`, `mean` of the neighbor values in one grouped aggregation.

4. **Join the aggregated stats back onto the main dataset.**

5. **Repeat for each of the 5 neighbor source variables.**

**Expected speedup:** From ~86 hours to **minutes**. The edge table has ~1.37M rows × 28 years = ~38.4M join rows, but `data.table` keyed joins and grouped aggregations handle this trivially. No R-level row loops at all.

**Memory:** The edge table is ~1.37M rows × 2 integer columns ≈ 11 MB. The yearly-expanded version is ~38.4M rows × 3 columns ≈ 920 MB. Well within 16 GB.

**Preserves:** The trained Random Forest model is untouched. The numerical output (neighbor max, min, mean per cell-year per variable) is identical because the same neighbor relationships and the same aggregation functions are used.

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 0: Convert cell_data to data.table if not already
# ============================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ============================================================
# STEP 1: Build the static spatial edge table ONCE
#
# rook_neighbors_unique is an nb object (list of integer vectors)
# id_order is the vector of cell IDs corresponding to each
# element of the nb object (i.e., id_order[i] is the cell ID
# for the i-th element of rook_neighbors_unique).
# ============================================================
build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(neighbors))
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_idx <- neighbors[[i]]
    if (length(nb_idx) > 0L) {
      n <- length(nb_idx)
      from_id[pos:(pos + n - 1L)] <- id_order[i]
      to_id[pos:(pos + n - 1L)]   <- id_order[nb_idx]
      pos <- pos + n
    }
  }
  
  data.table(cell_id = from_id, neighbor_id = to_id)
}

cat("Building static edge table...\n")
edge_table <- build_edge_table(id_order, rook_neighbors_unique)
cat(sprintf("Edge table: %d directed neighbor relationships\n", nrow(edge_table)))

# ============================================================
# STEP 2: Function to compute neighbor stats for one variable
#          using join + grouped aggregation
# ============================================================
compute_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {
  # Extract only the columns we need for the join:
  # neighbor_id + year -> variable value
  lookup_cols <- c("id", "year", var_name)
  lookup <- cell_dt[, ..lookup_cols]
  setnames(lookup, old = c("id", var_name), new = c("neighbor_id", "nb_val"))
  setkey(lookup, neighbor_id, year)
  
  # Expand edge table by year (cross join edges × years)
  # More memory-efficient: join edges onto the main data's (id, year) pairs,
  # then look up neighbor values.
  
  # Get unique years
  years <- sort(unique(cell_dt$year))
  
  # Cross join: edge_table × years
  # This gives us every (cell_id, neighbor_id, year) combination
  edges_by_year <- edge_dt[, .(year = years), by = .(cell_id, neighbor_id)]
  setkey(edges_by_year, neighbor_id, year)
  
  # Join neighbor values onto the expanded edge table
  edges_by_year[lookup, nb_val := i.nb_val, on = .(neighbor_id, year)]
  
  # Aggregate: for each (cell_id, year), compute max, min, mean
  # of neighbor values (excluding NAs)
  agg <- edges_by_year[
    !is.na(nb_val),
    .(
      nb_max  = max(nb_val),
      nb_min  = min(nb_val),
      nb_mean = mean(nb_val)
    ),
    by = .(cell_id, year)
  ]
  
  # Rename columns to match expected output naming convention
  new_names <- paste0(var_name, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
  setnames(agg, c("nb_max", "nb_min", "nb_mean"), new_names)
  
  # Join aggregated stats back onto cell_dt
  setkey(agg, cell_id, year)
  
  # Return the aggregation table (to be merged externally)
  agg
}

# ============================================================
# STEP 2b: Memory-optimized version that avoids the full
#           cross join materialization (for 16 GB RAM safety)
# ============================================================
compute_neighbor_features_lean <- function(cell_dt, edge_dt, var_name) {
  cat(sprintf("  Computing neighbor features for: %s\n", var_name))
  
  # Build a lookup: (id, year) -> value
  lookup <- cell_dt[, .(neighbor_id = id, year, nb_val = get(var_name))]
  setkey(lookup, neighbor_id, year)
  
  # For each year, join edge_dt with that year's values and aggregate
  years <- sort(unique(cell_dt$year))
  
  result_list <- vector("list", length(years))
  
  for (j in seq_along(years)) {
    yr <- years[j]
    
    # Subset lookup to this year
    lk_yr <- lookup[year == yr, .(neighbor_id, nb_val)]
    setkey(lk_yr, neighbor_id)
    
    # Join neighbor values onto edge table
    edges_yr <- edge_dt[lk_yr, on = .(neighbor_id), nomatch = NULL]
    
    # Aggregate by cell_id
    agg_yr <- edges_yr[
      !is.na(nb_val),
      .(
        nb_max  = max(nb_val),
        nb_min  = min(nb_val),
        nb_mean = mean(nb_val)
      ),
      by = .(cell_id)
    ]
    agg_yr[, year := yr]
    
    result_list[[j]] <- agg_yr
  }
  
  agg <- rbindlist(result_list)
  
  # Rename columns
  new_names <- paste0(var_name, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
  setnames(agg, c("nb_max", "nb_min", "nb_mean"), new_names)
  
  setkey(agg, cell_id, year)
  agg
}

# ============================================================
# STEP 3: Outer loop — compute for all 5 variables and merge
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Ensure cell_data is keyed for fast joins
setkey(cell_data, id, year)

cat("Computing neighbor features for all variables...\n")
t0 <- proc.time()

for (var_name in neighbor_source_vars) {
  agg <- compute_neighbor_features_lean(cell_data, edge_table, var_name)
  
  # Determine the new column names
  new_cols <- paste0(var_name, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
  
  # Remove old columns if they exist (idempotent re-runs)
  old_cols <- intersect(new_cols, names(cell_data))
  if (length(old_cols) > 0) {
    cell_data[, (old_cols) := NULL]
  }
  
  # Merge onto cell_data
  # Use id = cell_id mapping
  setnames(agg, "cell_id", "id")
  setkey(agg, id, year)
  cell_data <- agg[cell_data, on = .(id, year)]
  setkey(cell_data, id, year)
  
  cat(sprintf("  -> Merged %s features. cell_data now has %d columns.\n",
              var_name, ncol(cell_data)))
}

elapsed <- (proc.time() - t0)["elapsed"]
cat(sprintf("Done. Total time: %.1f seconds (%.1f minutes)\n", elapsed, elapsed / 60))

# ============================================================
# STEP 4: Predict with the existing trained Random Forest
#
# The trained model object (e.g., `rf_model`) is unchanged.
# The feature columns are numerically identical to the original
# pipeline's output, so predictions are preserved exactly.
# ============================================================
# Example (uncomment and adapt to your actual model object):
#
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Why This Produces Identical Results

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor definition | `rook_neighbors_unique[[i]]` | Same `nb` object → same edge table |
| Neighbor value lookup | `vals[idx]` per row | `data.table` keyed join on `(neighbor_id, year)` — same values |
| Aggregation | `max()`, `min()`, `mean()` on non-NA neighbor values | Identical functions with `!is.na()` filter |
| Output columns | `{var}_neighbor_max`, `{var}_neighbor_min`, `{var}_neighbor_mean` | Same names, same values |
| RF model | Untouched | Untouched — only `predict()` is called |

The numerical estimand is preserved exactly because the same neighbor pairs, the same variable values, and the same summary functions (max, min, mean with NA removal) are used. No approximation or sampling is introduced.

## Expected Performance

| Stage | Rows Processed | Estimated Time |
|---|---|---|
| Build edge table | ~1.37M edges | < 1 second |
| Per variable, per year: join + aggregate | ~1.37M edges × 28 years | ~5–15 seconds per variable |
| All 5 variables | 5 × above | ~1–2 minutes total |
| **Total** | | **~2–5 minutes** vs. ~86 hours |

The speedup factor is roughly **1,000–2,500×**, achieved entirely by eliminating R-level row iteration and exploiting the year-invariant spatial topology.