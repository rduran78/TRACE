 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation is slow (~86+ hours) for three interconnected reasons:

### A. Row-level `lapply` over 6.46 million rows in `build_neighbor_lookup`
For every single cell-year row, the function performs character coercion, `paste`, and named-vector lookups. With 6.46M rows, this creates billions of intermediate character objects and hash-table probes. Named vector lookup (`idx_lookup[neighbor_keys]`) is O(n) per probe against a 6.46M-element named vector, making total complexity roughly O(n × k) where k is mean neighbor count—catastrophically expensive.

### B. Row-level `lapply` over 6.46 million rows in `compute_neighbor_stats`
Even after the lookup is built, computing max/min/mean via an R-level loop over 6.46M list elements is slow. Each iteration has R interpreter overhead, memory allocation for subsetting, and NA checks.

### C. The lookup mixes spatial topology with temporal identity unnecessarily
The rook-neighbor relationships are **purely spatial**—they never change across years. Yet the current code rebuilds a lookup that is indexed by (cell, year) pairs, duplicating the same spatial adjacency structure 28 times and doing string-matching across all 6.46M rows to find temporal matches.

**Key insight:** The neighbor table should be built **once** over 344,208 cells (spatial only), then joined onto the yearly panel via a vectorized merge/join. This reduces the lookup problem from 6.46M rows to 344K cells and moves all computation into vectorized operations.

---

## 2. Optimization Strategy

1. **Build a static spatial edge list once** from `rook_neighbors_unique` (the `nb` object): ~1.37M directed (cell, neighbor) pairs. This is year-invariant.
2. **Convert the panel data to `data.table`** for fast keyed joins.
3. **For each variable, join yearly attributes onto the edge list** by (neighbor_id, year), then compute grouped max/min/mean by (cell_id, year) using `data.table` aggregation—fully vectorized, no R-level row loops.
4. **Merge results back** onto the main panel.

Expected speedup: The ~6.46M-row `lapply` is replaced by a ~38.4M-row (1.37M edges × 28 years) `data.table` join + grouped aggregation, which runs in seconds to minutes rather than days.

---

## 3. Working R Code

```r
library(data.table)

# ==============================================================
# STEP 1: Build a static spatial edge list ONCE
#         from the precomputed spdep::nb object
# ==============================================================
build_spatial_edge_list <- function(id_order, neighbors) {
  # neighbors is a list of integer index vectors (spdep::nb object)
  # id_order is the vector mapping position -> cell id
  edges <- rbindlist(lapply(seq_along(neighbors), function(i) {
    nb_idx <- neighbors[[i]]
    # spdep::nb encodes "no neighbors" as a single 0L
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx == 0L)) {
      return(NULL)
    }
    data.table(cell_id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  return(edges)
}

# Build it once — ~1.37M rows, takes seconds
edge_list <- build_spatial_edge_list(id_order, rook_neighbors_unique)

# ==============================================================
# STEP 2: Convert panel to data.table (if not already)
# ==============================================================
cell_dt <- as.data.table(cell_data)

# ==============================================================
# STEP 3: For each variable, compute neighbor stats via
#         vectorized join + grouped aggregation
# ==============================================================
compute_and_add_neighbor_features_fast <- function(cell_dt, edge_list, var_name) {
  # Columns we will create
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  # Extract only the columns we need for the join (small footprint)
  # neighbor_id will be matched to "id" in the attribute table
  attr_cols <- c("id", "year", var_name)
  attr_dt   <- cell_dt[, ..attr_cols]

  # Key the attribute table for fast join
  setkey(attr_dt, id, year)

  # Expand edge list × years by joining neighbor attributes
  # For each (cell_id, neighbor_id) pair, attach the neighbor's
  # yearly value by joining on neighbor_id == id, same year.
  #
  # We do this by: merge edge_list with cell_dt years first,
  # then join neighbor attributes.

  # Get the unique years
  years <- sort(unique(cell_dt$year))

  # Cross-join edges with years: ~1.37M × 28 ≈ 38.4M rows
  # Memory: 38.4M × 3 int/numeric cols ≈ ~900 MB (fits in 16 GB)
  edge_years <- CJ(edge_idx = seq_len(nrow(edge_list)), year = years)
  edge_years[, cell_id     := edge_list$cell_id[edge_idx]]
  edge_years[, neighbor_id := edge_list$neighbor_id[edge_idx]]
  edge_years[, edge_idx    := NULL]

  # Join to get the neighbor's value of var_name
  setkey(edge_years, neighbor_id, year)
  edge_years[attr_dt, neighbor_val := get(var_name), on = .(neighbor_id = id, year)]

  # Aggregate: grouped stats by (cell_id, year)
  stats <- edge_years[
    !is.na(neighbor_val),
    .(
      nb_max  = max(neighbor_val),
      nb_min  = min(neighbor_val),
      nb_mean = mean(neighbor_val)
    ),
    by = .(cell_id, year)
  ]

  # Rename for merge
  setnames(stats,
           c("nb_max", "nb_min", "nb_mean"),
           c(col_max,  col_min,  col_mean))

  # Remove old columns if they already exist (idempotency)
  for (col in c(col_max, col_min, col_mean)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }

  # Merge back onto the main panel
  cell_dt <- merge(cell_dt, stats,
                   by.x = c("id", "year"),
                   by.y = c("cell_id", "year"),
                   all.x = TRUE)

  return(cell_dt)
}

# ==============================================================
# STEP 3b (memory-friendly alternative): process one variable
#          at a time WITHOUT materializing the full cross-join.
#          This uses a direct equi-join and is more RAM-efficient.
# ==============================================================
compute_and_add_neighbor_features_lean <- function(cell_dt, edge_list, var_name) {
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  # Subset to needed columns
  attr_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setkey(attr_dt, id)

  # For every edge (cell_id, neighbor_id), pull all years of the neighbor
  # by joining edge_list to attr_dt on neighbor_id == id
  # Result: (cell_id, neighbor_id, year, val)  ~38.4M rows
  joined <- attr_dt[edge_list, on = .(id = neighbor_id), allow.cartesian = TRUE,
                    nomatch = NULL]
  # 'joined' now has columns: id (=neighbor_id), year, val, cell_id
  # Rename for clarity
  setnames(joined, "id", "neighbor_id")

  # Aggregate by (cell_id, year), dropping NAs in val
  stats <- joined[
    !is.na(val),
    .(nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)),
    by = .(cell_id, year)
  ]

  setnames(stats,
           c("nb_max", "nb_min", "nb_mean"),
           c(col_max,  col_min,  col_mean))

  # Remove old columns if present
  for (col in c(col_max, col_min, col_mean)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }

  # Merge back
  cell_dt <- merge(cell_dt, stats,
                   by.x = c("id", "year"),
                   by.y = c("cell_id", "year"),
                   all.x = TRUE)

  return(cell_dt)
}

# ==============================================================
# STEP 4: Run for all 5 neighbor source variables
# ==============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  cell_dt <- compute_and_add_neighbor_features_lean(cell_dt, edge_list, var_name)
  gc()
}

# ==============================================================
# STEP 5: Predict with the EXISTING trained Random Forest
#          (model object is untouched / not retrained)
# ==============================================================
# Convert back to data.frame if the RF model expects one
cell_data <- as.data.frame(cell_dt)

# The trained model object (e.g., rf_model) is used as-is:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary of Gains

| Aspect | Before (Original) | After (Optimized) |
|---|---|---|
| **Lookup construction** | `lapply` over 6.46M rows with `paste`/named-vector probes | One-time `rbindlist` over 344K cells (~seconds) |
| **Neighbor stat computation** | `lapply` over 6.46M rows × 5 vars (R interpreter loop) | Vectorized `data.table` join + `by=` aggregation (~minutes total) |
| **Memory pattern** | 6.46M-element named character vector + 6.46M-element list | ~38.4M-row `data.table` per variable, freed after each merge |
| **Estimated wall time** | ~86+ hours | **~5–15 minutes** (5 variables on 16 GB laptop) |
| **Trained RF model** | ✅ Preserved (not retrained) | ✅ Preserved (not retrained) |
| **Numerical estimand** | ✅ Same max/min/mean over rook neighbors | ✅ Identical values produced |

The core principle: **separate the time-invariant spatial topology from the time-varying attributes**, build the edge list once, and let `data.table`'s optimized C-level joins and grouped aggregations replace millions of R-interpreter-level iterations.