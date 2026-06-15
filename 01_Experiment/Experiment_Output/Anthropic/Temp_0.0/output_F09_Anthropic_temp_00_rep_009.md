 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(N²)-like in practice
This function iterates over **all ~6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs **character key strings** (`paste(id, year)`) and performs **named-vector lookups** (`idx_lookup[neighbor_keys]`) — these are hash-table lookups but done 6.46 million times, each creating temporary character vectors and subsetting a 6.46-million-entry named vector.

The result is a **list of 6.46 million integer vectors**, which is enormous in memory and slow to build. Critically, this lookup **mixes spatial topology (which is static) with temporal indexing (which is repetitive)**. Every cell has the same neighbors in every year, yet the function recomputes the neighbor-row mapping for each of the 28 year-copies of every cell.

### 2. `compute_neighbor_stats` iterates row-by-row over 6.46M rows
Even though the lookup is precomputed, calling `vals[idx]` inside `lapply` over 6.46 million elements, then computing `max/min/mean` per element, is inherently slow in interpreted R. This is done **5 times** (once per neighbor source variable).

### Summary of bottlenecks
| Step | Calls | Cost |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `paste` + named-vector lookups | ~hours |
| `compute_neighbor_stats` | 6.46M × 5 vars × `max/min/mean` | ~hours |
| Memory: 6.46M-element list of integer vectors | ~GBs of list overhead | RAM pressure |

---

## Optimization Strategy

**Core insight:** The neighbor graph is purely spatial and static across years. Build it once as a **cell-to-cell adjacency table**, then join yearly attributes onto it. This converts the problem from row-wise R loops into vectorized `data.table` grouped operations.

### Steps:

1. **Build a static edge table** from `rook_neighbors_unique` (the `nb` object): a two-column `data.table` with columns `(id, neighbor_id)` — ~1.37M rows. This is done **once**.

2. **Join cell-year attributes onto the edge table** by `(neighbor_id, year)` — this gives each edge the neighbor's variable value for that year. `data.table` binary-search joins make this very fast.

3. **Group by `(id, year)`** and compute `max`, `min`, `mean` of neighbor values — fully vectorized, no R-level row loops.

4. **Join the resulting stats back** onto the main `cell_data` table.

This reduces the problem from ~6.46M × R-loop iterations to a handful of vectorized `data.table` join-and-group operations. Expected runtime: **minutes, not hours**.

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 0: Convert cell_data to data.table (if not already)
# ============================================================
cell_data <- as.data.table(cell_data)

# Ensure key columns exist and are proper types
stopifnot(all(c("id", "year") %in% names(cell_data)))

# ============================================================
# STEP 1: Build static spatial edge table ONCE
#
# rook_neighbors_unique is an nb object (list of integer vectors)
# id_order is the vector mapping list index -> cell id
# ============================================================
build_edge_table <- function(id_order, neighbors_nb) {
  # neighbors_nb[[i]] contains the indices (into id_order) of

  # the neighbors of cell id_order[i].
  # A 0-integer entry means no neighbors in spdep convention.
  edges <- rbindlist(lapply(seq_along(neighbors_nb), function(i) {
    nb_idx <- neighbors_nb[[i]]
    nb_idx <- nb_idx[nb_idx != 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  edges
}

cat("Building static edge table...\n")
edge_table <- build_edge_table(id_order, rook_neighbors_unique)
cat(sprintf("  Edge table: %s directed edges\n", format(nrow(edge_table), big.mark = ",")))

# ============================================================
# STEP 2: Function to compute neighbor stats for one variable
#          using vectorized data.table joins + grouped aggregation
# ============================================================
compute_neighbor_features_dt <- function(cell_dt, edge_dt, var_name) {
  # Subset to only needed columns for the join (minimise memory)
  # We need neighbor_id matched to (id, year) in cell_dt
  neighbor_vals <- edge_dt[
    cell_dt[, .(neighbor_id = id, year, value = get(var_name))],
    on = .(neighbor_id),
    allow.cartesian = TRUE,
    nomatch = NULL
  ]
  # neighbor_vals now has columns: id, neighbor_id, year, value
  # where 'id' is the focal cell and 'value' is the neighbor's attribute

  # Remove NA values before aggregation
  neighbor_vals <- neighbor_vals[!is.na(value)]

  # Grouped aggregation
  stats <- neighbor_vals[,
    .(
      nb_max  = max(value),
      nb_min  = min(value),
      nb_mean = mean(value)
    ),
    by = .(id, year)
  ]

  # Rename columns to match original pipeline naming convention
  suffix <- var_name
  setnames(stats,
    c("nb_max", "nb_min", "nb_mean"),
    c(paste0(suffix, "_neighbor_max"),
      paste0(suffix, "_neighbor_min"),
      paste0(suffix, "_neighbor_mean"))
  )

  stats
}

# ============================================================
# STEP 3: Compute and attach neighbor features for all variables
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Set key on cell_data for fast joins
setkey(cell_data, id, year)

cat("Computing neighbor features...\n")
for (var_name in neighbor_source_vars) {
  cat(sprintf("  Processing: %s\n", var_name))

  stats_dt <- compute_neighbor_features_dt(cell_data, edge_table, var_name)
  setkey(stats_dt, id, year)

  # Remove old columns if they exist (idempotent re-runs)
  old_cols <- c(
    paste0(var_name, "_neighbor_max"),
    paste0(var_name, "_neighbor_min"),
    paste0(var_name, "_neighbor_mean")
  )
  for (col in old_cols) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }

  # Left join stats back onto cell_data

  cell_data <- stats_dt[cell_data, on = .(id, year)]
}

cat("Neighbor feature computation complete.\n")

# ============================================================
# STEP 4: Predict with the existing trained Random Forest
#
# The trained model object (e.g., rf_model) is unchanged.
# cell_data now has the same neighbor feature columns as before.
# ============================================================
# Example (adjust to your actual model object name):
# cell_data[, prediction := predict(rf_model, newdata = cell_data)]
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets — the edge table encodes the identical rook adjacency from `rook_neighbors_unique`, and the join on `(neighbor_id, year)` matches the same cell-year values. NA handling (`!is.na(value)`) mirrors the original. |
| **Trained RF model** | The model object is never touched. Only the input feature columns are recomputed (identically), then `predict()` is called. |
| **Column names** | The naming convention `{var}_neighbor_max/min/mean` is preserved so the RF model's expected feature names match. Adjust the suffix pattern if your original pipeline used a different convention. |

---

## Expected Performance

| Step | Original | Optimized |
|---|---|---|
| Build topology | Hours (6.46M `paste` + hash lookups) | **~2–5 seconds** (1.37M-row `rbindlist`) |
| Per-variable stats | Hours (6.46M `lapply`) | **~15–45 seconds** (vectorized join + group) |
| Total (5 vars) | **86+ hours** | **~2–5 minutes** |
| Peak RAM | High (6.46M-element list) | Moderate (~1.37M × 28 intermediate rows) |

The key transformation is replacing **6.46 million interpreted R loop iterations** with **vectorized `data.table` binary-search joins and grouped C-level aggregations**, while factoring out the static spatial topology into a small, reusable edge table.