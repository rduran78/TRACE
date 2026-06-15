 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is slow (~86+ hours) for three interconnected reasons:

### 1. Row-level `lapply` over 6.46 million rows in `build_neighbor_lookup`
For every single cell-year row, the function performs string pasting (`paste(id, year, sep="_")`), named-vector lookups, and `NA` filtering. With ~6.46M rows, this creates millions of small R-level operations. Named vector lookups via character keys (`idx_lookup[neighbor_keys]`) are hash-table lookups repeated millions of times inside an interpreted loop—extremely slow.

### 2. The neighbor topology is **year-invariant**, but the lookup is rebuilt per cell-year
The rook-neighbor relationships between cells never change across years. Yet `build_neighbor_lookup` produces a list of length 6.46M (one entry per cell-year) rather than a list of length 344,208 (one entry per cell). This 28× blowup is entirely unnecessary.

### 3. `compute_neighbor_stats` iterates over 6.46M list elements in R
Even after the lookup is built, computing max/min/mean by iterating `lapply` over 6.46M entries—each performing subsetting, `NA` removal, and three summary statistics—is extremely slow in interpreted R.

**Bottom line:** The entire approach conflates *spatial topology* (fixed) with *temporal attribute data* (varying). The fix is to separate the two, build the adjacency table once, then use fast vectorized joins and grouped aggregations.

---

## Optimization Strategy

1. **Build a static, two-column adjacency edge table once** from the `nb` object: `data.table(cell_id, neighbor_id)`. This table has ~1.37M rows and never changes.

2. **Join yearly attributes onto the edge table** by `(neighbor_id, year)` using `data.table` keyed joins. This is a single vectorized merge per variable—no R-level loops.

3. **Compute grouped summary statistics** `(max, min, mean)` by `(cell_id, year)` using `data.table`'s `by=` grouped aggregation—fully vectorized in C.

4. **Merge the resulting features back** onto the main dataset with a single keyed join per variable.

5. The trained Random Forest model is **never touched**—only the feature-engineering pipeline changes. The numerical outputs (max, min, mean of neighbor values) are **identical** to the original.

**Expected speedup:** From ~86 hours to **minutes** (typically 2–10 minutes on a 16 GB laptop), because all heavy operations become vectorized C-level `data.table` operations over ~38M edge-year rows (1.37M edges × 28 years).

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 0: Convert main data to data.table (if not already)
# ──────────────────────────────────────────────────────────────────────
# cell_data must have columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# plus all other predictor columns. We keep everything intact.
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Preserve original row order so the final dataset aligns with any
# downstream indexing (e.g., the trained Random Forest expects this order).
cell_data[, .row_order := .I]

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build the static adjacency edge table ONCE
# ──────────────────────────────────────────────────────────────────────
# rook_neighbors_unique : spdep nb object (list of integer index vectors)
# id_order              : vector of cell IDs in the same order as the nb object

build_adjacency_table <- function(id_order, neighbors_nb) {
  # Each element i of neighbors_nb is an integer vector of neighbor indices

  # into id_order. Index 0 (spdep convention for no neighbors) is handled.
  edges <- rbindlist(lapply(seq_along(neighbors_nb), function(i) {
    nb_idx <- neighbors_nb[[i]]
    nb_idx <- nb_idx[nb_idx > 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(cell_id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  return(edges)
}

cat("Building static adjacency table...\n")
adj_table <- build_adjacency_table(id_order, rook_neighbors_unique)
# adj_table has ~1,373,394 rows: (cell_id, neighbor_id)
cat(sprintf("  Adjacency table: %s directed edges\n", format(nrow(adj_table), big.mark = ",")))

# ──────────────────────────────────────────────────────────────────────
# STEP 2: For each neighbor source variable, compute neighbor stats
#         via vectorized join + grouped aggregation
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Create a slim lookup keyed by (id, year) for joining neighbor attributes.
# We only need the neighbor source variable columns plus id and year.
attr_cols <- unique(c("id", "year", neighbor_source_vars))
attr_lookup <- cell_data[, ..attr_cols]
setkey(attr_lookup, id, year)

# We will also key adj_table on neighbor_id for the join.
setkey(adj_table, neighbor_id)

cat("Computing neighbor features...\n")

for (var_name in neighbor_source_vars) {

  cat(sprintf("  Processing: %s\n", var_name))

  # Slim attribute table for this variable: (id, year, value)
  var_dt <- attr_lookup[, .(id, year, value = get(var_name))]
  setnames(var_dt, "id", "neighbor_id")
  setkey(var_dt, neighbor_id)

  # Cross join: expand adj_table × years by joining on neighbor_id

  # Result: for every (cell_id, neighbor_id) edge and every year,
  # we get the neighbor's attribute value.
  # Efficient approach: merge adj_table with var_dt on neighbor_id,
  # which naturally replicates across all years the neighbor appears in.
  edge_vals <- var_dt[adj_table, on = "neighbor_id", allow.cartesian = TRUE, nomatch = NA]
  # edge_vals columns: neighbor_id, year, value, cell_id

  # Remove rows where the neighbor's value is NA (matches original logic)
  edge_vals <- edge_vals[!is.na(value)]

  # Grouped aggregation: compute max, min, mean per (cell_id, year)
  stats <- edge_vals[, .(
    nb_max  = max(value),
    nb_min  = min(value),
    nb_mean = mean(value)
  ), by = .(cell_id, year)]

  # Name the new columns to match the original pipeline's naming convention
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
  setnames(stats, "cell_id", "id")

  # Merge back onto cell_data by (id, year)
  # First, remove these columns if they already exist (idempotent reruns)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }

  cell_data <- merge(cell_data, stats, by = c("id", "year"), all.x = TRUE)

  # Clean up

  rm(var_dt, edge_vals, stats)
}

# ──────────────────────────────────────────────────────────────────────
# STEP 3: Restore original row order and clean up
# ──────────────────────────────────────────────────────────────────────
setorder(cell_data, .row_order)
cell_data[, .row_order := NULL]

cat("Neighbor feature engineering complete.\n")

# ──────────────────────────────────────────────────────────────────────
# STEP 4: Predict with the existing trained Random Forest
# ──────────────────────────────────────────────────────────────────────
# The trained model object (e.g., rf_model) is unchanged.
# Predictions proceed exactly as before:
#
#   predictions <- predict(rf_model, newdata = cell_data)
#
# The numerical estimand is preserved because the neighbor max, min,
# and mean values are computed identically—only the computational
# method (vectorized joins vs. row-level R loops) has changed.
```

---

## Why This Is Correct and Equivalent

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor topology | Embedded in 6.46M-element list | Explicit 1.37M-row edge table |
| Attribute lookup | Character-key hash per row | `data.table` keyed binary-search join |
| Aggregation | `lapply` + per-element `max/min/mean` | `data.table` grouped `by=` (C-level) |
| NA handling | `neighbor_vals[!is.na(...)]` | `edge_vals[!is.na(value)]` before grouping |
| Result | 3 columns per variable (max, min, mean) | Identical 3 columns per variable |
| RF model | Untouched | Untouched |
| Numerical output | Identical | Identical |

**Expected runtime:** 2–10 minutes on a 16 GB RAM laptop instead of 86+ hours—a roughly **500–2,500× speedup**.