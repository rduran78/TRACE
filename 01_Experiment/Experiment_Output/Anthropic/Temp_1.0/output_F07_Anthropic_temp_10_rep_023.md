 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` — O(n²)-class overhead
- `lapply` over **6.46 million rows**, each performing string pasting, named-vector lookups (`idx_lookup[neighbor_keys]`), and NA filtering.
- `idx_lookup` is a **named character vector with 6.46M entries**. Each lookup by name triggers a linear or hash-table scan per key. With ~1.37M neighbor relationships replicated across 28 years, this produces tens of millions of named-vector lookups — extremely slow in R.
- String operations (`paste(id, year, sep="_")`) are repeated millions of times.

### Bottleneck 2: `compute_neighbor_stats` — repeated per variable but structure is reusable
- The `lapply` over 6.46M list elements is called **5 times** (once per variable). Each call iterates over the full panel, indexing into a numeric vector and computing max/min/mean. This is pure-R looping with no vectorization.

### Memory concern
- `neighbor_lookup` is a list of 6.46M integer vectors. Each list element has overhead (~128 bytes per SEXP). Even if most cells have 4 neighbors, this list alone can consume **several GB** just in R object overhead.

**Estimated wall-clock:** The named-vector lookups and per-row `lapply` calls over 6.46M rows dominate. The 86+ hour estimate is credible.

---

## Optimization Strategy

### 1. Replace the per-row list with a flat edge-list (CSR-style) representation
Instead of a 6.46M-element list, build a **sparse adjacency structure as two integer vectors** (a pointer vector and a neighbor-index vector), equivalent to Compressed Sparse Row format. This eliminates millions of R list elements and their per-element overhead.

### 2. Vectorize the neighbor lookup construction
- Use `data.table` for fast keyed joins instead of named-vector lookups.
- Expand the spatial neighbor list into an edge data.frame `(cell_id, neighbor_cell_id)` once (1.37M rows).
- Cross-join with years to get `(cell_id, year, neighbor_cell_id)` → ~1.37M × 28 ≈ 38.4M rows (but only for existing cell-years).
- Join against the panel to resolve each `(neighbor_cell_id, year)` → row index.
- This replaces 6.46M R-level iterations with a single vectorized merge.

### 3. Vectorize `compute_neighbor_stats` using `data.table` grouped aggregation
- Using the flat edge-list, join in the variable values, then `group by` the focal-row index and compute `max`, `min`, `mean` in one vectorized pass per variable.
- With `data.table`, each variable takes seconds, not hours.

### 4. Preserve the trained RF model and numerical estimand
- The output columns have the same names and identical numerical values (max, min, mean of non-NA neighbor values).
- No retraining needed.

**Expected speedup:** From 86+ hours to **~5–15 minutes** total.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 0: Convert panel to data.table and ensure row ordering is preserved
# ──────────────────────────────────────────────────────────────────────
cell_dt <- as.data.table(cell_data)
cell_dt[, .row_idx := .I]  # preserve original row order

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Expand spatial nb object into a flat edge-list (directed)
#
# rook_neighbors_unique is an nb object: a list of length
# length(id_order), where element i contains integer indices into
# id_order of the rook neighbors of cell id_order[i].
# ──────────────────────────────────────────────────────────────────────
build_edge_list <- function(id_order, neighbors) {
  # neighbors[[i]] contains integer positions referencing id_order
  n <- length(neighbors)
  from_list <- vector("list", n)
  to_list   <- vector("list", n)
  for (i in seq_len(n)) {
    nb <- neighbors[[i]]
    # spdep::nb encodes "no neighbors" as a single 0L
    if (length(nb) == 1L && nb[1] == 0L) next
    from_list[[i]] <- rep(id_order[i], length(nb))
    to_list[[i]]   <- id_order[nb]
  }
  data.table(
    focal_id    = unlist(from_list, use.names = FALSE),
    neighbor_id = unlist(to_list,   use.names = FALSE)
  )
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
cat("Spatial edges:", nrow(edge_dt), "\n")

# ──────────────────────────────────────────────────────────────────────
# STEP 2: Expand edges across years via keyed join to the panel
#
# For every (focal_id, year) row in the panel, find the row indices
# of its spatial neighbors in the same year.
# ──────────────────────────────────────────────────────────────────────

# Keyed lookup: (id, year) → .row_idx
setkey(cell_dt, id, year)

# Start from the focal side: get (focal_id, year, focal_row_idx)
focal_info <- cell_dt[, .(focal_id = id, year, focal_row_idx = .row_idx)]

# Join edges onto focal rows: for each focal row, repeat its neighbors
#   Result columns: focal_id, year, focal_row_idx, neighbor_id
edges_with_year <- edge_dt[focal_info, on = .(focal_id), allow.cartesian = TRUE, nomatch = 0L]

# Now resolve neighbor_id + year → neighbor_row_idx
# Build a small lookup
neighbor_key <- cell_dt[, .(neighbor_id = id, year, neighbor_row_idx = .row_idx)]
setkey(neighbor_key, neighbor_id, year)
setkey(edges_with_year, neighbor_id, year)

edges_resolved <- neighbor_key[edges_with_year, on = .(neighbor_id, year), nomatch = NA]
# Keep only edges where the neighbor actually exists in the panel that year
edges_resolved <- edges_resolved[!is.na(neighbor_row_idx)]

cat("Panel-expanded directed edges:", nrow(edges_resolved), "\n")

# ──────────────────────────────────────────────────────────────────────
# STEP 3: Compute neighbor stats for each variable — fully vectorized
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-extract the columns we need from the original ordering
# (cell_dt may have been re-keyed, so index by .row_idx)
setkey(cell_dt, .row_idx)

for (var_name in neighbor_source_vars) {
  cat("Processing neighbor stats for:", var_name, "... ")
  
  # Pull neighbor values via integer indexing (fast)
  vals_vec <- cell_dt[[var_name]]  # ordered by .row_idx because we setkey'd
  
  # Attach neighbor values to the edge table
  edges_resolved[, nval := vals_vec[neighbor_row_idx]]
  
  # Drop NAs before aggregation
  edges_valid <- edges_resolved[!is.na(nval)]
  
  # Grouped aggregation: one pass
  agg <- edges_valid[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), keyby = .(focal_row_idx)]
  
  # Initialize result columns with NA
  max_col  <- paste0("nb_max_",  var_name)
  min_col  <- paste0("nb_min_",  var_name)
  mean_col <- paste0("nb_mean_", var_name)
  
  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]
  
  # Fill in aggregated values by row index
  cell_dt[agg$focal_row_idx, (max_col)  := agg$nb_max]
  cell_dt[agg$focal_row_idx, (min_col)  := agg$nb_min]
  cell_dt[agg$focal_row_idx, (mean_col) := agg$nb_mean]
  
  cat("done.\n")
}

# ──────────────────────────────────────────────────────────────────────
# STEP 4: Clean up and restore original row order
# ──────────────────────────────────────────────────────────────────────
setorder(cell_dt, .row_idx)
cell_dt[, .row_idx := NULL]

# Convert back to data.frame if downstream code expects it
cell_data <- as.data.frame(cell_dt)

# ──────────────────────────────────────────────────────────────────────
# The trained Random Forest model is untouched.
# The 15 new columns (3 stats × 5 vars) are numerically identical
# to the original implementation's output (max, min, mean of
# non-NA rook-neighbor values per cell-year).
# ──────────────────────────────────────────────────────────────────────
```

---

## Summary of gains

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup construction** | 6.46M R-level iterations with named-vector string lookups | Single `data.table` keyed join (~38M rows) |
| **Stat computation per variable** | 6.46M R-level `lapply` iterations | Single `data.table` grouped aggregation |
| **Memory for adjacency** | 6.46M-element R list (~2–4 GB overhead) | Flat integer-column data.table (~300 MB) |
| **Total estimated time** | 86+ hours | **5–15 minutes** |
| **Numerical result** | max/min/mean of non-NA rook-neighbor values | Identical |
| **RF model** | Preserved (not retrained) | Preserved (not retrained) |

The key insight is replacing millions of R-level loop iterations and named-vector lookups with vectorized `data.table` joins and grouped aggregations, which operate in C-optimized code paths internally.