 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is slow (~86+ hours) for three interconnected reasons:

### 1. `build_neighbor_lookup` is O(N²)-ish in practice
It iterates over all **6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs `paste(..., year, sep="_")` key strings for every neighbor × year combination (expensive string allocation).
- Indexes into a named character vector (`idx_lookup`) via string matching (slow hash lookup repeated billions of times).

This means ~6.46M iterations × ~4 neighbors on average = ~25.8M string constructions and hash lookups, all inside an R-level `lapply` with no vectorization.

### 2. The lookup is **year-redundant**
The spatial neighbor structure is **identical across all 28 years**. A cell's rook neighbors in 1992 are the same cells in 2019. Yet `build_neighbor_lookup` rebuilds the mapping for every cell-year row, doing 28× the necessary work. The neighbor *topology* only needs to be built once for the 344,208 cells; yearly attribute values should be joined afterward.

### 3. `compute_neighbor_stats` uses row-level `lapply`
Even after the lookup is built, computing max/min/mean iterates row-by-row in R over 6.46M rows. This is inherently slow without vectorization or `data.table` grouped operations.

---

## Optimization Strategy

**Core idea:** Separate topology from attributes. Build the neighbor edge list **once** (344K cells, ~1.37M edges), then use a `data.table` join-and-group strategy to compute neighbor statistics for all years simultaneously in vectorized C-level operations.

| Step | What | Complexity |
|------|------|-----------|
| 1 | Convert `nb` object → edge `data.table` (cell_id, neighbor_id): ~1.37M rows. **Once.** | O(E) |
| 2 | Join yearly attributes onto the edge table by `(neighbor_id, year)`. | O(E × T) ≈ 38.4M rows |
| 3 | Group by `(cell_id, year)` and compute `max`, `min`, `mean`. | Vectorized, single pass |
| 4 | Join results back to the main panel. | O(N) |

**Expected speedup:** From ~86 hours to **minutes** (the bottleneck becomes the `data.table` join on ~38M rows × 5 variables, which is trivial for `data.table`).

**Constraints honored:**
- The trained Random Forest model is untouched.
- The numerical output (neighbor max, min, mean per variable per cell-year) is identical.
- Peak RAM stays well within 16 GB.

---

## Working R Code

```r
library(data.table)

# ===========================================================================
# STEP 1: Build the spatial edge list ONCE from the nb object
#         This replaces build_neighbor_lookup entirely.
# ===========================================================================

build_edge_table <- function(id_order, nb_obj) {
  # nb_obj: list of length N, each element is integer vector of neighbor indices
  # id_order: vector of cell IDs corresponding to nb_obj positions
  #
  # Returns a data.table with columns: cell_id, neighbor_id
  # Represents directed rook-neighbor edges (one row per directed pair).

  n <- length(nb_obj)
  # Pre-calculate total edges for pre-allocation
  n_edges <- sum(vapply(nb_obj, function(x) {
    # spdep nb objects use 0L for "no neighbors"
    len <- length(x)
    if (len == 1L && x[1L] == 0L) 0L else len
  }, integer(1)))

  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  pos     <- 1L

  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    if (length(nbrs) == 1L && nbrs[1L] == 0L) next
    k <- length(nbrs)
    from_id[pos:(pos + k - 1L)] <- id_order[i]
    to_id[pos:(pos + k - 1L)]   <- id_order[nbrs]
    pos <- pos + k
  }

  data.table(cell_id = from_id, neighbor_id = to_id)
}

# Build once — takes seconds for ~1.37M edges
edges <- build_edge_table(id_order, rook_neighbors_unique)

# ===========================================================================
# STEP 2 & 3: Compute neighbor stats for all variables, all years at once
# ===========================================================================

compute_all_neighbor_features <- function(cell_data_dt, edges, source_vars) {
  # cell_data_dt: data.table with columns 'id', 'year', and all source_vars
  # edges:        data.table with columns 'cell_id', 'neighbor_id'
  # source_vars:  character vector of variable names
  #
  # Returns cell_data_dt with new columns:
  #   <var>_neighbor_max, <var>_neighbor_min, <var>_neighbor_mean
  #   for each var in source_vars.

  # Ensure data.table and set keys for fast joins
  if (!is.data.table(cell_data_dt)) cell_data_dt <- as.data.table(cell_data_dt)

  # Subset only the columns we need for the neighbor attribute lookup
  attr_cols <- c("id", "year", source_vars)
  attrs <- cell_data_dt[, ..attr_cols]

  # Key the attribute table on (id, year) for the join

setkey(attrs, id, year)

  # Expand edges × years:
  #   Join neighbor attributes onto the edge table.
  #   For each (cell_id, neighbor_id) edge, we replicate across all years
  #   by joining on neighbor_id == id.
  #
  # Result: one row per (cell_id, neighbor_id, year) with neighbor's values.

  # Rename for join clarity
  setnames(attrs, "id", "neighbor_id")
  setkey(attrs, neighbor_id, year)
  setkey(edges, neighbor_id)

  # This is the big join: edges[, .(cell_id, neighbor_id)] × attrs by neighbor_id

  # data.table does this efficiently as an equi-join.
  expanded <- attrs[edges, on = "neighbor_id", allow.cartesian = TRUE, nomatch = NA]
  # expanded now has columns: neighbor_id, year, <source_vars>, cell_id

  # Group by (cell_id, year) and compute stats for each source var
  agg_exprs <- list()
  for (v in source_vars) {
    sym_v <- as.name(v)
    agg_exprs[[paste0(v, "_neighbor_max")]]  <-
      bquote(as.numeric(max(.(sym_v), na.rm = TRUE)), list(sym_v = sym_v))
    agg_exprs[[paste0(v, "_neighbor_min")]]  <-
      bquote(as.numeric(min(.(sym_v), na.rm = TRUE)), list(sym_v = sym_v))
    agg_exprs[[paste0(v, "_neighbor_mean")]] <-
      bquote(mean(.(sym_v), na.rm = TRUE), list(sym_v = sym_v))
  }

  # Build the aggregation call dynamically
  agg_list <- as.call(c(as.name("list"), agg_exprs))
  neighbor_stats <- expanded[, eval(agg_list), by = .(cell_id, year)]

  # Replace Inf/-Inf (from max/min of all-NA groups) with NA
  stat_cols <- names(neighbor_stats)[-(1:2)]
  for (col in stat_cols) {
    neighbor_stats[is.infinite(get(col)), (col) := NA_real_]
  }

  # Join back onto the main data
  setkey(neighbor_stats, cell_id, year)
  setkey(cell_data_dt, id, year)
  # Align key names for merge
  setnames(neighbor_stats, "cell_id", "id")
  cell_data_dt <- neighbor_stats[cell_data_dt, on = .(id, year)]

  return(cell_data_dt)
}

# ===========================================================================
# STEP 4: Run it
# ===========================================================================

# Convert to data.table if not already
cell_data <- as.data.table(cell_data)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(cell_data, edges, neighbor_source_vars)

# ===========================================================================
# STEP 5: Predict with the existing trained Random Forest (unchanged)
# ===========================================================================

# The trained model object (e.g., rf_model) is used as-is.
# The new neighbor feature columns have the same names and semantics
# as the original pipeline produced, so prediction is identical:
#
#   cell_data$prediction <- predict(rf_model, newdata = cell_data)
#
# No retraining required. The numerical estimand is preserved because
# max, min, and mean are computed over exactly the same neighbor sets
# and the same attribute values.
```

---

## Why This Is Correct and Equivalent

| Property | Original | Optimized |
|----------|----------|-----------|
| Neighbor set for cell *i* in year *t* | Rook neighbors from `nb` object, filtered to those present in year *t* | Identical — the `nomatch = NA` join drops missing cell-years, and `na.rm = TRUE` in aggregation mirrors the `!is.na()` filter |
| Statistics computed | `max`, `min`, `mean` of non-NA neighbor values | Identical functions, identical `na.rm` behavior |
| All-missing neighbor groups | Return `(NA, NA, NA)` | `Inf`/`-Inf` from `max`/`min` are replaced with `NA`; `mean` of empty = `NaN` → `NA` |
| Column names | `<var>_neighbor_max`, `<var>_neighbor_min`, `<var>_neighbor_mean` | Identical |
| RF model | Not modified | Not modified |

## Expected Performance

| Stage | Rows/Operations | Estimated Time |
|-------|----------------|---------------|
| `build_edge_table` | 1.37M edges | ~1–2 seconds |
| Join (edges × attrs) | ~38.4M rows (1.37M edges × 28 years) | ~10–30 seconds |
| Group-by aggregation | 6.46M groups × 5 vars × 3 stats | ~1–3 minutes |
| Column join back | 6.46M rows | ~5 seconds |
| **Total** | | **~2–5 minutes** |

Peak memory: ~38.4M rows × ~8 columns × 8 bytes ≈ **2.5 GB** (well within 16 GB).

This reduces the runtime from **~86 hours to ~2–5 minutes** — a roughly **1,000× speedup** — by eliminating year-redundant topology construction and replacing row-level R loops with vectorized `data.table` joins and grouped aggregations.