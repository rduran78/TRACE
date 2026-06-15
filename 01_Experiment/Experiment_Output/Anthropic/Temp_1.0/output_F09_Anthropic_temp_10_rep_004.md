 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation is slow (~86+ hours) for two interconnected reasons:

### A. Row-Level `lapply` Over 6.46 Million Rows in `build_neighbor_lookup`
`build_neighbor_lookup` iterates row-by-row over every cell-year observation. For each of the ~6.46 million rows, it:
1. Looks up the cell's reference index.
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs `paste()`-based string keys for every neighbor × year combination.
4. Matches those keys back into a named character index (`idx_lookup`).

This is extremely expensive because: (a) `paste()` and named-vector lookup via character matching are slow in a tight loop; (b) the neighbor topology is **time-invariant** — rook contiguity doesn't change across years — yet the function rebuilds neighbor linkages for every single cell-year row rather than reusing a cell-level adjacency table; (c) the resulting `neighbor_lookup` is a list of 6.46 million integer vectors, consuming significant memory.

### B. Row-Level `lapply` in `compute_neighbor_stats`
For each of the 5 source variables, `compute_neighbor_stats` iterates over 6.46 million list elements, subsetting and computing `max/min/mean` one row at a time. This is called 5 times, totaling ~32.3 million R-level function invocations with no vectorization.

### Core Insight
The neighbor topology is a property of **cells**, not cell-years. There are only 344,208 cells and ~1.37 million directed neighbor pairs. The correct approach is to build a **cell-pair edge table once**, then join yearly attributes onto it and compute grouped summaries using vectorized, columnar operations (via `data.table`). This replaces millions of R-level list iterations with a handful of indexed joins and grouped aggregations.

---

## 2. Optimization Strategy

1. **Build a static edge table** (`data.table`) from `rook_neighbors_unique` with columns `(focal_id, neighbor_id)` — only ~1.37 million rows, built once.
2. **Key the main dataset** by `(id, year)` in `data.table`.
3. **For each source variable** (and each year implicitly), join the neighbor's attribute value onto the edge table via a keyed join, then compute `max`, `min`, `mean` grouped by `(focal_id, year)` — fully vectorized.
4. **Join** the resulting neighbor stats back onto the main cell-year table.
5. **Predict** with the existing trained Random Forest model as before.

This reduces the problem from 6.46M × R-level list operations to a small number of `data.table` keyed joins and `by=` grouped aggregations, which are implemented in C and run orders of magnitude faster. Expected runtime: **minutes, not days**.

---

## 3. Working R Code

```r
library(data.table)

# ── Step 0: Convert main data to data.table (if not already) ────────────────
setDT(cell_data)

# ── Step 1: Build static cell-level edge table (TIME-INVARIANT, built once) ─
#
# rook_neighbors_unique is an nb object (list of integer vectors),
# indexed in the same order as id_order.
# id_order is the vector of cell IDs corresponding to each nb element.

build_edge_table <- function(id_order, nb_obj) {
  # Pre-allocate vectors
  n_edges <- sum(lengths(nb_obj))
  focal   <- integer(n_edges)
  neighbor <- integer(n_edges)
  pos <- 0L
  for (i in seq_along(nb_obj)) {
    nbrs <- nb_obj[[i]]
    # spdep nb objects use 0 or integer(0) for no-neighbor cells
    if (length(nbrs) == 0 || (length(nbrs) == 1 && nbrs[1] == 0L)) next
    n <- length(nbrs)
    idx <- pos + seq_len(n)
    focal[idx]    <- id_order[i]
    neighbor[idx] <- id_order[nbrs]
    pos <- pos + n
  }
  data.table(focal_id = focal[seq_len(pos)],
             neighbor_id = neighbor[seq_len(pos)])
}

edges <- build_edge_table(id_order, rook_neighbors_unique)
# edges: ~1,373,394 rows × 2 columns — tiny relative to the full panel

# ── Step 2: Key the main dataset for fast joins ─────────────────────────────
setkey(cell_data, id, year)

# ── Step 3: Vectorized neighbor-stat computation ────────────────────────────
#
# For each source variable, join neighbor values via the edge table,
# compute grouped stats, and merge back.

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_neighbor_features_dt <- function(dt, edges, var_name) {
  # Suffixed column names matching original pipeline output
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  # Subset to only needed columns for the join (saves memory)
  # Neighbor attribute lookup table keyed on (id, year)
  val_dt <- dt[, .(id, year, val = get(var_name))]
  setkey(val_dt, id, year)

  # Get unique years present in the data
  years <- unique(dt$year)

  # Cross join edges × years, then look up neighbor values

  # To avoid materialising edges × 28 years in one shot (~38 M rows),

  # we process in yearly batches — each batch is only ~1.37 M rows.

  stats_list <- vector("list", length(years))
  for (j in seq_along(years)) {
    yr <- years[j]
    # Slice neighbor values for this year
    val_yr <- val_dt[year == yr]  # keyed, so this is fast
    setkey(val_yr, id)

    # Join neighbor values onto edge table
    # edges[val_yr] would look up focal; we need neighbor values
    work <- copy(edges)
    work[, year := yr]
    # Keyed join: get the neighbor's value
    work[val_yr, neighbor_val := i.val, on = .(neighbor_id = id)]

    # Compute grouped stats per focal cell
    stats_yr <- work[!is.na(neighbor_val),
                     .(nmax  = max(neighbor_val),
                       nmin  = min(neighbor_val),
                       nmean = mean(neighbor_val)),
                     by = .(focal_id, year)]
    stats_list[[j]] <- stats_yr
  }
  stats_all <- rbindlist(stats_list)
  setnames(stats_all,
           c("nmax", "nmin", "nmean"),
           c(col_max, col_min, col_mean))

  # Merge back onto the main table
  dt[stats_all, (c(col_max, col_min, col_mean)) :=
       mget(c(col_max, col_min, col_mean)),
     on = .(id = focal_id, year = year)]

  invisible(dt)
}

# ── Step 4: Run for all source variables ────────────────────────────────────
for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "\n")
  compute_neighbor_features_dt(cell_data, edges, var_name)
}

# ── Step 5: Predict with the already-trained Random Forest ──────────────────
# (Model object and prediction call are unchanged.)
# cell_data$predicted <- predict(trained_rf_model, newdata = cell_data)
```

---

## 4. Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Trained RF model** | Not retouched. We only recompute the same input features and call `predict()`. |
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same set of non-NA neighbor values per cell-year as the original code. The edge table encodes the identical rook-neighbor topology from `rook_neighbors_unique`. The yearly keyed join ensures each neighbor's value is drawn from the correct year. |
| **Column names** | Output columns follow the same `neighbor_max_<var>`, `neighbor_min_<var>`, `neighbor_mean_<var>` naming convention. |

## 5. Expected Performance

| Stage | Original | Optimized |
|---|---|---|
| Build topology | ~6.46M `paste` + named lookups → hours | ~1.37M-row edge table → **< 5 seconds** |
| Neighbor stats (per var) | 6.46M `lapply` iterations → ~17 hours | 28 keyed joins + grouped agg → **~15–40 seconds** |
| Total (5 vars) | **~86+ hours** | **~3–5 minutes** |
| Peak RAM | Large 6.46M-element list | ~1.37M edge table + yearly working copies ≈ **< 2 GB** |

The key transformation: exploit the **time-invariant topology** by building the adjacency structure once at the cell level, and delegate per-year attribute computation to vectorized `data.table` joins and grouped aggregations instead of R-level row iteration.