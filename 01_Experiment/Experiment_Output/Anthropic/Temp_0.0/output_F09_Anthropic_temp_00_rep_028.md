 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(N²)-like in practice
This function iterates over **all ~6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs **year-specific string keys** by pasting cell ID and year (expensive string allocation × 6.46M).
- Performs **named-vector lookups** (`idx_lookup[neighbor_keys]`) — named vector lookup in R is O(n) per query against a character-named vector, not O(1) like a hash. With ~6.46M entries in `idx_lookup` and ~6.46M queries each touching ~4 neighbors on average, this is catastrophically slow.

The fundamental flaw: **the neighbor topology is purely spatial and time-invariant, yet the lookup is rebuilt per cell-year row**, fusing spatial structure with temporal indexing in the most expensive possible way.

### 2. `compute_neighbor_stats` uses per-row `lapply`
Even after the lookup is built, computing stats via `lapply` over 6.46M list elements with R-level anonymous functions is slow due to R's interpreter overhead.

### 3. Memory pressure
A list of 6.46M integer vectors (the neighbor lookup) consumes substantial RAM and creates GC pressure on a 16 GB machine.

---

## Optimization Strategy

**Core insight:** The neighbor graph is **time-invariant**. There are only 344,208 cells and ~1.37M directed rook-neighbor pairs. Build a **spatial-only edge table once**, then use vectorized joins and grouped aggregations per year to compute neighbor stats. This eliminates the 6.46M-row list entirely.

**Steps:**

1. **Build a spatial edge table** (`data.table` with columns `id`, `neighbor_id`) from the `spdep::nb` object — only ~1.37M rows.
2. **For each variable**, join the cell-year attribute table onto the edge table by `(neighbor_id, year)`, then compute grouped `max`, `min`, `mean` by `(id, year)` — fully vectorized via `data.table`.
3. **Join results back** to the main dataset.

This replaces 6.46M R-level iterations with a handful of vectorized `data.table` joins and group-bys, reducing runtime from ~86 hours to **minutes**.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# STEP 1 — Build the time-invariant spatial edge table (once)
# ---------------------------------------------------------------
# rook_neighbors_unique : spdep nb object (list of integer vectors)
# id_order              : vector of cell IDs in the same order as the nb object

build_edge_table <- function(id_order, neighbors) {
  # neighbors[[i]] gives the indices (into id_order) of cell i's rook neighbors
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  # Remove zero-length / no-neighbor entries (spdep uses 0L for "no neighbors")
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# ~1.37 M rows, two integer columns — trivial memory

cat("Edge table rows:", nrow(edge_dt), "\n")

# ---------------------------------------------------------------
# STEP 2 — Convert main data to data.table (if not already)
# ---------------------------------------------------------------
setDT(cell_data)
setkey(cell_data, id, year)          # index for fast joins

# ---------------------------------------------------------------
# STEP 3 — Vectorized neighbor-stat computation
# ---------------------------------------------------------------
compute_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {
  # Subset to only the columns we need for the join
  # (id, year, <var_name>)
  val_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setnames(val_dt, "id", "neighbor_id")
  setkey(val_dt, neighbor_id, year)

  # Join: for every (id, neighbor_id) edge, attach the neighbor's value
  # in each year.  We add year from the focal cell.
  # Approach: expand edges × years via join on neighbor_id + year.
  #
  # edge_dt has (id, neighbor_id).
  # val_dt  has (neighbor_id, year, val).
  # Merge on neighbor_id → gives (id, neighbor_id, year, val).

  merged <- merge(edge_dt, val_dt, by = "neighbor_id",
                  allow.cartesian = TRUE, sort = FALSE)
  # merged now has columns: neighbor_id, id, year, val
  # Each row = one directed edge in one year with the neighbor's value.

  # Aggregate by (id, year)
  stats <- merged[!is.na(val),
                  .(nmax  = max(val),
                    nmin  = min(val),
                    nmean = mean(val)),
                  keyby = .(id, year)]

  # Name the output columns to match the original pipeline's convention
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats, c("nmax", "nmin", "nmean"),
                  c(max_col,  min_col,  mean_col))

  stats
}

# ---------------------------------------------------------------
# STEP 4 — Loop over the 5 source variables and join back
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "...\n")
  stats_dt <- compute_neighbor_features_fast(cell_data, edge_dt, var_name)

  # Left-join the three new columns onto cell_data
  cell_data <- merge(cell_data, stats_dt, by = c("id", "year"), all.x = TRUE)
  cat("  Done.\n")
}

# ---------------------------------------------------------------
# STEP 5 — Predict with the existing (already-trained) RF model
# ---------------------------------------------------------------
# The trained model object (e.g., `rf_model`) is unchanged.
# All original predictor columns plus the new neighbor_* columns
# are now present in cell_data.

cell_data[, prediction := predict(rf_model, newdata = .SD)]

cat("Pipeline complete.\n")
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets and the same variable values. The edge table encodes the identical rook-neighbor topology. No approximation is introduced. |
| **Trained RF model** | The model object is never modified or retrained. Only `predict()` is called. |
| **Missing-data handling** | `!is.na(val)` in the aggregation mirrors the original `neighbor_vals[!is.na(neighbor_vals)]` filter. Cells with no valid neighbors receive `NA` via the `all.x = TRUE` left join, matching the original `c(NA, NA, NA)` return. |

---

## Expected Performance

| Stage | Old | New |
|---|---|---|
| Build lookup / edge table | ~hours (6.46M string ops) | **< 1 second** (1.37M integer pairs) |
| Compute stats (5 vars) | ~80+ hours (6.46M × 5 `lapply`) | **~2–5 minutes** (vectorized `data.table` join + group-by) |
| Total | **~86+ hours** | **~5–10 minutes** on a 16 GB laptop |

The speedup is roughly **500–1000×**, achieved entirely by separating the time-invariant spatial topology from the time-varying attributes and replacing R-level iteration with vectorized `data.table` operations.