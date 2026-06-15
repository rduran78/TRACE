 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The bottleneck is clear: **`build_neighbor_lookup`** creates a 6.46-million-element list via `lapply` over every row, performing per-row string-pasting, hash-lookups, and NA-filtering. This is O(rows × avg_neighbors) with enormous R-level loop overhead. Then **`compute_neighbor_stats`** iterates over that same 6.46M-element list again—once per variable (×5). Together they produce ~86+ hours of runtime.

**Root causes:**

| Component | Problem |
|---|---|
| `build_neighbor_lookup` | 6.46M iterations of an R `lapply` each doing `paste`, named-vector lookup, and `is.na` filtering. String-based keying (`"id_year"`) is slow. |
| `compute_neighbor_stats` | Another 6.46M-iteration `lapply` **per variable** (×5 = 32.3M iterations), each subsetting a vector by index, removing NAs, and computing `max/min/mean`. |
| Memory | The `neighbor_lookup` list of 6.46M integer vectors is itself large, plus the `do.call(rbind, ...)` on 6.46M 3-element vectors is extremely slow (incremental allocation). |

**Why raster focal/kernel operations are not the right analogy here:**
Focal operations assume a regular grid with a fixed kernel window. While the grid *is* regular, the panel structure (cell × year) and the precomputed `spdep::nb` object with potentially irregular boundaries (coastal cells, edge cells with < 4 neighbors) mean a focal convolution would need careful masking per time-slice and would not naturally produce max/min/mean of *only* rook neighbors. More importantly, the `spdep::nb` object is already serialized and tested—reimplementing via focal risks subtle mismatches at boundaries, violating the requirement to **preserve the original numerical estimand**. The correct strategy is to vectorize the existing neighbor logic rather than switch paradigms.

---

## 2. Optimization Strategy

### Key ideas

1. **Eliminate the row-level lookup entirely.** Instead of building a 6.46M-element list mapping each *row* to its neighbor *rows*, exploit the panel structure: every cell has the same neighbors in every year. Build a sparse adjacency structure once over the 344,208 cells, then use **vectorized year-aligned joins** to compute stats.

2. **Use `data.table` for vectorized grouped operations.** Expand the neighbor pairs into a long edge table (`from_id`, `to_id`), join on `(to_id, year)` to pull neighbor values, then group by `(from_id, year)` to compute `max`, `min`, `mean`—all in one vectorized pass per variable.

3. **Sparse matrix alternative for `mean` (optional further speedup).** A sparse adjacency matrix `W` (344,208 × 344,208) times a dense matrix of variable values (344,208 × 28) gives neighbor sums; divide by neighbor counts for `mean`. But since we also need `max` and `min`, the `data.table` join approach is the cleanest single-pass solution.

4. **Expected runtime:** The edge table has ~1.37M directed edges. For each year, that's ~1.37M join lookups → 28 years × 1.37M ≈ 38.4M rows in the long edge table. Grouping 38.4M rows by `(from_id, year)` to compute 3 summary stats is a ~10-second operation in `data.table` per variable. **Total: ~1–2 minutes for all 5 variables**, down from 86+ hours.

---

## 3. Working R Code

```r
library(data.table)

# ─────────────────────────────────────────────────────────────────────
# 0. Convert cell_data to data.table (if not already) and ensure key
# ─────────────────────────────────────────────────────────────────────
cell_dt <- as.data.table(cell_data)

# Ensure 'id' and 'year' are integer for fast joins
cell_dt[, id   := as.integer(id)]
cell_dt[, year := as.integer(year)]

# ─────────────────────────────────────────────────────────────────────
# 1. Build a long edge table from the nb object (one-time, ~seconds)
#    rook_neighbors_unique is an nb object: list of length 344,208
#    id_order is the vector mapping list index → cell id
# ─────────────────────────────────────────────────────────────────────
build_edge_table <- function(id_order, nb_obj) {
  # nb objects store neighbor indices (into the same list); 0 means no neighbors
  from_ids <- rep(
    as.integer(id_order),
    times = vapply(nb_obj, function(x) {
      nx <- x[x != 0L]
      length(nx)
    }, integer(1))
  )
  to_ids <- as.integer(id_order[unlist(lapply(nb_obj, function(x) {
    nx <- x[x != 0L]
    nx
  }))])
  data.table(from_id = from_ids, to_id = to_ids)
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# edge_dt has ~1,373,394 rows (directed pairs)

cat("Edge table rows:", nrow(edge_dt), "\n")

# ─────────────────────────────────────────────────────────────────────
# 2. Function: compute and attach neighbor max/min/mean for one variable
# ─────────────────────────────────────────────────────────────────────
compute_and_add_neighbor_features_fast <- function(cell_dt, var_name, edge_dt) {

  # Column names for output (must match original pipeline naming)
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  # --- a) Build a small lookup: (id, year, value) ----
  lookup <- cell_dt[, .(to_id = id, year = year, nb_val = get(var_name))]
  setkey(lookup, to_id, year)

  # --- b) Expand edge table across all years ----
  # Instead of a full cross-join (38M rows up front), use a rolling/equi join:
  #   For every (from_id, to_id) pair, join each year from cell_dt.
  # More memory-efficient: get the unique years, then cross with edges.
  years <- sort(unique(cell_dt$year))

  # CJ of edges × years: ~1.37M × 28 ≈ 38.5M rows — fits in RAM (~600 MB)
  edge_year <- CJ_dt_edges(edge_dt, years)

  # --- c) Join to get neighbor values ----
  setkey(edge_year, to_id, year)
  edge_year[lookup, nb_val := i.nb_val, on = .(to_id, year)]

  # --- d) Aggregate: group by (from_id, year) ----
  stats <- edge_year[
    !is.na(nb_val),
    .(
      nb_max  = max(nb_val),
      nb_min  = min(nb_val),
      nb_mean = mean(nb_val)
    ),
    keyby = .(from_id, year)
  ]

  # --- e) Merge back into cell_dt ----
  setkey(cell_dt, id, year)
  setkey(stats, from_id, year)

  cell_dt[stats, (col_max)  := i.nb_max,  on = .(id = from_id, year)]
  cell_dt[stats, (col_min)  := i.nb_min,  on = .(id = from_id, year)]
  cell_dt[stats, (col_mean) := i.nb_mean, on = .(id = from_id, year)]

  # Cells with no valid neighbors get NA (already NA by default in data.table)
  invisible(cell_dt)
}

# Helper: cross-join edges with years (memory-efficient)
CJ_dt_edges <- function(edge_dt, years) {
  n_edges <- nrow(edge_dt)
  n_years <- length(years)
  data.table(
    from_id = rep(edge_dt$from_id, times = n_years),
    to_id   = rep(edge_dt$to_id,   times = n_years),
    year    = rep(years, each = n_edges)
  )
}

# ─────────────────────────────────────────────────────────────────────
# 3. Run for all 5 neighbor source variables
# ─────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "...\n")
  t0 <- proc.time()
  compute_and_add_neighbor_features_fast(cell_dt, var_name, edge_dt)
  elapsed <- (proc.time() - t0)["elapsed"]
  cat("  Done in", round(elapsed, 1), "seconds.\n")
}

# ─────────────────────────────────────────────────────────────────────
# 4. Convert back to data.frame if downstream code expects one,
#    then predict with the existing (trained) Random Forest model
# ─────────────────────────────────────────────────────────────────────
cell_data <- as.data.frame(cell_dt)

# The trained RF model object is assumed to be in memory (e.g., `rf_model`)
# Predict as before — the model is NOT retrained:
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## 4. Memory Budget Check

| Object | Approximate Size |
|---|---|
| `cell_dt` (6.46M × 110 cols) | ~5.7 GB |
| `edge_dt` (1.37M × 2 int cols) | ~11 MB |
| `edge_year` (38.4M × 3 cols, peak) | ~920 MB |
| Temporary `lookup`, `stats` | ~200 MB |
| **Peak total** | **~7–8 GB** ✓ fits 16 GB |

The `edge_year` table is the peak allocation and is created/destroyed once per variable. If memory is tight, process one variable at a time and `rm(edge_year); gc()` between iterations (which the loop already does implicitly since `edge_year` is local to the function).

---

## 5. Why This Preserves the Original Numerical Estimand

- The `edge_dt` encodes exactly the same directed rook-neighbor pairs as `rook_neighbors_unique`.
- The join on `(to_id, year)` pulls exactly the same neighbor values as the original `idx_lookup`.
- `max`, `min`, `mean` with `na.rm` (via `!is.na` filter before grouping) matches the original `compute_neighbor_stats` logic.
- The trained Random Forest model is used as-is for prediction—no retraining occurs.
- The output columns are identically named and carry the same values, so downstream code is unaffected.

**Expected wall-clock time: ~1–2 minutes total (all 5 variables), vs. the original 86+ hours.**