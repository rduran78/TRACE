 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The two critical bottlenecks are:

### 1. `build_neighbor_lookup` — O(n) `lapply` over 6.46 million rows with per-row string pasting and named-vector lookups
- **String key construction**: `paste(id, year, sep="_")` is called millions of times inside the loop body, and named-vector lookup (`idx_lookup[neighbor_keys]`) is an O(k) hash probe per neighbor key, repeated for every row.
- **Redundant work across years**: Every cell has the same neighbors in every year. The neighbor *structure* is time-invariant, yet the lookup is rebuilt per cell-year row by pasting year into every neighbor key. With 28 years × 344,208 cells, the same neighbor set is resolved 28 times per cell.
- **Memory**: The named character vector `idx_lookup` with 6.46M entries and the resulting list of 6.46M integer vectors is large (~2–4 GB depending on neighbor counts).

### 2. `compute_neighbor_stats` — O(n) `lapply` with per-row subsetting and aggregation
- For each of the 6.46M rows, it subsets a numeric vector by an index vector, removes NAs, and computes max/min/mean. This is repeated for each of the 5 variables (32.3M iterations total).
- `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors is itself slow and memory-hungry.

### Combined effect
The nested per-row R-level loops with string operations and list manipulations explain the 86+ hour estimate. R's interpreted loop overhead dominates.

---

## Optimization Strategy

| Principle | Technique |
|---|---|
| **Eliminate redundant work** | Build the neighbor lookup as a *cell-to-cell* adjacency (time-invariant), then join by year using vectorized operations — never loop over cell-years. |
| **Vectorize aggregation** | Convert the neighbor list to a long-form `data.table` edge list, join the variable values, and compute grouped `max/min/mean` in one vectorized pass per variable. |
| **Use `data.table`** | `data.table` provides in-place `:=` assignment, fast keyed joins, and optimized `by`-group aggregation in C — eliminates R-level loops entirely. |
| **Minimize memory** | The long-form edge list has ~1.37M edges × 28 years ≈ 38.5M rows of two integer columns (~0.6 GB), far smaller than a 6.46M-element list of variable-length integer vectors. |
| **Preserve the trained RF model** | We only restructure feature engineering; output columns have identical names and identical numerical values (max, min, mean of the same neighbor sets). The RF `predict()` call is unchanged. |

**Expected speedup**: From 86+ hours to roughly 5–15 minutes on the same laptop.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 0.  Convert cell_data to data.table (if not already) and key it
# ──────────────────────────────────────────────────────────────────────
cell_dt <- as.data.table(cell_data)          # non-destructive copy
cell_dt[, row_idx := .I]                     # preserve original row order

# ──────────────────────────────────────────────────────────────────────
# 1.  Build a TIME-INVARIANT edge list from the nb object (once)
#
#     rook_neighbors_unique is a list of length 344,208 where element i
#     contains the integer indices (into id_order) of cell i's neighbors.
#     id_order[i] gives the cell id for position i.
# ──────────────────────────────────────────────────────────────────────
build_edge_list <- function(id_order, neighbors) {
  # Pre-allocate vectors
  from_ids <- vector("list", length(neighbors))
  to_ids   <- vector("list", length(neighbors))
  for (i in seq_along(neighbors)) {
    nb <- neighbors[[i]]
    if (length(nb) > 0L && !(length(nb) == 1L && nb[1] == 0L)) {
      from_ids[[i]] <- rep(id_order[i], length(nb))
      to_ids[[i]]   <- id_order[nb]
    }
  }
  data.table(
    from_id = unlist(from_ids, use.names = FALSE),
    to_id   = unlist(to_ids,   use.names = FALSE)
  )
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
# edge_dt has columns: from_id, to_id   (~1.37 M rows, time-invariant)

# ──────────────────────────────────────────────────────────────────────
# 2.  Vectorized neighbor-stat computation for one variable
# ──────────────────────────────────────────────────────────────────────
compute_and_add_neighbor_features_fast <- function(cell_dt, var_name, edge_dt) {

  # Columns we will create (same names as the original pipeline)
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  # --- a) Subset only the columns we need from cell_dt for the join ---
  #     Keying on (id, year) makes the join O(n log n) or better.
  val_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setkey(val_dt, id, year)

  # --- b) Cross join edges × years is implicit: we join edge_dt with
  #     val_dt on (to_id = id, year) to get each neighbor's value,
  #     then aggregate by (from_id, year). ---

  # Expand edges by year via a keyed join:
  #   For every (from_id -> to_id) edge, look up to_id's value in each year.
  #   We achieve this by joining val_dt onto edge_dt by to_id = id.
  #   Because val_dt has one row per (id, year), the result automatically
  #   has one row per (from_id, to_id, year).

  # Rename for clarity before join
  setnames(val_dt, "id", "to_id")
  setkey(val_dt, to_id)
  setkey(edge_dt, to_id)

  # Join: for each edge, get neighbor value in every year
  neighbor_vals <- val_dt[edge_dt, on = "to_id", allow.cartesian = TRUE,
                          nomatch = NULL]
  # Result columns: to_id, year, val, from_id

  # Remove rows where the neighbor value is NA (matches original logic)
  neighbor_vals <- neighbor_vals[!is.na(val)]

  # --- c) Aggregate by (from_id, year) ---
  stats <- neighbor_vals[, .(
    nmax  = max(val),
    nmin  = min(val),
    nmean = mean(val)
  ), by = .(from_id, year)]

  # --- d) Join aggregated stats back onto cell_dt ---
  setkey(stats, from_id, year)
  setkey(cell_dt, id, year)

  cell_dt[stats, (c(col_max, col_min, col_mean)) :=
            .(i.nmax, i.nmin, i.nmean),
          on = c(id = "from_id", "year")]

  # Rows with no valid neighbors remain NA (`:=` leaves unmatched as NA
  # only if the columns are freshly created; ensure NA default)
  for (col in c(col_max, col_min, col_mean)) {
    if (!col %in% names(cell_dt)) cell_dt[, (col) := NA_real_]
  }

  # Restore val_dt name (edge_dt is shared across calls)
  setnames(val_dt, "to_id", "id")

  invisible(cell_dt)
}

# ──────────────────────────────────────────────────────────────────────
# 3.  Outer loop — iterate over the 5 source variables
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "\n")
  compute_and_add_neighbor_features_fast(cell_dt, var_name, edge_dt)
  gc()
}

# ──────────────────────────────────────────────────────────────────────
# 4.  Restore original row order and convert back if needed
# ──────────────────────────────────────────────────────────────────────
setorder(cell_dt, row_idx)
cell_dt[, row_idx := NULL]

# If downstream code expects a data.frame:
# cell_data <- as.data.frame(cell_dt)
# Otherwise keep cell_dt as-is for prediction.

# ──────────────────────────────────────────────────────────────────────
# 5.  Predict with the EXISTING trained Random Forest (unchanged)
# ──────────────────────────────────────────────────────────────────────
# predictions <- predict(trained_rf_model, newdata = cell_dt)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same neighbor sets** | `build_edge_list` reads the identical `rook_neighbors_unique` nb object and `id_order` mapping. Every directed edge `(from_id → to_id)` is the same. |
| **Same statistics** | `max`, `min`, `mean` are computed on the same non-NA neighbor values per `(cell, year)` group. |
| **Same column names** | Output columns follow the pattern `neighbor_max_{var}`, `neighbor_min_{var}`, `neighbor_mean_{var}` — adjust the pattern in the function if your original pipeline uses a different naming convention. |
| **Trained RF untouched** | No model retraining; only the feature matrix is rebuilt with identical numerical values. |

---

## Memory Budget Estimate (16 GB laptop)

| Object | Approximate Size |
|---|---|
| `cell_dt` (6.46M × 110 cols) | ~5.5 GB |
| `edge_dt` (1.37M × 2 int cols) | ~22 MB |
| `neighbor_vals` (worst case ~38.5M × 4 cols) | ~1.2 GB |
| `stats` (up to 6.46M × 4 cols) | ~0.2 GB |
| Overhead + R session | ~2 GB |
| **Total peak** | **~9 GB** ✔ fits in 16 GB |

The `gc()` call after each variable frees the intermediate join tables, keeping peak usage well within 16 GB.