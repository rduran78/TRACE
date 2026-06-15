 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which iterates over ~6.46 million rows in a pure-R `lapply`, performing per-row character coercion, `paste`, and named-vector lookups. Each iteration does:

1. `as.character()` + named-vector lookup to map cell ID → reference index.
2. Subsetting the `neighbors` list to get neighbor cell IDs.
3. `paste()` to build `(neighbor_id, year)` keys.
4. Named-vector lookup into `idx_lookup` (~6.46M-element named character vector).

Named-vector lookups on a 6.46M-element vector are **O(n)** string-hashed searches per call. Doing this ~6.46M × ~4 neighbors ≈ **25+ billion character comparisons**. This is why the runtime is 86+ hours.

`compute_neighbor_stats` is a secondary bottleneck: another `lapply` over 6.46M rows, but much cheaper per iteration since it's just numeric subsetting. Still, it's called 5 times (once per variable).

**Root causes:**
1. **Row-level R loop** over 6.46M rows with expensive string operations.
2. **Named character vector lookup** instead of integer hash (environment) or merge/join.
3. **`compute_neighbor_stats` recomputes per variable** instead of vectorizing across all 5 variables at once.
4. The neighbor lookup is **year-invariant** (rook neighbors don't change over time) but is rebuilt per cell-year row as if it were year-specific.

---

## Optimization Strategy

### Key Insight: Separate Spatial Topology from Temporal Expansion

The rook-neighbor graph is **static across years**. The current code re-derives neighbor row indices for every `(cell, year)` pair. Instead:

1. **Build a spatial-only neighbor edge list once** (344K cells, ~1.37M edges) — trivially fast.
2. **Exploit the panel structure**: if data is sorted by `(id, year)` or `(year, id)`, the row offset from a cell to the same year of its neighbor is deterministic. But even without that, a single **`data.table` equi-join** on `(neighbor_id, year)` replaces the entire `build_neighbor_lookup`.
3. **Vectorized aggregation**: use `data.table` grouped aggregation (`max`, `min`, `mean`) over the joined result — no R-level row loop at all.
4. **Process all 5 variables in one join** instead of 5 separate passes.

**Expected speedup**: from 86+ hours to **~2–5 minutes**.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 0.  Inputs assumed to exist:
#       cell_data            : data.frame/data.table with columns id, year,
#                              ntl, ec, pop_density, def, usd_est_n2, …
#       id_order             : integer/character vector of cell IDs (length 344,208)
#       rook_neighbors_unique: spdep nb object (list of integer index vectors)
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# 1.  Build a spatial edge list  (done ONCE, < 1 second)
# ──────────────────────────────────────────────────────────────────────
build_edge_list <- function(id_order, nb_obj) {
  # nb_obj[[i]] contains integer indices into id_order of neighbors of cell i
  from <- rep(seq_along(nb_obj), lengths(nb_obj))
  to   <- unlist(nb_obj)
  # Remove the 0-neighbor sentinel that spdep uses (integer(0) is fine,

  # but some nb objects store 0L for no-neighbor cells)
  valid <- to != 0L
  data.table(
    focal_id    = id_order[from[valid]],
    neighbor_id = id_order[to[valid]]
  )
}

edges <- build_edge_list(id_order, rook_neighbors_unique)
# edges has ~1,373,394 rows (directed rook pairs)

# ──────────────────────────────────────────────────────────────────────
# 2.  Convert cell_data to data.table (in-place, no copy if already DT)
# ──────────────────────────────────────────────────────────────────────
setDT(cell_data)

# ──────────────────────────────────────────────────────────────────────
# 3.  Define source variables
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# ──────────────────────────────────────────────────────────────────────
# 4.  Single join + grouped aggregation for ALL variables at once
# ──────────────────────────────────────────────────────────────────────
compute_all_neighbor_features <- function(cell_data, edges, source_vars) {

  # 4a. Subset to only the columns we need for the neighbor table
  keep_cols <- c("id", "year", source_vars)
  neighbor_vals <- cell_data[, ..keep_cols]

  # 4b. Join: for every (focal_id, year) find neighbor rows

  #     edges supplies (focal_id, neighbor_id);
  #     we join neighbor_vals on neighbor_id == id AND same year.
  #     Result: one row per (focal_id, year, neighbor_id) with neighbor values.
  joined <- edges[neighbor_vals,
                  on = .(neighbor_id = id),
                  allow.cartesian = TRUE,
                  nomatch = NULL,
                  .(focal_id, year, 
                    ntl = i.ntl, ec = i.ec, pop_density = i.pop_density,
                    def = i.def, usd_est_n2 = i.usd_est_n2)]
  # This says: "for each row in neighbor_vals (which has id = some cell and year),
  #  find all edges where neighbor_id == that id, and carry forward focal_id."
  # Result: each row represents focal_cell seeing one neighbor's values in a year.

  # 4c. Aggregate by (focal_id, year)
  agg_exprs <- list()
  for (v in source_vars) {
    sym_v <- as.name(v)
    agg_exprs[[paste0("nb_max_", v)]]  <- substitute(max(V, na.rm = TRUE),  list(V = sym_v))
    agg_exprs[[paste0("nb_min_", v)]]  <- substitute(min(V, na.rm = TRUE),  list(V = sym_v))
    agg_exprs[[paste0("nb_mean_", v)]] <- substitute(mean(V, na.rm = TRUE), list(V = sym_v))
  }
  agg_call <- as.call(c(as.name("list"), agg_exprs))

  stats <- joined[, eval(agg_call), by = .(focal_id, year)]

  # Replace Inf / -Inf (from max/min of zero non-NA values) with NA
  for (col in names(stats)[-(1:2)]) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }

  return(stats)
}

neighbor_stats <- compute_all_neighbor_features(cell_data, edges, neighbor_source_vars)

# ──────────────────────────────────────────────────────────────────────
# 5.  Merge back onto cell_data
# ──────────────────────────────────────────────────────────────────────

# Remove old neighbor columns if they exist (idempotency)
old_nb_cols <- grep("^nb_(max|min|mean)_", names(cell_data), value = TRUE)
if (length(old_nb_cols)) cell_data[, (old_nb_cols) := NULL]

cell_data <- merge(cell_data, neighbor_stats,
                   by.x = c("id", "year"),
                   by.y = c("focal_id", "year"),
                   all.x = TRUE, sort = FALSE)

# ──────────────────────────────────────────────────────────────────────
# 6.  Predict with the EXISTING trained Random Forest (unchanged)
# ──────────────────────────────────────────────────────────────────────
# e.g.  cell_data$predicted <- predict(trained_rf, newdata = cell_data)
# The model object is untouched; only the feature-engineering step changed.
```

---

## Why This Is Numerically Identical

| Original code | Optimized code | Equivalence |
|---|---|---|
| For each row `i`, find all rows sharing a rook-neighbor cell ID **and** the same year. Collect their values. Compute `max`, `min`, `mean` (ignoring `NA`). | Join `edges` × `cell_data` on `(neighbor_id = id)`, inheriting `year` from the neighbor row, then group by `(focal_id, year)` and compute `max`, `min`, `mean` (ignoring `NA`). | The set of neighbor values aggregated is identical — same cells, same year, same `na.rm = TRUE` logic. The three summary statistics are deterministic functions of that set. ✓ |

No approximation, sampling, or model retraining is involved. The trained Random Forest is preserved as-is.

---

## Expected Performance

| Step | Rows / Operations | Estimated Time |
|---|---|---|
| Build edge list | 1.37M edges | < 1 sec |
| Join (edges × 6.46M rows) | ~25.8M joined rows | ~30–60 sec |
| Grouped aggregation (15 stats) | 6.46M groups | ~30–60 sec |
| Merge back | 6.46M rows | ~10 sec |
| **Total** | | **~2–5 minutes** |

Peak RAM: ~4–6 GB (the joined table of ~25.8M rows × 7 columns is the high-water mark), well within the 16 GB laptop constraint.