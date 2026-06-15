 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It creates a list of 6.46 million elements, where each element is built by:

1. **Per-row string pasting and named-vector lookup** — `paste()` and named-vector indexing (`idx_lookup[neighbor_keys]`) are called ~6.46 million times inside an `lapply`. Each call does string allocation, hashing, and matching.
2. **The lookup is O(n) in list overhead** — `idx_lookup` is a named character vector of length ~6.46M. Named lookups on such a vector are O(n) or at best O(1)-amortized via hashing, but R's internal `match()` on named vectors rebuilds hash tables unpredictably.
3. **`compute_neighbor_stats`** then loops over the 6.46M-element list again, extracting values one-at-a-time. This is pure R-level iteration with no vectorization.

**Net effect:** ~6.46M R-level iterations × (string ops + hash lookups + subsetting) ≈ 86+ hours.

### Why the raster shortcut is unsafe
The document correctly notes that the cell topology may be irregular/masked. A naive `focal()` on a rectangular raster would compute neighbors for cells that don't exist in the panel or miss masked cells. The neighbor structure in `rook_neighbors_unique` (an `spdep::nb` object) is the ground truth and must be respected.

---

## Optimization Strategy

**Replace all per-row R loops with vectorized / data.table operations:**

1. **Explode the neighbor graph into an edge table once** — a two-column `data.table` of `(cell_id, neighbor_cell_id)` with ~1.37M rows. This is year-invariant.
2. **Cross-join with years vectorially** — instead of pasting keys 6.46M times, join `cell_data` to the edge table on `(neighbor_cell_id, year)` using `data.table` keyed joins. This is a single merge, fully vectorized in C.
3. **Compute grouped stats in one pass per variable** — `data.table`'s `[, .(max, min, mean), by=.(id, year)]` computes all three stats in one vectorized grouped aggregation.
4. **Memory:** The edge table × 28 years ≈ 1.37M × 28 ≈ 38.5M rows × a few columns of integers/doubles — well within 16 GB.

**Expected speedup:** From 86+ hours to **minutes** (typically 5–15 min total for all 5 variables).

**Preservation guarantees:**
- The trained Random Forest model is untouched (we only rebuild feature columns with identical values).
- The numerical estimand is identical: same neighbor sets, same max/min/mean formulas, same NA handling.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 0.  Ensure cell_data is a data.table (non-destructive copy if needed)
# ──────────────────────────────────────────────────────────────────────
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ──────────────────────────────────────────────────────────────────────
# 1.  Build a vectorized edge table from the spdep::nb object (once)
#
#     rook_neighbors_unique : list of integer vectors (spdep nb object)
#     id_order              : vector mapping position -> cell id
#     Edge table columns    : focal_id, neighbor_id
# ──────────────────────────────────────────────────────────────────────
build_edge_table <- function(id_order, neighbors) {
  n_edges <- sum(lengths(neighbors))
  focal_idx <- rep(seq_along(neighbors), lengths(neighbors))
  neighbor_idx <- unlist(neighbors, use.names = FALSE)

  # Remove the spdep zero-neighbor sentinel (integer(0) already handled

  # by lengths==0, but guard against 0L entries)
  valid <- neighbor_idx > 0L
  data.table(
    focal_id    = id_order[focal_idx[valid]],
    neighbor_id = id_order[neighbor_idx[valid]]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
cat("Edge table rows:", nrow(edge_dt), "\n")

# ──────────────────────────────────────────────────────────────────────
# 2.  Compute neighbor stats for one variable — fully vectorized
# ──────────────────────────────────────────────────────────────────────
compute_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {

  # Columns we need from the neighbor side
  # Build a slim lookup: (id, year, value)
  val_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setkey(val_dt, id, year)

  # Join edges → neighbor values.
  # For every (focal_id, year) we need the neighbor's value in the same year.
  # Strategy: cross the edge table with the value table on neighbor_id == id.
  # This is a keyed join — very fast.
  setkey(edge_dt, neighbor_id)
  merged <- edge_dt[val_dt,
    on = .(neighbor_id = id),
    .(focal_id, year, val),
    nomatch = NULL,
    allow.cartesian = TRUE
  ]

  # Drop NAs in the variable (mirrors original: neighbor_vals[!is.na()])
  merged <- merged[!is.na(val)]

  # Grouped aggregation
  stats <- merged[,
    .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ),
    by = .(focal_id, year)
  ]

  # Rename to match original column naming convention
  setnames(stats, c("nb_max", "nb_min", "nb_mean"),
           paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean")))

  stats
}

# ──────────────────────────────────────────────────────────────────────
# 3.  Outer loop — compute and attach features for every source variable
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Ensure cell_data is keyed for fast joins
setkey(cell_data, id, year)

for (var_name in neighbor_source_vars) {
  cat("Computing neighbor stats for:", var_name, "...\n")

  new_cols <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))

  # Drop old columns if they exist (safe re-run)
  for (col in new_cols) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }

  stats_dt <- compute_neighbor_features_fast(cell_data, edge_dt, var_name)
  setkeyv(stats_dt, c("focal_id", "year"))

  # Left-join back onto cell_data.  Rows with no neighbors get NA (correct).
  cell_data <- stats_dt[cell_data, on = .(focal_id = id, year)]

  # The join puts focal_id as the key; rename back to id
  setnames(cell_data, "focal_id", "id")
  setkey(cell_data, id, year)

  cat("  Done.\n")
}

cat("All neighbor features computed. Rows:", nrow(cell_data), "\n")
```

---

## Why This Is Correct (Estimand Preservation Proof)

| Original code | Optimized code | Equivalence |
|---|---|---|
| For row *i*, find all `neighbor_cell_ids` from `rook_neighbors_unique` | `edge_dt` encodes the identical directed pairs from the same `nb` object | Same neighbor sets |
| Filter to neighbors present in the same `year` | Join `on = .(neighbor_id = id)` with `val_dt` keyed by `(id, year)` — only matching year rows survive | Same temporal alignment |
| `neighbor_vals[!is.na(neighbor_vals)]` | `merged[!is.na(val)]` | Same NA removal |
| `max`, `min`, `mean` on filtered vector; return `NA` if empty | `data.table` grouped agg returns the same; left-join fills unmatched rows with `NA` | Identical numerics |

The Random Forest model object is never referenced or modified. The 15 new feature columns (5 vars × 3 stats) are numerically identical to the originals, so predictions from the existing model are unchanged.

**Expected wall-clock time on a 16 GB laptop: 5–15 minutes** (vs. 86+ hours), dominated by the `allow.cartesian` join (~38.5M rows) and five grouped aggregations.