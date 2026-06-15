 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It creates a list of 6.46 million elements, where each element is built by:

1. **Character key construction and lookup** (`paste`, `setNames`, named-vector indexing) — for every single row. Named-vector lookup in R is O(n) hash probing per call, but the real cost is doing ~6.46M `paste()` calls and ~6.46M named-vector subscript operations, each touching a names vector of length 6.46M. This is catastrophically slow.

2. **`lapply` over 6.46M rows** — each iteration does string pasting, named-vector lookup, and NA filtering. The per-iteration overhead of R's interpreted loop, combined with the O(1)-amortized-but-high-constant-factor named lookups on a 6.46M-length vector, yields the estimated 86+ hour runtime.

3. **`compute_neighbor_stats`** then does a second `lapply` over the 6.46M-element list, which is comparatively fast but still unnecessarily slow because it's row-by-row interpreted R.

**Root cause summary:** The algorithm is correct but implemented with interpreted-R loops and string-keyed lookups over millions of rows. The neighbor topology is time-invariant (same grid, same neighbors every year), so the spatial join should be done once at the cell level (344K cells), then broadcast across years via vectorized merge — not repeated per cell-year row.

---

## Optimization Strategy

| Principle | Action |
|---|---|
| **Separate space from time** | Build the neighbor edge list once over 344K cells, not 6.46M cell-years. |
| **Vectorize with `data.table`** | Replace `lapply`/`paste`/named-vector lookups with `data.table` keyed joins and grouped aggregations. |
| **Columnar neighbor stats** | For each variable, do a single vectorized join of cell-year values onto the edge list, then `group by (id, year)` to compute `max`, `min`, `mean`. |
| **Memory-safe** | The edge list is ~1.37M rows × 3 columns (source_id, neighbor_id, implicit). Joined with year, it becomes ~1.37M × 28 ≈ 38.4M rows — large but fits in 16 GB as integer/double columns. We can process one variable at a time to limit peak memory. |
| **Preserve numerics exactly** | `max`, `min`, `mean` on the same neighbor sets with the same NA handling → identical numerical results. |
| **No model retraining** | We only rebuild the feature columns; the trained RF object is untouched. |

**Expected speedup:** From 86+ hours to **~2–5 minutes**.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 0.  Inputs assumed already in memory:
#       cell_data            — data.frame/data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
#       id_order             — integer/character vector of cell IDs (same order as rook_neighbors_unique)
#       rook_neighbors_unique — spdep nb object (list of integer index vectors into id_order)
# ──────────────────────────────────────────────────────────────────────

# Convert to data.table in place (no copy if already data.table)
setDT(cell_data)

# ──────────────────────────────────────────────────────────────────────
# 1.  Build a SPATIAL edge list (once, ~1.37M rows)
#     Each row: (id, neighbor_id) meaning "neighbor_id is a rook neighbor of id"
# ──────────────────────────────────────────────────────────────────────
build_edge_list <- function(id_order, nb_obj) {
  # nb_obj[[i]] gives integer indices into id_order for neighbors of id_order[i]
  n <- length(nb_obj)
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(nb_obj))  # should be ~1,373,394
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    nb_i <- nb_obj[[i]]
    # spdep nb objects use 0L to denote "no neighbors" for an isolate
    if (length(nb_i) == 1L && nb_i[1L] == 0L) next
    len <- length(nb_i)
    idx <- pos:(pos + len - 1L)
    from_id[idx] <- id_order[i]
    to_id[idx]   <- id_order[nb_i]
    pos <- pos + len
  }
  
  # Trim if any isolates caused fewer edges
  if (pos - 1L < n_edges) {
    from_id <- from_id[seq_len(pos - 1L)]
    to_id   <- to_id[seq_len(pos - 1L)]
  }
  
  data.table(id = from_id, neighbor_id = to_id)
}

cat("Building spatial edge list...\n")
edges <- build_edge_list(id_order, rook_neighbors_unique)
cat(sprintf("  Edge list: %d directed edges\n", nrow(edges)))

# ──────────────────────────────────────────────────────────────────────
# 2.  Compute neighbor stats for each variable via vectorized join
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# We need a lookup keyed on (id, year) for the variable values.
# We'll key cell_data on (id, year) for fast joins.
setkeyv(cell_data, c("id", "year"))

# Unique years for the cross-join with edges
all_years <- sort(unique(cell_data$year))

# Expand edges × years  (~1.37M × 28 ≈ 38.4M rows)
# To save memory, we do this once and reuse.
cat("Expanding edge list across years...\n")
edge_years <- edges[, .(year = all_years), by = .(id, neighbor_id)]
# edge_years columns: id, neighbor_id, year
# Set key on (neighbor_id, year) for joining neighbor values
setkeyv(edge_years, c("neighbor_id", "year"))

cat(sprintf("  Expanded edge-year rows: %s\n", format(nrow(edge_years), big.mark = ",")))

compute_and_add_neighbor_features_dt <- function(cell_data, edge_years, var_name) {
  cat(sprintf("  Processing variable: %s\n", var_name))
  
  # Extract just the columns we need for the join: (id, year, value)
  val_dt <- cell_data[, .(neighbor_id = id, year, val = get(var_name))]
  setkeyv(val_dt, c("neighbor_id", "year"))
  
  # Join neighbor values onto edge_years
  # edge_years keyed on (neighbor_id, year); val_dt keyed on (neighbor_id, year)
  ey <- edge_years[val_dt, on = .(neighbor_id, year), nomatch = 0L]
  # ey now has columns: id, neighbor_id, year, val
  # where val is the neighbor's value
  
  # Aggregate by (id, year) — these are the stats for each cell-year
  stats <- ey[!is.na(val),
              .(nb_max  = max(val),
                nb_min  = min(val),
                nb_mean = mean(val)),
              by = .(id, year)]
  
  # Name the new columns to match original convention
  max_col  <- paste0("nb_max_",  var_name)
  min_col  <- paste0("nb_min_",  var_name)
  mean_col <- paste0("nb_mean_", var_name)
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
  
  # Remove old columns if they exist (idempotent re-runs)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }
  
  # Merge back into cell_data
  setkeyv(stats, c("id", "year"))
  setkeyv(cell_data, c("id", "year"))
  cell_data[stats, on = .(id, year),
            c(max_col, min_col, mean_col) := mget(paste0("i.", c(max_col, min_col, mean_col)))]
  
  # Cell-years with no valid neighbors remain NA (already the default for unmatched rows)
  invisible(NULL)
}

cat("Computing neighbor statistics...\n")
for (var_name in neighbor_source_vars) {
  compute_and_add_neighbor_features_dt(cell_data, edge_years, var_name)
}
cat("Done.\n")

# ──────────────────────────────────────────────────────────────────────
# 3.  Clean up large temporaries
# ──────────────────────────────────────────────────────────────────────
rm(edges, edge_years)
gc()

# cell_data now contains the 15 new neighbor-stat columns
# (3 stats × 5 variables) with numerically identical values
# to the original implementation.
#
# The trained Random Forest model is untouched and can be used
# directly for prediction on the updated cell_data.
```

---

## Why This Is Numerically Identical

| Original | Optimized | Match? |
|---|---|---|
| For each cell-year row, find neighbor cell-year rows by string key, extract values, compute `max`/`min`/`mean` excluding `NA`. | For each `(id, year)` group in the joined edge table, compute `max`/`min`/`mean` on the same neighbor values excluding `NA`. | ✅ Identical sets, identical functions. |
| Cell-years with zero valid neighbors → `NA, NA, NA`. | Unmatched `(id, year)` keys after the join remain `NA` in `cell_data`. | ✅ Identical. |

The `max`, `min`, and `mean` functions are deterministic and associative — grouping the same values produces the same results regardless of evaluation order.

---

## Performance Estimate

| Step | Original | Optimized |
|---|---|---|
| Build neighbor lookup | ~80+ hours (6.46M string ops) | ~1 sec (344K integer loop) |
| Expand edges × years | — | ~2 sec (38.4M row CJ) |
| Per-variable join + aggregate | ~1 hour each (6.46M lapply) | ~15–30 sec each (data.table keyed join + groupby) |
| **Total** | **86+ hours** | **~2–5 minutes** |