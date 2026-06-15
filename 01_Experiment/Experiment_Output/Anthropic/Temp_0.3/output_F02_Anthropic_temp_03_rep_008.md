 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting and named-vector lookups. Named vector lookups in R via character keys are O(n) hash lookups, but doing ~6.46M of them — each touching multiple neighbors — creates billions of small string allocations and hash probes. The `compute_neighbor_stats` function then iterates over the same 6.46M rows again per variable, and this is repeated for 5 variables.

**Specific problems:**

1. **String-key lookups at scale:** `paste(id, year)` keys and `idx_lookup[neighbor_keys]` create millions of temporary character vectors and perform hash-table lookups one row at a time. This is extremely slow in base R.
2. **`lapply` over 6.46M rows:** Each iteration has R-level overhead (function call, allocation, GC pressure). Returning a list of 6.46M integer vectors is also memory-heavy.
3. **Redundant work across years:** The neighbor *structure* is purely spatial (rook contiguity) and identical for every year. Yet the lookup rebuilds neighbor indices per cell-year row, effectively repeating the same spatial logic 28 times.
4. **`do.call(rbind, result)` on a 6.46M-element list:** This is a known anti-pattern — it is O(n²) in memory copies.
5. **Serial, single-threaded execution** for all of the above.

---

## Optimization Strategy

| Principle | Action |
|---|---|
| **Eliminate string hashing** | Replace character-key lookups with integer-indexed joins via `data.table`. |
| **Vectorize the neighbor join** | Explode the `nb` object into an edge-list `data.table` once, then use keyed `data.table` joins (binary search, C-level) instead of per-row `lapply`. |
| **Exploit year-invariance** | Build the spatial edge list once (344K cells × ~4 neighbors each ≈ 1.37M edges). Join to panel data by `(neighbor_id, year)` — the join engine handles the 28-year fan-out. |
| **Vectorize aggregation** | Use `data.table`'s `by=` grouped aggregation (`max`, `min`, `mean`) instead of `lapply` + `rbind`. |
| **Minimize memory** | Operate column-by-column; never materialize a 6.46M-element list of vectors. |
| **Preserve the trained RF model** | Only the feature columns are being computed; the model object is untouched. |
| **Preserve the numerical estimand** | `max`, `min`, `mean` over the same non-NA neighbor values — identical results. |

**Expected improvement:** From ~86+ hours down to **minutes** (the dominant operation becomes a keyed `data.table` join on ~38M rows, which is highly optimized in C).

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 1.  Build the spatial edge list ONCE  (small: ~1.37M rows)
#     rook_neighbors_unique is an nb object (list of integer vectors).
#     id_order is the vector of cell IDs in the same order as the nb list.
# ──────────────────────────────────────────────────────────────────────
build_edge_list <- function(id_order, neighbors) {
  # neighbors[[i]] contains integer indices into id_order for the
  # neighbors of cell id_order[i].  Index 0 means no neighbors (spdep convention).
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  # Remove the spdep "no-neighbor" sentinel (0)
  valid    <- to_idx != 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    id       = id_order[from_idx],   # focal cell
    nb_id    = id_order[to_idx]      # neighbor cell
  )
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
# edge_dt has columns: id, nb_id   (~1.37M rows)

# ──────────────────────────────────────────────────────────────────────
# 2.  Convert panel data to data.table (if not already)
# ──────────────────────────────────────────────────────────────────────
setDT(cell_data)

# ──────────────────────────────────────────────────────────────────────
# 3.  Compute neighbor features — one variable at a time
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_neighbor_features_dt <- function(cell_data, edge_dt, var_name) {
  # --- a) Slim table of neighbor values: (nb_id, year, value) ----------
  #     We only need the variable of interest to keep memory low.
  nb_vals <- cell_data[, .(nb_id = id, year, value = get(var_name))]
  setkey(nb_vals, nb_id, year)

  # --- b) Expand edges × years: join focal→neighbor, then look up value
  #     edge_dt has (id, nb_id).  We cross-join with the focal cell's year
  #     by merging edge_dt onto the focal cell's (id, year) rows,
  #     then look up the neighbor's value for that same year.

  # Focal rows: just id and year (small projection)
  focal <- cell_data[, .(id, year)]
  setkey(focal, id)

  # Merge focal rows with edge list → one row per (focal cell-year, neighbor)
  # This is the "fan-out" step: ~6.46M × ~4 ≈ ~26-38M rows
  edges_by_year <- edge_dt[focal, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # edges_by_year columns: id, nb_id, year

  # Look up the neighbor's value for the same year
  setkey(edges_by_year, nb_id, year)
  edges_by_year[nb_vals, value := i.value, on = .(nb_id, year)]

  # --- c) Aggregate: max, min, mean per (id, year) --------------------
  stats <- edges_by_year[
    !is.na(value),
    .(
      nb_max  = max(value),
      nb_min  = min(value),
      nb_mean = mean(value)
    ),
    by = .(id, year)
  ]

  # Name the new columns to match the original pipeline's convention
  new_names <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), new_names)

  # --- d) Left-join back onto cell_data --------------------------------
  #     Cells with no valid neighbors get NA (preserving original behavior).
  cell_data[stats, (new_names) := mget(paste0("i.", new_names)),
            on = .(id, year)]

  invisible(cell_data)
}

# ──────────────────────────────────────────────────────────────────────
# 4.  Run for each variable
# ──────────────────────────────────────────────────────────────────────
for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  compute_neighbor_features_dt(cell_data, edge_dt, var_name)
}

# cell_data now contains the 15 new columns (3 stats × 5 variables)
# and is ready for prediction with the existing trained Random Forest.
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same neighbor set** | `build_edge_list` reads the same `rook_neighbors_unique` nb object and `id_order`; the directed edges are identical. |
| **Same aggregation** | `max`, `min`, `mean` over non-NA neighbor values, grouped by `(id, year)` — mathematically identical to the original `lapply`. |
| **NA handling** | Rows with no valid neighbors receive `NA` for all three stats (via the left-join; unmatched rows stay `NA`). |
| **Trained RF model** | Untouched. Only feature columns are computed. Column names follow the same convention so `predict()` works as before. |

## Expected Performance

- **Edge list build:** < 1 second (vectorized `unlist` + indexing).
- **Per-variable join + aggregation:** ~30–90 seconds each (keyed `data.table` join on ~38M rows, C-level grouped aggregation).
- **Total for 5 variables:** **~3–8 minutes** on a 16 GB laptop, versus the original ~86+ hours.
- **Peak RAM:** The largest intermediate (`edges_by_year`) is ~38M rows × 4 columns of integers/doubles ≈ ~1.2 GB, well within 16 GB.