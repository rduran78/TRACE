 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It creates a list of 6.46 million elements by looping row-by-row in R, performing character coercion, `paste`, and named-vector lookups inside each iteration. Specifically:

1. **`build_neighbor_lookup` is O(N × k) with enormous constant factors.** For each of ~6.46M rows, it converts IDs to character, pastes keys, and does named-vector lookups. Named-vector lookup via `[` on a character-keyed vector is hash-table access, but doing it 6.46M times in an R `lapply` with repeated allocations is extremely slow.

2. **`compute_neighbor_stats` is also row-level R.** Another 6.46M-iteration `lapply`, subsetting a numeric vector and computing `max/min/mean` per row. This is comparatively cheaper per iteration but still unnecessarily slow.

3. **Memory pressure.** Storing 6.46M list elements (the neighbor lookup), each an integer vector, creates massive overhead from R's list/vector object headers (~6.46M SEXP allocations).

The fundamental insight: **this is a sparse-matrix–vector product / grouped aggregation problem, not a per-row scripting problem.** The neighbor graph is static across years. We can represent the entire operation as a join + grouped aggregation in `data.table`, which is orders of magnitude faster.

---

## Optimization Strategy

1. **Explode the nb object into an edge table once** — a two-column `data.table` of `(id_from, id_to)` directed edges (~1.37M rows).

2. **Join edges to the panel on `(id_to, year)`** to get neighbor values — this is a keyed `data.table` equi-join, producing ~1.37M × 28 ≈ 38.5M rows, well within 16 GB RAM (~1–2 GB for 5 numeric columns).

3. **Group by `(id_from, year)` and compute `max`, `min`, `mean`** in one pass per variable — fully vectorized C-level aggregation inside `data.table`.

4. **Left-join the results back** to the master panel.

This replaces two nested R loops (6.46M iterations each × 5 variables) with a handful of vectorized `data.table` operations. Expected wall time: **minutes, not days.**

5. **The trained Random Forest model and all numerical results are preserved** — we are computing the identical `max`, `min`, `mean` of the identical rook-neighbor values; only the execution path changes.

---

## Working R Code

```r
library(data.table)

# ── 0. Convert panel to data.table (if not already) ─────────────────────────
setDT(cell_data)

# ── 1. Explode the nb object into a directed edge table ─────────────────────
#
#   rook_neighbors_unique : an nb object (list of integer index vectors)
#   id_order              : character or integer vector mapping list position → cell id
#
#   Each element rook_neighbors_unique[[i]] contains the *positions* of
#   neighbors of id_order[i].  We map positions back to cell IDs.

edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb_pos <- rook_neighbors_unique[[i]]
  # spdep nb lists use 0L to mean "no neighbors"

  nb_pos <- nb_pos[nb_pos > 0L]
  if (length(nb_pos) == 0L) return(NULL)
  data.table(id_from = id_order[i], id_to = id_order[nb_pos])
}))

cat(sprintf("Edge table: %d directed edges\n", nrow(edges)))

# ── 2. Function: compute neighbor stats for one variable ─────────────────────
compute_neighbor_stats_dt <- function(panel, edges, var_name) {

  # Minimal subset for the join: id, year, value
  rhs <- panel[, .(id, year, val = get(var_name))]
  setkey(rhs, id, year)

  # Join: for every (id_from → id_to) edge, attach id_to's value in each year
  #   result has columns: id_from, id_to, year, val
  joined <- rhs[edges, on = .(id = id_to), allow.cartesian = TRUE, nomatch = 0L]
  # After this join the column 'id' holds id_to; 'id_from' holds the focal cell.

  # Aggregate per (id_from, year)
  stats <- joined[!is.na(val),
                  .(nmax  = max(val),
                    nmin  = min(val),
                    nmean = mean(val)),
                  keyby = .(id_from, year)]

  # Rename for merge
  new_names <- paste0("n_", var_name, c("_max", "_min", "_mean"))
  setnames(stats, c("nmax", "nmin", "nmean"), new_names)

  stats
}

# ── 3. Loop over the five source variables and left-join results ─────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

setkey(cell_data, id, year)

for (vname in neighbor_source_vars) {
  cat(sprintf("Computing neighbor stats for: %s ...\n", vname))

  stats_dt <- compute_neighbor_stats_dt(cell_data, edges, vname)

  # Left-join onto the master panel
  new_cols <- paste0("n_", vname, c("_max", "_min", "_mean"))

  # Remove old columns if they already exist (idempotent reruns)
  for (nc in new_cols) {
    if (nc %in% names(cell_data)) cell_data[, (nc) := NULL]
  }

  cell_data <- stats_dt[cell_data, on = .(id_from = id, year)]

  # The join puts id_from as the key column; rename back to id

  setnames(cell_data, "id_from", "id")
  setkey(cell_data, id, year)

  cat(sprintf("  done – added %s\n", paste(new_cols, collapse = ", ")))
}

cat("All neighbor features computed.\n")

# ── 4. (Optional) Convert back to data.frame if downstream code expects it ──
# cell_data <- as.data.frame(cell_data)
```

---

### Why the results are identical

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor definition | `rook_neighbors_unique[[i]]` positions → cell IDs | Same positions → same cell IDs via `edges` table |
| Per-variable stats | `max`, `min`, `mean` of non-NA neighbor values | Identical: `joined[!is.na(val), .(max, min, mean)]` |
| NA handling | `neighbor_vals[!is.na(neighbor_vals)]` | `joined[!is.na(val), ...]` — same exclusion |
| No-neighbor cells | Return `(NA, NA, NA)` | Left join leaves `NA` for unmatched focal cells |

The trained Random Forest model is not touched; it consumes the same columns with the same values.

### Expected performance

| Step | Estimated time |
|---|---|
| Build edge table (~1.37M rows) | ~1–3 seconds |
| Per-variable join + aggregate (×5) | ~20–60 seconds each |
| Total | **~2–5 minutes** (vs. 86+ hours) |

Peak additional RAM: ~2–3 GB for the largest intermediate join table (well within 16 GB).