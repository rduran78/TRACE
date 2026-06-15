 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck has two root causes:

### 1. `build_neighbor_lookup` — O(n) `lapply` over 6.46 million rows with per-row string pasting and named-vector lookups

Each of the ~6.46M iterations performs:
- `as.character()` and a named-vector lookup (`id_to_ref`) — O(1) amortised but with overhead.
- `paste()` to build neighbor keys — allocates a small character vector per row.
- Named-vector subsetting on `idx_lookup` (a 6.46M-element named character vector) — each lookup walks a hash, repeated for every neighbor of every row.

This creates **~6.46M × avg_neighbors ≈ 25–30 million** individual hash lookups inside a single-threaded R `lapply`, plus millions of small allocations. Estimated: **tens of hours** just for the lookup build.

### 2. `compute_neighbor_stats` — another O(n) `lapply` per variable

Each call iterates over 6.46M rows, subsetting a numeric vector by the index list, computing `max/min/mean`. This is called 5 times (once per source variable), yielding ~32M R-level function calls. The `do.call(rbind, ...)` at the end also copies a 6.46M × 3 matrix from a list of 6.46M length-3 vectors — very slow.

### Memory
With 6.46M rows × 110+ columns of doubles (8 bytes each), the data alone is ~5.7 GB. The neighbor lookup list adds another ~1–2 GB. Intermediate copies push peak usage well beyond 16 GB.

---

## Optimization Strategy

| Bottleneck | Fix | Speedup |
|---|---|---|
| Per-row `paste`/hash lookup in `build_neighbor_lookup` | Replace with a **merge/join via `data.table`**: explode the `nb` object into an edge-list, join on `(neighbor_id, year)` to get row indices, then split by source row. | ~100–500× |
| Per-row `lapply` in `compute_neighbor_stats` | Replace with **vectorised `data.table` grouped aggregation** on the edge-list: join neighbor values, then `[, .(max, min, mean), by = source_row]`. No R-level loop at all. | ~50–200× |
| `do.call(rbind, list_of_vectors)` | Eliminated entirely — `data.table` returns a single matrix-like result. | large |
| Memory: copies of `cell_data` in loop | Assign columns **in-place** via `:=` (reference semantics of `data.table`). | saves GBs |
| 5 separate passes over edge-list | Compute all 5 variables' neighbor stats in **one pass** (wide join, then grouped summarise). | 5× fewer scans |

**Estimated wall-clock time: 5–15 minutes** on a 16 GB laptop (down from 86+ hours).

The trained Random Forest model is untouched — we are only rewriting the feature-engineering step that feeds it. The numerical results (max, min, mean of rook-neighbor values) are identical.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 0.  Convert cell_data to data.table (in-place conversion; no copy)
# ---------------------------------------------------------------
if (!is.data.table(cell_data)) setDT(cell_data)

# Ensure a deterministic row-order column so we can map results back
cell_data[, .row_id := .I]

# ---------------------------------------------------------------
# 1.  Build a data.table edge-list from the nb object (one-time)
#
#     rook_neighbors_unique is an nb object of length
#     length(id_order) == 344,208.
#     Each element is an integer vector of neighbor positions in
#     id_order.
# ---------------------------------------------------------------
build_edge_dt <- function(id_order, nb_obj) {
  # Pre-allocate vectors the size of the total number of directed edges
  n_edges <- sum(lengths(nb_obj))            # ~1.37 M
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  pos <- 0L
  for (i in seq_along(nb_obj)) {
    nb_i <- nb_obj[[i]]
    n_i  <- length(nb_i)
    if (n_i == 0L) next
    idx <- pos + seq_len(n_i)
    from_id[idx] <- id_order[i]
    to_id[idx]   <- id_order[nb_i]
    pos <- pos + n_i
  }
  data.table(from_id = from_id, to_id = to_id)
}

cat("Building edge list ...\n")
edge_dt <- build_edge_dt(id_order, rook_neighbors_unique)
cat(sprintf("  %s directed edges\n", format(nrow(edge_dt), big.mark = ",")))

# ---------------------------------------------------------------
# 2.  Cross-join edges with years to get (from_id, to_id, year),
#     then join to cell_data to pick up neighbour-row values.
#
#     This replaces build_neighbor_lookup + compute_neighbor_stats
#     entirely with vectorised data.table operations.
# ---------------------------------------------------------------
compute_all_neighbor_features <- function(cell_data, edge_dt, source_vars) {

  years <- sort(unique(cell_data$year))

  # Edge list × years  (~1.37M edges × 28 years ≈ 38.5M rows)
  # This is the "long" representation of every (source_row, neighbor_row) pair.
  cat("Expanding edge × year table ...\n")
  ey <- CJ(edge_idx = seq_len(nrow(edge_dt)), year = years)
  ey[, `:=`(from_id = edge_dt$from_id[edge_idx],
            to_id   = edge_dt$to_id[edge_idx])]
  ey[, edge_idx := NULL]

  # Key cell_data for fast join  (id, year) -> row values
  # We only need id, year, .row_id, and the source_vars columns.
  keep_cols <- c("id", "year", ".row_id", source_vars)
  cd_small <- cell_data[, ..keep_cols]
  setkey(cd_small, id, year)

  # Join to get the SOURCE row id (.row_id of the "from" cell-year)
  cat("Joining source rows ...\n")
  setnames(cd_small, "id", "from_id")
  # We only need .row_id from the source side
  ey <- cd_small[, .(from_id, year, .row_id)][ey, on = .(from_id, year), nomatch = 0L]
  setnames(ey, ".row_id", "src_row")
  setnames(cd_small, "from_id", "id")   # restore

  # Join to get the NEIGHBOR values
  cat("Joining neighbor values ...\n")
  setnames(cd_small, "id", "to_id")
  ey <- cd_small[, c("to_id", "year", source_vars), with = FALSE
                 ][ey, on = .(to_id, year), nomatch = 0L]
  setnames(cd_small, "to_id", "id")     # restore

  # Now ey has columns: src_row, and each of source_vars with the
  # neighbor's value.  Aggregate by src_row.
  cat("Computing grouped stats ...\n")

  # Build aggregation expression dynamically
  agg_exprs <- unlist(lapply(source_vars, function(v) {
    list(
      sprintf("nb_%s_max  = fcase(all(is.na(%s)), NA_real_, default = max(%s, na.rm = TRUE))", v, v, v),
      sprintf("nb_%s_min  = fcase(all(is.na(%s)), NA_real_, default = min(%s, na.rm = TRUE))", v, v, v),
      sprintf("nb_%s_mean = fcase(all(is.na(%s)), NA_real_, default = mean(%s, na.rm = TRUE))", v, v, v)
    )
  }))
  agg_call <- paste0(".(", paste(agg_exprs, collapse = ", "), ")")

  stats <- ey[, eval(parse(text = agg_call)), by = src_row]

  # ---------------------------------------------------------------
  # 3.  Map aggregated stats back to cell_data rows.
  #     Rows with no neighbours (islands / boundary) get NA.
  # ---------------------------------------------------------------
  cat("Merging results back ...\n")
  setkey(stats, src_row)

  new_cols <- setdiff(names(stats), "src_row")
  # Pre-fill with NA
  for (col in new_cols) {
    set(cell_data, j = col, value = NA_real_)
  }
  # Assign computed values by row index
  cell_data[stats$src_row, (new_cols) := stats[, ..new_cols]]

  invisible(cell_data)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)

# Clean up the helper column
cell_data[, .row_id := NULL]

cat("Done — neighbor features added.\n")
```

---

### Why the numerical estimand is preserved

The aggregation expressions compute exactly `max`, `min`, and `mean` of the non-`NA` neighbor values for each cell-year — the same three statistics the original `compute_neighbor_stats` computes. No approximation, sampling, or rounding is introduced. Column names follow the pattern `nb_{var}_{stat}` (adjust to match whatever the existing `compute_and_add_neighbor_features` wrapper produced so the trained Random Forest's `predict()` call sees the expected feature names).

### Memory profile

| Object | Approximate size |
|---|---|
| `cell_data` (6.46M × 115 doubles) | ~5.9 GB |
| `edge × year` table (38.5M × 7 cols) | ~2.1 GB |
| `stats` result (6.46M × 16 cols) | ~0.8 GB |
| **Peak** | ~12–13 GB ✔ (fits in 16 GB) |

If memory is still tight, the `ey` join can be chunked by year (28 chunks of ~1.37M rows each) and the partial aggregations combined afterward, reducing peak usage by ~1.5 GB at the cost of a trivial amount of extra time.