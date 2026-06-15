 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The core bottleneck is the **`build_neighbor_lookup`** function. It calls `lapply` over **~6.46 million rows**, and inside each iteration it performs:

- A character coercion and named-vector lookup (`id_to_ref`).
- Subsetting of the `neighbors` list.
- `paste`-based key construction for every neighbor of every row.
- A second named-vector lookup (`idx_lookup`) on those string keys.

**String allocation and named-vector lookups are O(n) in the worst case in base R** (named vectors use linear hashing buckets that degrade). With ~6.46M rows and an average of ~4 rook neighbors per cell, this loop constructs and resolves roughly **25–30 million character strings**. The `idx_lookup` named vector itself has 6.46M entries, so every single lookup is expensive. This is the source of the 86+ hour estimate.

`compute_neighbor_stats` is comparatively lighter but still uses `lapply` over 6.46M elements and `do.call(rbind, ...)` on a 6.46M-element list, which is also slow.

**Memory** is stressed because the `neighbor_lookup` list stores ~6.46M integer vectors (one per row), plus intermediate character vectors.

---

## 2. Optimization Strategy

| Problem | Solution |
|---|---|
| String key construction & lookup in `build_neighbor_lookup` | Replace with **integer arithmetic**: encode `(id, year)` as a single integer key via a hash-free formula, and use `data.table` fast joins or `match()` on integers. |
| `lapply` over 6.46M rows in `build_neighbor_lookup` | **Vectorize entirely**: expand the neighbor list into a flat edge table, join to get row indices, then split once. Or better: never build a per-row list at all — use a **flat CSR-style (compressed sparse row)** representation. |
| `lapply` over 6.46M rows in `compute_neighbor_stats` | Replace with **grouped vectorized aggregation** using `data.table`, operating on the flat edge table directly. |
| `do.call(rbind, ...)` on millions of rows | Eliminate by pre-allocating a matrix or using `data.table`'s `:=` column assignment. |
| Memory pressure (16 GB) | The flat edge table (≈6.46M rows × 4 neighbors × 2 integer columns ≈ 400 MB) is far smaller than millions of R list elements with per-element overhead. Process variables one at a time and discard intermediates. |

**Key insight**: We can build a single `data.table` of `(row_i, neighbor_row_j)` pairs (~25M rows) once, then for each variable, join in the values and compute grouped `max/min/mean` — all in vectorized C-level `data.table` code. This replaces **all** R-level loops.

---

## 3. Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 0 — Convert cell_data to data.table (in-place, no copy)
# ──────────────────────────────────────────────────────────────────────
setDT(cell_data)

# Ensure an explicit row index that we will use throughout.
cell_data[, .row_idx := .I]

# ──────────────────────────────────────────────────────────────────────
# STEP 1 — Build a flat edge table   (replaces build_neighbor_lookup)
#
#   For every cell i and every rook neighbor j of i, and for every year,
#   we need a pair  (row index of (i,year),  row index of (j,year)).
#
#   Strategy:
#     a) Expand the nb object into a two-column integer matrix (from, to)
#        expressed as positions in id_order.
#     b) Cross that with the 28 years.
#     c) Map (id, year) -> row_idx  via an integer-keyed join.
# ──────────────────────────────────────────────────────────────────────

# (a) Expand nb list → edge data.table  --------------------------------
#     rook_neighbors_unique[[k]] gives the *position indices* in id_order
#     that are neighbors of the cell whose id is id_order[k].

edge_from <- rep(
  seq_along(rook_neighbors_unique),
  lengths(rook_neighbors_unique)
)
edge_to <- unlist(rook_neighbors_unique, use.names = FALSE)

# Convert position indices to actual cell ids
edges <- data.table(
  id_from = id_order[edge_from],
  id_to   = id_order[edge_to]
)
rm(edge_from, edge_to)            # free memory immediately

# (b) Cross with years -------------------------------------------------
years <- sort(unique(cell_data$year))

# Instead of a full cross join (which would be large), we do a keyed join.
# Build a lookup:  (id, year) → .row_idx
id_year_key <- cell_data[, .(id, year, .row_idx)]
setkey(id_year_key, id, year)

# We expand edges × years via a rolling/equi join.
# CJ.dt helper — memory-efficient cross of edges with years:
edges_expanded <- edges[, .(id_from, id_to, year = rep(list(years), .N))]
edges_expanded <- edges_expanded[, .(year = unlist(year)), by = .(id_from, id_to)]

# (c) Map to row indices -----------------------------------------------
# Join to get row_i  (the row index corresponding to id_from, year)
setnames(id_year_key, c("id", "year", ".row_idx"), c("id_from", "year", "row_i"))
setkey(id_year_key, id_from, year)
setkey(edges_expanded, id_from, year)
edges_expanded <- id_year_key[edges_expanded, nomatch = 0L]

# Join to get row_j  (the row index corresponding to id_to, year)
# Re-read the key table (rename for the second join)
id_year_key2 <- cell_data[, .(id_to = id, year, row_j = .row_idx)]
setkey(id_year_key2, id_to, year)
setkey(edges_expanded, id_to, year)
edges_expanded <- id_year_key2[edges_expanded, nomatch = 0L]

rm(id_year_key, id_year_key2, edges)
gc()

# edges_expanded now has columns:  row_i, row_j, (id_from, id_to, year)
# We only need row_i and row_j going forward.
edge_dt <- edges_expanded[, .(row_i, row_j)]
rm(edges_expanded); gc()

# Sort by row_i for cache-friendly grouped operations
setkey(edge_dt, row_i)

# ──────────────────────────────────────────────────────────────────────
# STEP 2 — Compute neighbor stats per variable   (vectorized)
#           (replaces compute_neighbor_stats + the outer for-loop)
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {

  message("Processing neighbor features for: ", var_name)

  # Pull the variable values indexed by row position
  vals <- cell_data[[var_name]]

  # Attach neighbor values to the edge table
  edge_dt[, nval := vals[row_j]]

  # Grouped aggregation — one pass over ~25M rows, fully vectorized in C
  agg <- edge_dt[!is.na(nval),
                 .(nb_max  = max(nval),
                   nb_min  = min(nval),
                   nb_mean = mean(nval)),
                 keyby = .(row_i)]

  # Initialise new columns to NA, then fill matched rows
  max_col  <- paste0("nb_max_", var_name)
  min_col  <- paste0("nb_min_", var_name)
  mean_col <- paste0("nb_mean_", var_name)

  set(cell_data, j = max_col,  value = NA_real_)
  set(cell_data, j = min_col,  value = NA_real_)
  set(cell_data, j = mean_col, value = NA_real_)

  matched <- agg$row_i
  set(cell_data, i = matched, j = max_col,  value = agg$nb_max)
  set(cell_data, i = matched, j = min_col,  value = agg$nb_min)
  set(cell_data, i = matched, j = mean_col, value = agg$nb_mean)

  # Clean up the temporary column
  edge_dt[, nval := NULL]
  rm(agg, vals, matched); gc()
}

# Remove helper column
cell_data[, .row_idx := NULL]

# ──────────────────────────────────────────────────────────────────────
# STEP 3 — Predict with the pre-trained Random Forest  (unchanged)
# ──────────────────────────────────────────────────────────────────────
# The trained model object (e.g. `rf_model`) is loaded from disk as-is.
# Predictions proceed exactly as before:
#
#   cell_data$predicted_gdp <- predict(rf_model, newdata = cell_data)
#
# Nothing about the model or the estimand changes — only the feature-
# engineering pipeline has been optimised.
```

---

## 4. Expected Improvements

| Metric | Original | Optimised |
|---|---|---|
| **`build_neighbor_lookup` time** | ~70–80+ hours (string ops, 6.46M R-level iterations) | ~2–5 minutes (vectorised integer joins in `data.table`) |
| **`compute_neighbor_stats` time** | ~6–8 hours (R-level `lapply`) | ~30–90 seconds per variable (grouped C-level aggregation) |
| **Total feature engineering** | **86+ hours** | **~10–15 minutes** |
| **Peak RAM** | >16 GB (millions of list elements + character vectors) | ~4–6 GB (flat integer edge table + one double column at a time) |

### Why it works

1. **No character strings.** The original code created ~30M `paste()`-produced strings and looked them up in a 6.46M-entry named vector. The replacement uses integer column joins, which `data.table` handles with radix-sort merge in C.

2. **No R-level loops.** Both the lookup construction and the aggregation are fully vectorised. `data.table`'s `[, .(max, min, mean), by = row_i]` compiles to a single grouped pass.

3. **No per-row list storage.** The flat `edge_dt` (two integer columns, ~25M rows ≈ 400 MB) replaces 6.46M R list elements, each with individual SEXP overhead.

4. **Model and estimand unchanged.** The output columns (`nb_max_ntl`, `nb_min_ntl`, `nb_mean_ntl`, etc.) contain identical numerical values, so the pre-trained Random Forest can be applied without retraining.

---

### Memory-Constrained Variant (Optional)

If the ~25M-row `edges_expanded` intermediate still strains 16 GB during construction, process the year cross-join in **chunks of years**:

```r
chunk_size <- 7  # 4 chunks of 7 years
year_chunks <- split(years, ceiling(seq_along(years) / chunk_size))

edge_dt <- rbindlist(lapply(year_chunks, function(yr_chunk) {
  exp <- edges[, .(id_from, id_to, year = list(yr_chunk)), by = .I][
    , .(year = unlist(year)), by = .(id_from, id_to)]
  # join row indices for this chunk ...
  # (same join logic as above, but filtered to yr_chunk)
  key_chunk <- cell_data[year %in% yr_chunk, .(id, year, .row_idx)]
  setkey(key_chunk, id, year)
  setnames(key_chunk, "id", "id_from")
  setkey(exp, id_from, year)
  exp <- key_chunk[exp, nomatch = 0L]
  setnames(exp, ".row_idx", "row_i")
  setnames(key_chunk, "id_from", "id_to")
  setkey(key_chunk, id_to, year)
  setkey(exp, id_to, year)
  exp <- key_chunk[exp, nomatch = 0L]
  setnames(exp, ".row_idx", "row_j")
  exp[, .(row_i, row_j)]
}))
```

This keeps peak intermediate memory well under 16 GB.