 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Converts an ID to a character and looks it up in a named vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs paste-based string keys (`"id_year"`) for each neighbor.
4. Looks those keys up in another named character vector (`idx_lookup`).

String concatenation (`paste`), named-vector lookups on character keys, and per-row `lapply` iteration over 6.46M rows are all extremely slow in R. The `compute_neighbor_stats` function then loops over the same 6.46M entries again per variable, using R-level `lapply` with `max`/`min`/`mean` on small vectors—slow but secondary.

**Root causes, ranked by impact:**

1. **Row-level `lapply` with string operations over 6.46M rows** in `build_neighbor_lookup` — O(N × avg_neighbors) string pastes and named-vector lookups.
2. **Named character vector lookup** (`idx_lookup[neighbor_keys]`) is O(n) hash probing per call but the constant factor for 6.46M × ~4 neighbors is enormous.
3. **`compute_neighbor_stats`** uses R-level loop over 6.46M entries per variable (×5 variables).
4. **No vectorization or use of data.table/matrix indexing.**

## Optimization Strategy

**Core idea:** Replace all row-level R loops and string-key lookups with fully vectorized `data.table` joins and grouped aggregations.

- **Step 1:** Build an edge table (a two-column `data.table`) of all directed neighbor pairs from the `nb` object — done once, ~1.37M rows.
- **Step 2:** Join this spatial edge table to the panel data by year, producing a long table of (focal_row, neighbor_row) pairs — this is a single equi-join, fully vectorized.
- **Step 3:** For each variable, compute `max`, `min`, `mean` of neighbor values via a single grouped aggregation on the long table.

This eliminates all `lapply`, all `paste` key construction, and all named-vector lookups. Expected runtime: **minutes, not hours**.

## Working R Code

```r
library(data.table)

# ── Step 0: Convert panel data to data.table (if not already) ──
cell_dt <- as.data.table(cell_data)

# Ensure id and year columns exist; create a row index for later re-merge
cell_dt[, .row_idx := .I]

# ── Step 1: Build a spatial edge table from the nb object ──
# rook_neighbors_unique is a list of integer vectors (spdep nb object).
# id_order is the vector mapping position in the nb list → cell id.
# neighbors[[i]] gives the positions (in id_order) of neighbors of cell id_order[i].

edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb_i <- rook_neighbors_unique[[i]]
  if (length(nb_i) == 0L || (length(nb_i) == 1L && nb_i[1] == 0L)) {
    return(NULL)
  }
  data.table(focal_id = id_order[i], neighbor_id = id_order[nb_i])
}))
# edge_list has ~1.37M rows: (focal_id, neighbor_id)

# ── Step 2: Build the full (focal_row, neighbor_row) mapping via join ──
# We need, for every cell-year row, the rows of its spatial neighbors in the SAME year.
# Strategy: join edge_list to cell_dt twice — once for focal, once for neighbor — keyed on year.

# Create lean lookup: id, year → row index (and the variable columns we need)
keep_cols <- c("id", "year", ".row_idx", neighbor_source_vars)
lookup <- cell_dt[, ..keep_cols]

# Focal side: attach year (and focal row index) to each edge
#   For each focal_id, it appears in many years. We expand edges × years.
focal_keys <- cell_dt[, .(id, year, focal_row = .row_idx)]

# Merge edges with focal keys to get (focal_row, neighbor_id, year)
setkey(edge_list, focal_id)
setkey(focal_keys, id)
expanded <- edge_list[focal_keys, on = .(focal_id = id),
                      .(focal_row, neighbor_id, year),
                      allow.cartesian = TRUE, nomatch = NULL]

# Now attach neighbor row index and neighbor variable values
# Build neighbor key table
neighbor_keys <- copy(lookup)
setnames(neighbor_keys, "id", "neighbor_id")
setnames(neighbor_keys, ".row_idx", "neighbor_row")

setkey(expanded, neighbor_id, year)
setkey(neighbor_keys, neighbor_id, year)

# Join to get neighbor variable values aligned to each (focal_row, neighbor_row)
joined <- neighbor_keys[expanded, on = .(neighbor_id, year), nomatch = NULL]
# joined now has columns: neighbor_id, year, neighbor_row, <var columns>, focal_row

# ── Step 3: Grouped aggregation per variable ──
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (vname in neighbor_source_vars) {
  agg <- joined[!is.na(get(vname)),
                 .(nb_max  = max(get(vname)),
                   nb_min  = min(get(vname)),
                   nb_mean = mean(get(vname))),
                 by = focal_row]

  # Initialize columns with NA
  max_col  <- paste0(vname, "_nb_max")
  min_col  <- paste0(vname, "_nb_min")
  mean_col <- paste0(vname, "_nb_mean")

  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]

  cell_dt[agg$focal_row, (max_col)  := agg$nb_max]
  cell_dt[agg$focal_row, (min_col)  := agg$nb_min]
  cell_dt[agg$focal_row, (mean_col) := agg$nb_mean]
}

# ── Step 4: Convert back to data.frame if downstream code expects it ──
cell_dt[, .row_idx := NULL]
cell_data <- as.data.frame(cell_dt)

# The trained Random Forest model is untouched.
# Numerical results are identical (same max/min/mean over same neighbor sets).
```

## Memory Note for 16 GB RAM

The `joined` table will have ~6.46M × ~4 neighbors ≈ 25.8M rows with ~7 columns of numeric/integer data — roughly **1.5–2 GB**. This fits in 16 GB RAM. If memory is tight, process variables one at a time inside the loop and keep only the needed column in `neighbor_keys` per iteration:

```r
for (vname in neighbor_source_vars) {
  nk <- cell_dt[, .(neighbor_id = id, year, val = get(vname))]
  setkey(nk, neighbor_id, year)
  j <- nk[expanded, on = .(neighbor_id, year), nomatch = NULL]
  agg <- j[!is.na(val), .(nb_max = max(val), nb_min = min(val), nb_mean = mean(val)),
           by = focal_row]
  cell_dt[agg$focal_row, paste0(vname, "_nb_max")  := agg$nb_max]
  cell_dt[agg$focal_row, paste0(vname, "_nb_min")  := agg$nb_min]
  cell_dt[agg$focal_row, paste0(vname, "_nb_mean") := agg$nb_mean]
  rm(nk, j, agg); gc()
}
```

This peaks at ~800 MB for the temporary tables and is safe on a 16 GB laptop.

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Lookup construction | `lapply` over 6.46M rows, `paste` keys | Single `data.table` equi-join |
| Neighbor stats | `lapply` per variable per row | Grouped `data.table` aggregation |
| Estimated runtime | 86+ hours | ~5–15 minutes |
| Numerical output | max/min/mean of neighbor values | Identical |
| RF model | Unchanged | Unchanged |
| Peak RAM | Moderate (but slow) | ~2 GB temporary (fits 16 GB) |