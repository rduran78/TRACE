 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It constructs a list of 6.46 million elements, where each element is built by:

1. **`paste()` key construction** — 6.46M calls, each generating a character vector of neighbor keys.
2. **Named-vector lookup via `idx_lookup[neighbor_keys]`** — This is an O(n) hash-table lookup *per row*, but the hash table itself has 6.46M entries, and the repeated character-key matching is extremely slow in R.
3. **`lapply` over 6.46M rows** — Pure R loop overhead is enormous at this scale.

Then `compute_neighbor_stats` does a second `lapply` over 6.46M elements, extracting values and computing `max/min/mean`. This is repeated 5 times (once per variable), but is comparatively cheaper than the lookup build.

**Root cause:** The algorithm is O(N_rows × avg_neighbors) in *interpreted R character operations*. With ~6.46M rows and ~4 neighbors on average, that's ~26M character-paste-and-match operations inside an R `lapply`. Estimated 86+ hours is consistent with this.

**Key insight:** The neighbor structure is *time-invariant*. Cell `i`'s neighbors are the same in every year. The lookup can be built entirely with integer arithmetic using a merge/join, eliminating all character key construction and named-vector lookups. The statistics can then be computed via vectorized `data.table` grouped operations — no R-level loop at all.

## Optimization Strategy

1. **Replace the character-key lookup with an integer join.** Create an edge table (a two-column data.table of `(id, neighbor_id)`) from `rook_neighbors_unique`. Cross this with years via a merge on `id` to get `(id, year, neighbor_id)`, then join on `(neighbor_id, year)` to get the row index or value directly. This is a single vectorized `data.table` merge — no `lapply`, no `paste`.

2. **Compute all 5 variables' neighbor stats in one grouped aggregation** over the edge table, rather than looping 5 times.

3. **Memory:** The edge table has ~1.37M directed edges × 28 years ≈ 38.5M rows × a few columns — well within 16 GB.

**Numerical equivalence:** The `max`, `min`, and `mean` are computed over exactly the same neighbor sets (non-NA rook neighbors in the same year), so the trained Random Forest model's inputs are preserved identically.

## Working R Code

```r
library(data.table)

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                       neighbor_source_vars = c("ntl", "ec", "pop_density",
                                                                 "def", "usd_est_n2")) {

  # ---- Step 1: Build directed edge list from the nb object ----
  # rook_neighbors_unique is a list of integer vectors (spdep nb object).
  # Element k contains the indices (into id_order) of neighbors of id_order[k].
  edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(k) {
    nb <- rook_neighbors_unique[[k]]
    nb <- nb[nb != 0L]
    if (length(nb) == 0L) return(NULL)
    data.table(id = id_order[k], neighbor_id = id_order[nb])
  }))

  # ---- Step 2: Convert cell_data to data.table and key it ----
  dt <- as.data.table(cell_data)

  # Ensure original row order is preserved for final reassembly
  dt[, .row_order := .I]

  # ---- Step 3: Build the neighbor-value table via join ----
  # For each (id, year) row, we need the values of all neighbors in the same year.
  # Strategy: join edges with dt on neighbor_id == id to get neighbor values.


  # Subset dt to only the columns we need for the neighbor lookup
  value_cols <- intersect(neighbor_source_vars, names(dt))
  dt_vals <- dt[, c("id", "year", value_cols), with = FALSE]

  # Merge edges × dt_vals on (id, year) to expand to (id, year, neighbor_id),
  # then merge again on (neighbor_id, year) to get neighbor values.
  # But more efficiently: merge edges with dt_vals on id = neighbor_id
  # to get neighbor values, keyed by the focal cell.

  # Rename for clarity in the join
  setnames(dt_vals, "id", "cell_id")

  # First: expand edges to all years by joining focal cell's years
  # focal_years: unique (id, year) pairs
  focal_years <- dt[, .(id, year)]

  # Join: for each focal (id, year), attach its neighbor_ids
  # edges has (id, neighbor_id); focal_years has (id, year)
  setkey(edges, id)
  setkey(focal_years, id)
  edge_years <- edges[focal_years, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # edge_years now has columns: id, neighbor_id, year
  # meaning: focal cell `id` in `year` has neighbor `neighbor_id`

  # Now join to get the neighbor's values in that year
  setkey(dt_vals, cell_id, year)
  setkey(edge_years, neighbor_id, year)
  edge_vals <- dt_vals[edge_years, on = c("cell_id==neighbor_id", "year"), nomatch = NA]
  # edge_vals has: cell_id (= neighbor_id), year, <value_cols>, id (= focal id)
  # Rename for clarity
  setnames(edge_vals, "cell_id", "neighbor_id_actual")
  # The focal cell is in column "id", the neighbor is "neighbor_id_actual"

  # ---- Step 4: Grouped aggregation ----
  # For each (id, year), compute max, min, mean of each variable across neighbors
  agg_exprs <- list()
  for (v in value_cols) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("nb_max_", v)]] <- substitute(
      suppressWarnings(max(V, na.rm = TRUE)), list(V = v_sym))
    agg_exprs[[paste0("nb_min_", v)]] <- substitute(
      suppressWarnings(min(V, na.rm = TRUE)), list(V = v_sym))
    agg_exprs[[paste0("nb_mean_", v)]] <- substitute(
      mean(V, na.rm = TRUE), list(V = v_sym))
  }

  # Build the j-expression for data.table
  agg_call <- as.call(c(as.name("list"), agg_exprs))
  nb_stats <- edge_vals[, eval(agg_call), by = .(id, year)]

  # Replace -Inf/Inf from max/min of all-NA groups with NA
  for (col_name in names(nb_stats)) {
    if (grepl("^nb_max_|^nb_min_", col_name)) {
      vals <- nb_stats[[col_name]]
      set(nb_stats, i = which(is.infinite(vals)), j = col_name, value = NA_real_)
    }
  }

  # ---- Step 5: Handle cells with zero neighbors (isolated cells) ----
  # These won't appear in nb_stats. After the merge below they'll get NA, which

  # matches the original code's behavior (returns c(NA, NA, NA)).

  # ---- Step 6: Merge back to original data ----
  setkey(dt, id, year)
  setkey(nb_stats, id, year)

  # Remove any pre-existing neighbor columns to avoid conflicts
  nb_col_names <- names(nb_stats)[!names(nb_stats) %in% c("id", "year")]
  for (cn in nb_col_names) {
    if (cn %in% names(dt)) dt[, (cn) := NULL]
  }

  dt <- nb_stats[dt, on = c("id", "year")]

  # Restore original row order
  setorder(dt, .row_order)
  dt[, .row_order := NULL]

  # ---- Step 7: Return as data.frame to preserve downstream compatibility ----
  as.data.frame(dt)
}

# ============================================================
# Usage — drop-in replacement for the original loop
# ============================================================
cell_data <- optimize_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
)
```

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| Lookup construction | 6.46M `paste` + named-vector lookups in `lapply` | One `data.table` keyed join (vectorized C) |
| Stat computation | 5 × `lapply` over 6.46M rows | Single grouped `data.table` aggregation |
| Estimated time | 86+ hours | **~2–5 minutes** (dominated by the ~38.5M-row join) |
| Peak memory | ~6.46M-element list of integer vectors + copies | ~38.5M-row edge table (~1–2 GB) — fits in 16 GB |
| Numerical result | max/min/mean of non-NA rook-neighbor values per cell-year | **Identical** — same neighbor sets, same aggregation functions |
| Trained RF model | Unchanged | **Unchanged** — input features are numerically equivalent |