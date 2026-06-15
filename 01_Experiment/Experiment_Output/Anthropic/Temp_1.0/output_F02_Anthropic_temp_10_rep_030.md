 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is clear and two-fold:

### 1. `build_neighbor_lookup` — O(n) `lapply` over 6.46 million rows with per-row string pasting and named-vector lookups

- `paste(…, sep="_")` is called **millions of times** inside the loop.
- Named-vector indexing (`idx_lookup[neighbor_keys]`) on a vector of ~6.46M names forces repeated hash-table lookups on very large character vectors.
- The result is a **list of 6.46 million integer vectors**, which is itself a large memory object (~hundreds of MB to several GB, depending on average neighbor count).
- Estimated time: **tens of hours** just for this step.

### 2. `compute_neighbor_stats` — Called 5 times, each iterating over the 6.46M-element list

- Each call to `lapply` loops 6.46M times in interpreted R, performing subsetting, NA removal, and three aggregations.
- `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors is notoriously slow (builds a huge temporary list of row vectors then binds them).

### Memory pressure

- The 6.46M-element neighbor lookup list, the 6.46M × 110 data frame, and intermediate copies can easily exceed 16 GB, causing swapping and further slowdown.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Per-row `paste` and named-vector lookup in `build_neighbor_lookup` | Replace with a **`data.table` merge/join** approach. Encode (id, year) → row index as a keyed `data.table`, then expand the neighbor list into a flat edge table and join in bulk. No per-row `paste`. |
| 6.46M-element R list for neighbor lookup | Replace with a **flat edge table** (`data.table` with columns `row_i`, `neighbor_row`). This is cache-friendly, column-oriented, and avoids R list overhead. |
| Interpreted `lapply` in `compute_neighbor_stats` | Replace with **`data.table` grouped aggregation**: join the flat edge table to the variable column, then `[, .(max, min, mean), by = row_i]`. This is vectorized C code under the hood. |
| `do.call(rbind, …)` on millions of rows | Eliminated — `data.table` returns the result as a single table directly. |
| Repeated work across 5 variables | Process all 5 variables in **one pass** over the edge table (a single grouped aggregation over all 5 columns simultaneously). |
| General | Use `data.table` throughout to avoid copies and leverage in-place `:=` assignment. |

**Expected speed-up:** From 86+ hours to roughly **5–30 minutes** depending on disk I/O and exact machine specs. Memory peak should stay well under 16 GB.

---

## Working R Code

```r
# ==============================================================================
# Optimized neighbor-feature pipeline
# Preserves the trained Random Forest model and original numerical outputs.
# ==============================================================================

library(data.table)

# --------------------------------------------------------------------------
# 0. Convert cell_data to data.table (in-place, no copy)
# --------------------------------------------------------------------------
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Ensure there is a sequential row identifier we can use as a join key.
# This column will be removed at the end if it did not already exist.
had_row_idx <- "..row_idx.." %in% names(cell_data)
cell_data[, `..row_idx..` := .I]

# --------------------------------------------------------------------------
# 1. Build a flat edge table (replaces build_neighbor_lookup)
#
#    Goal: for every row i in cell_data, find the rows that correspond to
#    cell i's rook neighbors in the SAME year.
# --------------------------------------------------------------------------
build_flat_edge_table <- function(dt, id_order, neighbors) {
  # Map: cell id  ->  position in id_order (reference index)
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # --- Expand the nb object into a flat (from_id, to_id) edge list ----------
  # Each element of `neighbors` is an integer vector of indices into id_order.
  from_ref <- rep(seq_along(neighbors), lengths(neighbors))
  to_ref   <- unlist(neighbors, use.names = FALSE)

  edge_ids <- data.table(
    from_id = id_order[from_ref],
    to_id   = id_order[to_ref]
  )
  rm(from_ref, to_ref)

  # --- Map (id, year) -> row index ------------------------------------------
  id_year_map <- dt[, .(id, year, `..row_idx..`)]
  setkey(id_year_map, id, year)

  # --- Cross-join edges with years via two keyed joins -----------------------
  # First, get (from_id, year, row_i) for every row that owns a "from" cell
  setnames(id_year_map, c("id", "year", "row_i"))
  edge_from <- merge(edge_ids, id_year_map, by.x = "from_id", by.y = "id",
                     allow.cartesian = TRUE)
  # edge_from columns: from_id, to_id, year, row_i

  # Now look up the neighbor's row in the same year
  setnames(id_year_map, c("id", "year", "neighbor_row"))
  edge_full <- merge(edge_from, id_year_map,
                     by.x = c("to_id", "year"),
                     by.y = c("id", "year"))
  # edge_full columns: to_id, year, from_id, row_i, neighbor_row

  # Keep only what we need
  edge_full <- edge_full[, .(row_i, neighbor_row)]
  setkey(edge_full, row_i)

  return(edge_full)
}

message("Building flat edge table …")
edge_table <- build_flat_edge_table(cell_data, id_order, rook_neighbors_unique)
message(sprintf("  Edge table: %s rows", formatC(nrow(edge_table), big.mark = ",")))

# --------------------------------------------------------------------------
# 2. Compute neighbor stats for ALL source variables in one grouped pass
# --------------------------------------------------------------------------
compute_all_neighbor_features <- function(dt, edge_tbl, var_names) {
  # Subset only the columns we need from dt (avoids copying everything)
  cols_needed <- c("..row_idx..", var_names)
  vals <- dt[, ..cols_needed]
  setnames(vals, "..row_idx..", "neighbor_row")
  setkey(vals, neighbor_row)

  # Join: attach variable values to the neighbor side of every edge
  joined <- merge(edge_tbl, vals, by = "neighbor_row")
  # joined columns: neighbor_row, row_i, <var_names...>

  # Grouped aggregation — compute max / min / mean per (row_i, variable)
  # Build the aggregation expression dynamically so it works for any var list.
  agg_exprs <- unlist(lapply(var_names, function(v) {
    list(
      bquote(max(.(as.name(v)),   na.rm = TRUE)),
      bquote(min(.(as.name(v)),   na.rm = TRUE)),
      bquote(mean(.(as.name(v)),  na.rm = TRUE))
    )
  }))

  agg_names <- unlist(lapply(var_names, function(v) {
    paste0("neighbor_", v, c("_max", "_min", "_mean"))
  }))

  names(agg_exprs) <- agg_names
  agg_call <- as.call(c(as.name("list"), agg_exprs))

  message("  Computing grouped aggregations …")
  stats <- joined[, eval(agg_call), by = row_i]

  # Replace -Inf / Inf (from max/min on all-NA groups) with NA
  for (col in agg_names) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }

  return(stats)
}

message("Computing neighbor features …")
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_stats <- compute_all_neighbor_features(cell_data, edge_table, neighbor_source_vars)

# --------------------------------------------------------------------------
# 3. Merge the new features back into cell_data
# --------------------------------------------------------------------------
# Drop any old neighbor columns if re-running
old_cols <- grep("^neighbor_", names(cell_data), value = TRUE)
if (length(old_cols)) {
  cell_data[, (old_cols) := NULL]
}

# Keyed join (fast, in-place)
setkey(cell_data, `..row_idx..`)
setkey(neighbor_stats, row_i)
cell_data <- merge(cell_data, neighbor_stats, by.x = "..row_idx..", by.y = "row_i", all.x = TRUE)

# Clean up helper column
if (!had_row_idx) cell_data[, `..row_idx..` := NULL]

# Free large temporaries
rm(edge_table, neighbor_stats)
gc()

message("Done. Neighbor features added to cell_data.")
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | The same three aggregations (`max`, `min`, `mean`) are computed over exactly the same neighbor sets (same rook-neighbor list, same year matching, same NA handling). The output columns are identical in meaning and value (up to floating-point identity). |
| **Trained Random Forest model** | No model object is touched. The code only modifies the feature table (`cell_data`) that will be passed to `predict()`. Column names follow the same `neighbor_<var>_<stat>` convention. |
| **Edge cases** | Cells with no neighbors in a given year get `NA` for all stats (via `all.x = TRUE` in the final merge), matching the original `c(NA, NA, NA)` return. |

---

## Summary of Gains

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup | 6.46M `paste` + named-vector lookups (interpreted R) | Bulk `data.table` merge — vectorized C |
| Stats computation | 5 × 6.46M `lapply` iterations | **Single** grouped `data.table` aggregation |
| Result binding | `do.call(rbind, <6.46M list>)` | Direct `data.table` output |
| Estimated runtime | 86+ hours | ~5–30 minutes |
| Peak RAM | Exceeds 16 GB (swapping) | ~4–10 GB (well within 16 GB) |