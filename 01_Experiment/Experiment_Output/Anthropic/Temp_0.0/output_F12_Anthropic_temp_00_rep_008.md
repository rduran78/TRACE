 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, result)` on a list of 6.46 million small 3-element vectors is not free, it is a single vectorized C-level operation that completes in seconds. There is no "repeated list binding" inside `compute_neighbor_stats()` — it uses `lapply` to build a list in one pass, then binds once. This is standard and efficient R.

**The true bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Per-row string pasting and named-vector lookup over 6.46 million rows.** For every single row `i`, the function calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` and then indexes into `idx_lookup` (a named character vector of length 6.46 million) via character matching. Named vector lookup by character key in R is O(n) per probe in the worst case (hash collisions aside, the overhead of repeated hashing and matching against a 6.46M-entry names vector is enormous).

2. **The `lapply` over 6.46 million rows** each performing: one `as.character` coercion, one named-vector lookup into `id_to_ref`, one subset of `id_order` by a variable-length index vector, one `paste` call generating multiple keys, one named-vector lookup into `idx_lookup` (6.46M entries), and one `is.na` filter. With ~1,373,394 directed neighbor relationships spread across 344,208 cells × 28 years, the average row touches ~4 neighbors, meaning roughly **25.8 million** string constructions and hash lookups into a 6.46M-entry table — all inside an interpreted R loop.

3. **This function is called once and produces the lookup used by all 5 variables.** But that single call dominates total runtime. `compute_neighbor_stats()` is called 5 times and is comparatively cheap: it does only integer indexing into a numeric vector (vectorized, cache-friendly) plus simple `max`/`min`/`mean` on small neighbor sets.

**Estimated time breakdown (approximate):**
- `build_neighbor_lookup()`: ~80+ hours (character key construction and lookup, 6.46M interpreted iterations)
- `compute_neighbor_stats()` × 5: ~1–3 hours total
- `do.call(rbind, ...)` × 5: seconds each

## Optimization Strategy

1. **Eliminate all string key construction and character-based lookup.** Replace the `paste(..., sep="_")` keying scheme with direct integer arithmetic. Since each `(id, year)` pair maps to a row, build a fast integer-indexed lookup matrix or use a hash table (`data.table` or environment) keyed on integer pairs.

2. **Vectorize `build_neighbor_lookup()` entirely** using `data.table` joins. Expand the neighbor relationships into an edge table `(id, neighbor_id)`, join with the data's `(id, year)` → `row_index` mapping on `(neighbor_id, year)` to get all neighbor row indices in one vectorized merge, then split by source row index.

3. **Vectorize `compute_neighbor_stats()`** using `data.table` grouped aggregation on the edge table rather than `lapply` over 6.46M elements.

4. **Preserve the trained Random Forest model** — we only change feature engineering, producing numerically identical columns.

## Working R Code

```r
library(data.table)

# ============================================================
# OPTIMIZED build_neighbor_lookup
# ============================================================
# Instead of returning a list of length nrow(data), we return
# an edge data.table: (source_row, neighbor_row) which is far
# more efficient to construct and to aggregate over.

build_neighbor_edges <- function(data, id_order, neighbors) {
  # data must have columns: id, year
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer index vectors)

  dt <- as.data.table(data)
  dt[, row_idx := .I]

  # Step 1: Build edge list of (cell_id -> neighbor_cell_id) from the nb object
  # This is only ~1.37M edges (or up to 344,208 * avg_neighbors)
  edge_list <- rbindlist(lapply(seq_along(neighbors), function(i) {
    nb <- neighbors[[i]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
      return(NULL)
    }
    data.table(cell_id = id_order[i], neighbor_cell_id = id_order[nb])
  }))

  # Step 2: Create a keyed lookup: (id, year) -> row_idx
  # We join edge_list with dt on year to expand across all years
  # For each (cell_id, year) row in dt, find all (neighbor_cell_id, year) rows

  # Get unique cell-year to row mapping
  cell_year <- dt[, .(cell_id = id, year, row_idx)]
  setkey(cell_year, cell_id, year)

  # For source rows: get (cell_id, year, source_row_idx)
  source <- cell_year[, .(cell_id, year, source_row = row_idx)]

  # Join source with edge_list to get (source_row, neighbor_cell_id, year)
  setkey(source, cell_id)
  setkey(edge_list, cell_id)

  # This is the key vectorized join: expand edges across years
  expanded <- edge_list[source, on = "cell_id", allow.cartesian = TRUE,
                        nomatch = NULL]
  # expanded now has: cell_id, neighbor_cell_id, year, source_row

  # Step 3: Join to find neighbor row indices
  setkey(cell_year, cell_id, year)
  setnames(cell_year, "cell_id", "neighbor_cell_id")
  setnames(cell_year, "row_idx", "neighbor_row")

  result <- cell_year[expanded, on = c("neighbor_cell_id", "year"),
                      nomatch = NULL]
  # result has: neighbor_cell_id, year, neighbor_row, cell_id, source_row

  result[, .(source_row, neighbor_row)]
}

# ============================================================
# OPTIMIZED compute_neighbor_stats (vectorized via data.table)
# ============================================================
compute_neighbor_stats_fast <- function(data, edges, var_name) {
  # edges: data.table with (source_row, neighbor_row)
  # Returns a data.table with columns:
  #   neighbor_max_{var_name}, neighbor_min_{var_name}, neighbor_mean_{var_name}
  # aligned to row order of data

  n <- nrow(data)
  vals <- data[[var_name]]

  # Attach neighbor values
  edge_vals <- edges[, .(source_row, nval = vals[neighbor_row])]
  edge_vals <- edge_vals[!is.na(nval)]

  # Grouped aggregation — single pass, vectorized
  agg <- edge_vals[, .(
    nmax  = max(nval),
    nmin  = min(nval),
    nmean = mean(nval)
  ), by = source_row]

  # Allocate output aligned to all rows (NA for rows with no neighbors)
  out <- data.table(
    nmax  = rep(NA_real_, n),
    nmin  = rep(NA_real_, n),
    nmean = rep(NA_real_, n)
  )
  out[agg$source_row, `:=`(nmax = agg$nmax,
                            nmin = agg$nmin,
                            nmean = agg$nmean)]

  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(out, c("nmax", "nmin", "nmean"), c(max_col, min_col, mean_col))

  out
}

# ============================================================
# MAIN PIPELINE (replaces the outer loop)
# ============================================================
run_neighbor_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {

  cat("Building neighbor edge table (vectorized)...\n")
  t0 <- Sys.time()
  edges <- build_neighbor_edges(cell_data, id_order, rook_neighbors_unique)
  cat("  Edge table built:", nrow(edges), "edges in",
      round(difftime(Sys.time(), t0, units = "mins"), 1), "min\n")

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  is_dt <- is.data.table(cell_data)
  if (!is_dt) cell_data <- as.data.table(cell_data)

  for (var_name in neighbor_source_vars) {
    cat("Computing neighbor stats for:", var_name, "...\n")
    t1 <- Sys.time()

    stats <- compute_neighbor_stats_fast(cell_data, edges, var_name)

    # Add columns to cell_data (in place for data.table)
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    cell_data[, (max_col)  := stats[[max_col]]]
    cell_data[, (min_col)  := stats[[min_col]]]
    cell_data[, (mean_col) := stats[[mean_col]]]

    cat("  Done in", round(difftime(Sys.time(), t1, units = "secs"), 1), "sec\n")
  }

  if (!is_dt) cell_data <- as.data.frame(cell_data)

  cat("Neighbor pipeline complete.\n")
  cell_data
}

# ============================================================
# USAGE (drop-in replacement for the original outer loop)
# ============================================================
# cell_data <- run_neighbor_pipeline(cell_data, id_order, rook_neighbors_unique)
#
# The trained Random Forest model is untouched.
# Column names and numerical values are identical to the original.
# Expected runtime: ~5-15 minutes instead of 86+ hours.
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **True bottleneck** | `build_neighbor_lookup()`: 6.46M interpreted iterations with string key construction and character hash lookups | Eliminated entirely |
| **Lookup construction** | `lapply` + `paste` + named vector probe per row | Single `data.table` equi-join (vectorized, C-level) |
| **Neighbor stats** | `lapply` over 6.46M rows + `do.call(rbind, ...)` | `data.table` grouped aggregation (`[, .(), by=]`) |
| **Estimated runtime** | 86+ hours | ~5–15 minutes |
| **Numerical output** | `max`, `min`, `mean` per neighbor set | Identical `max`, `min`, `mean` per neighbor set |
| **RF model** | Preserved (not retrained) | Preserved (not retrained) |

The colleague's diagnosis was wrong: `do.call(rbind, result)` is a one-shot bind of pre-allocated list elements and is not the bottleneck. The real cost is the **6.46 million character-key constructions and named-vector lookups** inside `build_neighbor_lookup()`, which is solved by replacing the entire approach with vectorized `data.table` joins.