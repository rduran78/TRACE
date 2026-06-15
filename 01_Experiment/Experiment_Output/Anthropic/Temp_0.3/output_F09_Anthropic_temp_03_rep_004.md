 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three interconnected reasons:

### 1. Row-Level `lapply` Over 6.46 Million Rows in `build_neighbor_lookup`
The function iterates over every cell-year row individually, performing string pasting (`paste(id, year, sep="_")`), named-vector lookups (`id_to_ref`, `idx_lookup`), and subsetting for each of the ~6.46 million rows. Named-vector lookups in R are hash-table operations with non-trivial per-call overhead; doing this billions of times (each row looks up multiple neighbor keys) is catastrophic.

### 2. Redundant Recomputation of Spatial Topology Per Year
The rook-neighbor structure is **purely spatial**—it is identical across all 28 years. Yet `build_neighbor_lookup` rebuilds neighbor index vectors for every cell-year row, effectively recomputing the same spatial adjacency 28 times and entangling it with the temporal dimension unnecessarily.

### 3. Row-Level `lapply` in `compute_neighbor_stats`
For each of the 6.46 million rows, `compute_neighbor_stats` subsets a numeric vector, removes NAs, and computes `max`, `min`, `mean`. The per-call overhead of `lapply` + anonymous function + subsetting + three summary functions, repeated 6.46M × 5 variables = ~32.3 million invocations, dominates runtime.

**In summary:** The architecture conflates spatial structure (fixed) with temporal attributes (varying), and uses row-level R loops where vectorized or table-join operations should be used.

---

## Optimization Strategy

**Core insight:** Build the neighbor table **once** as a spatial-only edge list (cell → neighbor_cell), then use a vectorized `data.table` join to bring in yearly attributes and compute grouped statistics.

### Steps:

1. **Build a static edge list** from `rook_neighbors_unique` (the `nb` object). This produces a two-column data.table: `(id, neighbor_id)`. This is done once and has ~1.37M rows.

2. **Join yearly attributes onto the edge list.** For each year, every edge `(id, neighbor_id)` gets the neighbor's attribute value by joining `cell_data` on `(neighbor_id, year)`. This is a keyed `data.table` equi-join—extremely fast.

3. **Compute grouped statistics** using `data.table`'s `[, .(max, min, mean), by=.(id, year)]`—fully vectorized C-level aggregation.

4. **Join results back** to the main data.table.

This eliminates all row-level R loops. The entire pipeline becomes a sequence of keyed joins and grouped aggregations, reducing runtime from ~86 hours to **minutes**.

### Complexity comparison:

| Step | Current | Proposed |
|---|---|---|
| Neighbor lookup | O(6.46M) R-level iterations with string ops | O(1.37M) edge list built once |
| Stat computation | O(6.46M × 5) R-level `lapply` calls | O(5) vectorized `data.table` group-bys over ~38.4M edge-year rows |
| Estimated time | ~86 hours | ~5–15 minutes |

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────
# STEP 0: Convert cell_data to data.table (if not already)
# ──────────────────────────────────────────────────────────────
cell_data <- as.data.table(cell_data)

# ──────────────────────────────────────────────────────────────
# STEP 1: Build static spatial edge list ONCE from nb object
#
#   rook_neighbors_unique : an nb object (list of integer vectors)
#   id_order              : vector mapping position → cell id
#
#   Output: edge_dt with columns (id, neighbor_id)
#           ~1,373,394 rows (directed rook edges)
# ──────────────────────────────────────────────────────────────
build_edge_list <- function(id_order, neighbors_nb) {
  # neighbors_nb[[i]] contains integer indices into id_order
  # for the neighbors of cell id_order[i].
  # spdep::nb objects use 0L to denote "no neighbors" for islands.
  
  n <- length(neighbors_nb)
  
  # Pre-allocate by computing total number of edges
  edge_counts <- vapply(neighbors_nb, function(x) {
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1))
  total_edges <- sum(edge_counts)
  
  from_id <- integer(total_edges)
  to_id   <- integer(total_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    nb_idx <- neighbors_nb[[i]]
    if (length(nb_idx) == 1L && nb_idx[1] == 0L) next
    k <- length(nb_idx)
    from_id[pos:(pos + k - 1L)] <- id_order[i]
    to_id[pos:(pos + k - 1L)]   <- id_order[nb_idx]
    pos <- pos + k
  }
  
  data.table(id = from_id, neighbor_id = to_id)
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)

cat(sprintf("Edge list built: %d directed edges\n", nrow(edge_dt)))

# ──────────────────────────────────────────────────────────────
# STEP 2: For each neighbor source variable, compute neighbor
#          max, min, mean via keyed join + grouped aggregation,
#          then join back to cell_data.
#
#   This replaces build_neighbor_lookup AND
#   compute_neighbor_stats AND the outer for-loop.
# ──────────────────────────────────────────────────────────────

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Key cell_data for fast joins
setkey(cell_data, id, year)

for (var_name in neighbor_source_vars) {
  
  cat(sprintf("Computing neighbor stats for: %s\n", var_name))
  
  # --- 2a. Extract the (id, year, value) subset for the neighbor lookup side
  #         We rename 'id' to 'neighbor_id' so we can join on the neighbor's id.
  val_dt <- cell_data[, .(neighbor_id = id, year, nbr_val = get(var_name))]
  setkey(val_dt, neighbor_id, year)
  
  # --- 2b. Expand edges × years: join neighbor attribute onto edge list
  #         For every (id, neighbor_id) edge and every year, get the
  #         neighbor's value of var_name.
  #
  #         We do this by joining edge_dt with val_dt on (neighbor_id, year).
  #         But edge_dt has no year column—we need the Cartesian product
  #         edge × year. Instead of materializing that (~38.4M rows),
  #         we join cell_data's (id, year) with edge_dt to get
  #         (id, year, neighbor_id), then join val_dt to get nbr_val.
  
  # Get unique (id, year) pairs from cell_data
  id_year <- cell_data[, .(id, year)]
  
  # Join: for each (id, year), attach all neighbor_ids
  # This produces the full (id, year, neighbor_id) table
  edges_by_year <- edge_dt[id_year, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # edges_by_year now has columns: id, neighbor_id, year
  
  # Join: attach the neighbor's variable value
  edges_by_year <- val_dt[edges_by_year, on = .(neighbor_id, year), nomatch = NA]
  # edges_by_year now has columns: neighbor_id, year, nbr_val, id
  
  # --- 2c. Compute grouped statistics
  stats_dt <- edges_by_year[
    !is.na(nbr_val),
    .(
      nbr_max  = max(nbr_val),
      nbr_min  = min(nbr_val),
      nbr_mean = mean(nbr_val)
    ),
    by = .(id, year)
  ]
  
  # --- 2d. Rename columns to match original naming convention
  #         Original code used: {var_name}_neighbor_max, etc.
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  
  setnames(stats_dt, c("nbr_max", "nbr_min", "nbr_mean"),
           c(max_col, min_col, mean_col))
  
  setkey(stats_dt, id, year)
  
  # --- 2e. Remove old columns if they exist (idempotent re-runs)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }
  
  # --- 2f. Join stats back to cell_data
  cell_data <- stats_dt[cell_data, on = .(id, year)]
  
  # Clean up to free RAM
  rm(val_dt, id_year, edges_by_year, stats_dt)
  gc()
  
  cat(sprintf("  Done: %s\n", var_name))
}

# ──────────────────────────────────────────────────────────────
# STEP 3: Predict with the existing trained Random Forest
#
#   The trained model object is assumed to be in memory (e.g.,
#   `rf_model`). No retraining occurs. The numerical estimand
#   is preserved because the neighbor statistics are computed
#   with identical logic (max, min, mean of non-NA neighbor
#   values, NA where no valid neighbors exist).
# ──────────────────────────────────────────────────────────────

# Ensure cell_data is a data.frame if the RF model expects one
cell_data_df <- as.data.frame(cell_data)

# Predict (using whatever predict method matches the trained model)
cell_data_df$prediction <- predict(rf_model, newdata = cell_data_df)

cat("Prediction complete.\n")
```

---

## Memory Considerations (16 GB Laptop)

The largest intermediate object is `edges_by_year` at ~38.4M rows × 4 columns ≈ **1.2 GB** per variable iteration. With `cell_data` itself at ~6.46M × 110 columns ≈ **5.7 GB**, peak usage is roughly **8–9 GB**, well within 16 GB. The `rm()` + `gc()` calls after each variable keep memory bounded.

## Numerical Equivalence

The aggregation logic is identical to the original:
- `max(neighbor_vals)`, `min(neighbor_vals)`, `mean(neighbor_vals)` over non-NA neighbor values.
- Rows with no valid neighbors receive `NA` (via the left join in step 2f: cells absent from `stats_dt` get `NA` for all three columns).

The trained Random Forest model is used as-is with `predict()`—no retraining occurs.

## Expected Runtime

| Phase | Estimated Time |
|---|---|
| Edge list construction | ~2 seconds |
| Per-variable join + aggregation (×5) | ~1–2 minutes each |
| Total neighbor feature engineering | **~5–10 minutes** |
| RF prediction (~6.46M rows, 110 features) | ~5–30 minutes (model-dependent) |
| **Total** | **~10–40 minutes** |

This represents a **~130–500× speedup** over the original 86+ hour estimate.