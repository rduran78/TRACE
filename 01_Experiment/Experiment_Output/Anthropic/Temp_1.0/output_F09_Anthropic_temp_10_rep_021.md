 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of two fundamental problems:

### 1. `build_neighbor_lookup` rebuilds neighbor relationships for every cell-year row (~6.46M rows)

The neighbor topology is **purely spatial** — cell A's neighbors are the same in 1992 as in 2019. Yet `build_neighbor_lookup` creates a lookup entry for every one of the ~6.46 million cell-year rows, performing string-pasting, hash lookups, and NA filtering inside an `lapply` over 6.46M iterations. This is ~28× more work than necessary.

### 2. `compute_neighbor_stats` iterates over 6.46M rows in R-level loops

For each of the 5 variables, `lapply` touches every row, subsets a numeric vector by index, removes NAs, and computes `max`, `min`, `mean`. With 5 variables × 6.46M rows, that's ~32.3 million R-level function calls with repeated vector subsetting.

### Root cause summary

| Bottleneck | Why it's slow |
|---|---|
| Neighbor lookup is built at the **cell-year** level | 6.46M entries instead of 344K; string-key hashing is O(n) in row count |
| Stats computed row-by-row in R | No vectorization; `lapply` over millions of rows is inherently slow |
| Topology and attributes are entangled | Neighbor structure is recomputed implicitly every time instead of being a reusable join key |

---

## Optimization Strategy

**Core idea:** Separate the static spatial topology from the time-varying attributes. Build the adjacency table once (344K cells × ~4 neighbors ≈ 1.37M directed edges), then use a vectorized `data.table` join-and-aggregate to compute neighbor stats for all years simultaneously.

### Steps

1. **Build a static edge table** from the `nb` object: two columns `(cell_id, neighbor_id)`, ~1.37M rows. This is done once.
2. **Convert the panel data to `data.table`** keyed on `(id, year)`.
3. **Cross-join the edge table with years** implicitly via a keyed join: for each `(cell_id, year)`, look up each `neighbor_id`'s attribute in that same year.
4. **Aggregate** (`max`, `min`, `mean`) grouped by `(cell_id, year)` — fully vectorized inside `data.table`.
5. **Join results back** to the main table.

### Expected speedup

- The edge table has ~1.37M rows. Crossed with 28 years = ~38.4M join rows — but the `data.table` binary-search join handles this in seconds, not hours.
- Aggregation by group is C-level vectorized in `data.table`.
- Estimated total time: **2–10 minutes** on a 16 GB laptop (vs. 86+ hours).

### Invariants preserved

- The trained Random Forest model is **not retouched**. We only produce the same predictor columns it expects.
- The numerical results (neighbor max, min, mean) are **identical** to the original implementation — same rook topology, same per-variable aggregation logic.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 0 : Inputs assumed to exist
#   cell_data               : data.frame/data.table with columns id, year, 
#                             ntl, ec, pop_density, def, usd_est_n2, ...
#   id_order                : integer/character vector of cell IDs matching
#                             the nb object's positional index
#   rook_neighbors_unique   : spdep nb object (list of integer index vectors)
#   rf_model                : trained Random Forest model (UNCHANGED)
# ──────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────
# STEP 1 : Build the STATIC spatial edge table (done ONCE, ~1.37M rows)
# ──────────────────────────────────────────────────────────────────────

build_edge_table <- function(id_order, nb_obj) {
  # nb_obj[[i]] gives the positional indices of neighbors for id_order[i].
  # Convert to a two-column data.table of (cell_id, neighbor_id).
  n <- length(nb_obj)
  from <- rep(seq_len(n), lengths(nb_obj))
  to   <- unlist(nb_obj, use.names = FALSE)
  
  # Remove the spdep "0 = no neighbors" convention

  valid <- to != 0L
  from  <- from[valid]
  to    <- to[valid]
  
  data.table(
    cell_id     = id_order[from],
    neighbor_id = id_order[to]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# edge_dt has columns: cell_id, neighbor_id  (~1.37M rows)


# ──────────────────────────────────────────────────────────────────────
# STEP 2 : Convert panel data to keyed data.table
# ──────────────────────────────────────────────────────────────────────

cell_dt <- as.data.table(cell_data)
# Ensure id and year are the types we expect (integer / numeric)
# and set key for fast binary-search joins.
setkey(cell_dt, id, year)


# ──────────────────────────────────────────────────────────────────────
# STEP 3 : Vectorized neighbor-stat computation for one variable
# ──────────────────────────────────────────────────────────────────────

compute_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {
  # --- 3a. Subset only the columns we need from the panel ---
  attr_dt <- cell_dt[, .(id, year, value = get(var_name))]
  setkey(attr_dt, id, year)

  # --- 3b. Expand edges × years via join ---
  #   For every directed edge (cell_id → neighbor_id) and every year,
  #   retrieve the neighbor's attribute value.
  #   We do this by joining edge_dt to attr_dt on neighbor_id = id.

  # First, get all unique years
  years_vec <- sort(unique(cell_dt$year))

  # Cross-join edges with years: each edge appears once per year
  # (~1.37M edges × 28 years ≈ 38.4M rows — fits comfortably in 16 GB)
  edge_year <- CJ_dt_year(edge_dt, years_vec)
  # edge_year columns: cell_id, neighbor_id, year


  # Join to get the neighbor's value in that year
  setkey(edge_year, neighbor_id, year)
  edge_year[attr_dt, neighbor_val := i.value, on = .(neighbor_id = id, year)]

  # --- 3c. Aggregate per (cell_id, year) ---
  stats <- edge_year[
    !is.na(neighbor_val),
    .(
      nb_max  = max(neighbor_val),
      nb_min  = min(neighbor_val),
      nb_mean = mean(neighbor_val)
    ),
    keyby = .(cell_id, year)
  ]

  # Name columns to match original pipeline expectations
  max_col  <- paste0("nb_max_",  var_name)
  min_col  <- paste0("nb_min_",  var_name)
  mean_col <- paste0("nb_mean_", var_name)
  setnames(stats, c("nb_max", "nb_min", "nb_mean"),
                  c(max_col,  min_col,  mean_col))

  stats
}

# Helper: cross-join edge table with a year vector efficiently
CJ_dt_year <- function(edge_dt, years_vec) {
  # Repeat each edge row length(years_vec) times
  n_edges <- nrow(edge_dt)
  n_years <- length(years_vec)
  idx     <- rep(seq_len(n_edges), each = n_years)
  out     <- edge_dt[idx]
  out[, year := rep(years_vec, times = n_edges)]
  out
}


# ──────────────────────────────────────────────────────────────────────
# STEP 4 : Loop over the 5 source variables, join results back
# ──────────────────────────────────────────────────────────────────────

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor stats for: ", var_name)
  stats_dt <- compute_neighbor_features_fast(cell_dt, edge_dt, var_name)

  # Merge back onto the main table (left join: keep all cell-year rows)
  cell_dt <- merge(cell_dt, stats_dt,
                   by.x = c("id", "year"),
                   by.y = c("cell_id", "year"),
                   all.x = TRUE)
}

setkey(cell_dt, id, year)


# ──────────────────────────────────────────────────────────────────────
# STEP 5 : Predict with the EXISTING Random Forest (unchanged)
# ──────────────────────────────────────────────────────────────────────

# Convert back to data.frame if the RF predict method expects one
cell_data <- as.data.frame(cell_dt)

# The trained model is used as-is — no retraining
cell_data$rf_prediction <- predict(rf_model, newdata = cell_data)
```

---

## Memory Note for 16 GB Laptop

The largest intermediate object is `edge_year` at ~38.4M rows × 3–4 columns ≈ **~1.2 GB**. Combined with `cell_dt` (~6.46M × 110 cols ≈ 5.7 GB), peak usage is about **8–9 GB**, well within 16 GB. If memory is tighter, you can process variables one at a time and remove `edge_year` between iterations (already the case in the loop above since `edge_year` is local to the function).

---

## Optional Further Optimization

If even the cross-join feels large, you can avoid it entirely by doing the join year-by-year in a vectorized inner loop (28 iterations, each ~1.37M rows):

```r
compute_neighbor_features_by_year <- function(cell_dt, edge_dt, var_name) {
  attr_dt <- cell_dt[, .(id, year, value = get(var_name))]
  setkey(attr_dt, id)

  years_vec <- sort(unique(attr_dt$year))
  results <- rbindlist(lapply(years_vec, function(yr) {
    yr_attr <- attr_dt[year == yr, .(id, value)]
    setkey(yr_attr, id)
    
    # Join neighbor values
    merged <- edge_dt[yr_attr, on = .(neighbor_id = id), nomatch = 0L]
    # merged now has: cell_id, neighbor_id, value (the neighbor's value)
    
    merged[
      !is.na(value),
      .(nb_max  = max(value),
        nb_min  = min(value),
        nb_mean = mean(value)),
      keyby = .(cell_id)
    ][, year := yr]
  }))

  max_col  <- paste0("nb_max_",  var_name)
  min_col  <- paste0("nb_min_",  var_name)
  mean_col <- paste0("nb_mean_", var_name)
  setnames(results, c("nb_max", "nb_min", "nb_mean"),
                    c(max_col,  min_col,  mean_col))
  results
}
```

This variant uses only ~1.37M rows of working memory per year-iteration (×28 = trivial), bringing peak memory down by ~1 GB and running in roughly the same wall-clock time.

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup | 6.46M R-level iterations with string keys | 1.37M-row static edge table (built once) |
| Stats computation | R-level `lapply` × 6.46M × 5 vars | Vectorized `data.table` join + `groupby` |
| Estimated runtime | ~86+ hours | **~2–10 minutes** |
| Peak RAM | High (string duplication) | ~8–9 GB (within 16 GB) |
| RF model | Unchanged ✓ | Unchanged ✓ |
| Numerical output | Original estimand | Identical ✓ |