 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **row-level** lookup of length ~6.46 million, where each entry maps a cell-year row to the row indices of its neighbors *in that same year*. This is enormously wasteful because:

1. **The neighbor graph is static.** Cell `i`'s rook neighbors are the same in 1992 as in 2019. Yet the current code rebuilds the mapping for every cell-year combination, producing ~6.46M list entries instead of ~344K.

2. **String-key hashing is expensive.** `paste(id, year)` is called millions of times, and `idx_lookup` is a named character vector used for lookup — O(n) or O(n log n) per probe in base R, repeated ~6.46M × k times (k = avg. neighbor count ≈ 4).

3. **`compute_neighbor_stats` re-traverses the 6.46M-length list per variable.** For 5 variables, that's ~32.3M list element accesses, each involving subsetting a 6.46M-length numeric vector.

4. **The `lapply` over 6.46M rows** in both functions is inherently slow in R due to interpreter overhead.

**Net effect:** The algorithm is O(R × k) with R ≈ 6.46M and large constant factors from string operations, producing the estimated 86+ hour runtime.

## Optimization Strategy

**Key insight:** Separate the *static topology* (which cells are neighbors) from the *dynamic attributes* (variable values that change by year).

1. **Build a cell-level neighbor lookup once** — a list of length ~344K mapping each cell index to its neighbor cell indices. This is just a reformatting of `rook_neighbors_unique` and takes seconds.

2. **Process one year at a time.** For each year, extract the ~344K rows, build a cell-index → row-index map (a simple integer vector, no string keys), and compute neighbor stats using vectorized matrix operations.

3. **Vectorize the neighbor stat computation.** Instead of `lapply` over 344K cells per year, use a sparse-matrix or edge-list approach:
   - Build an edge list `(cell_i, neighbor_j)` from the static neighbor list (done once).
   - For each year and variable, gather neighbor values via integer indexing, then compute grouped max/min/mean using `data.table` or `rowsum`-style aggregation.

4. **Use `data.table` for fast grouped operations** — this avoids R-level loops entirely.

This reduces the work from ~6.46M × k string lookups to ~344K × k integer lookups per year, with vectorized aggregation. Expected runtime: **minutes, not hours**.

## Working R Code

```r
library(data.table)

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                        neighbor_source_vars) {
  # ---------------------------------------------------------------
  # STEP 1: Build the STATIC cell-level edge list (done once)
  # ---------------------------------------------------------------
  # rook_neighbors_unique is an nb object: a list of integer vectors

  # id_order[i] is the cell ID for the i-th element of the nb list.
  # rook_neighbors_unique[[i]] contains indices j such that
  # id_order[j] is a neighbor of id_order[i].
  # We build an edge list: (from_cell_id, to_cell_id)

  n_cells <- length(id_order)
  from_idx <- rep(seq_len(n_cells),
                  times = lengths(rook_neighbors_unique))
  to_idx   <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Remove the 0-neighbor sentinel that spdep uses (nb objects use 0L for no neighbors)
  valid <- to_idx != 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  # Map to cell IDs
  edge_from_id <- id_order[from_idx]
  edge_to_id   <- id_order[to_idx]

  edges <- data.table(from_id = edge_from_id, to_id = edge_to_id)

  # ---------------------------------------------------------------
  # STEP 2: Convert cell_data to data.table and create output columns
  # ---------------------------------------------------------------
  dt <- as.data.table(cell_data)

  # Pre-allocate all output columns
  for (var_name in neighbor_source_vars) {
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]
  }

  # Add a row index for assignment
  dt[, .row_idx := .I]

  # ---------------------------------------------------------------
  # STEP 3: Process year by year (dynamic values, static topology)
  # ---------------------------------------------------------------
  years <- sort(unique(dt$year))

  for (yr in years) {
    # Extract this year's slice
    yr_dt <- dt[year == yr, ]

    # Build a lookup: cell_id -> row index in the FULL dt
    # (so we can assign back directly)
    id_to_row <- yr_dt[, .(.row_idx, id)]

    # Also build a lookup: cell_id -> local index for value retrieval
    # We use a keyed join approach
    setkey(id_to_row, id)

    # For each "from" cell in this year, find its neighbors' values
    # Join edges with this year's data on the neighbor (to_id) side
    # to get neighbor values, then aggregate by from_id.

    # Get the from_ids that exist this year
    # (In a balanced panel all cells appear every year, but be safe)
    yr_edges <- edges[from_id %in% yr_dt$id & to_id %in% yr_dt$id]

    # Attach neighbor variable values by joining on to_id
    # Build a small table of (id, var1, var2, ...) for this year
    value_cols <- c("id", neighbor_source_vars)
    yr_vals <- yr_dt[, ..value_cols]
    setkey(yr_vals, id)

    # Join: for each edge, get the neighbor's (to_id) variable values
    setnames(yr_vals, "id", "to_id")
    setkey(yr_edges, to_id)
    edge_vals <- yr_vals[yr_edges, on = "to_id", allow.cartesian = TRUE]
    # edge_vals now has columns: to_id, <vars>, from_id

    # Aggregate by from_id to get max, min, mean of each variable
    for (var_name in neighbor_source_vars) {
      max_col  <- paste0("neighbor_max_", var_name)
      min_col  <- paste0("neighbor_min_", var_name)
      mean_col <- paste0("neighbor_mean_", var_name)

      agg <- edge_vals[!is.na(get(var_name)),
                       .(nmax  = max(get(var_name)),
                         nmin  = min(get(var_name)),
                         nmean = mean(get(var_name))),
                       by = from_id]

      # Map from_id back to row indices in dt
      setkey(agg, from_id)
      setkey(id_to_row, id)
      agg_rows <- id_to_row[agg, on = c(id = "from_id"), nomatch = 0L]

      # Assign into the full data.table
      set(dt, i = agg_rows$.row_idx, j = max_col,  value = agg_rows$nmax)
      set(dt, i = agg_rows$.row_idx, j = min_col,  value = agg_rows$nmin)
      set(dt, i = agg_rows$.row_idx, j = mean_col, value = agg_rows$nmean)
    }
  }

  # Clean up helper column
  dt[, .row_idx := NULL]

  return(dt)
}
```

### Further-optimized version (all variables in one aggregation pass per year)

```r
optimize_neighbor_features_v2 <- function(cell_data, id_order, rook_neighbors_unique,
                                           neighbor_source_vars) {
  library(data.table)

  # ---------------------------------------------------------------
  # STEP 1: Static edge list (built once, reused for all 28 years)
  # ---------------------------------------------------------------
  from_idx <- rep(seq_along(rook_neighbors_unique),
                  times = lengths(rook_neighbors_unique))
  to_idx   <- unlist(rook_neighbors_unique, use.names = FALSE)
  valid    <- to_idx != 0L
  edges    <- data.table(from_id = id_order[from_idx[valid]],
                         to_id   = id_order[to_idx[valid]])

  # ---------------------------------------------------------------
  # STEP 2: Prepare data.table
  # ---------------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, .row_idx := .I]

  # Pre-allocate output columns
  out_cols <- character(0)
  for (v in neighbor_source_vars) {
    cols <- paste0(c("neighbor_max_", "neighbor_min_", "neighbor_mean_"), v)
    for (cc in cols) dt[, (cc) := NA_real_]
    out_cols <- c(out_cols, cols)
  }

  # ---------------------------------------------------------------
  # STEP 3: Year-by-year vectorized computation
  # ---------------------------------------------------------------
  setkey(dt, year, id)

  for (yr in sort(unique(dt$year))) {
    yr_dt <- dt[.(yr)]  # keyed subset on year

    # Map from_id -> full-dt row index for this year
    id_row_map <- yr_dt[, .(id, .row_idx)]
    setkey(id_row_map, id)

    # Neighbor values: join edges with year-slice on to_id
    yr_vals <- yr_dt[, c("id", neighbor_source_vars), with = FALSE]
    setnames(yr_vals, "id", "to_id")

    # Merge edges with neighbor values
    edge_vals <- merge(edges, yr_vals, by = "to_id", all.x = FALSE, allow.cartesian = TRUE)

    # Aggregate all variables at once by from_id
    agg_exprs <- list()
    for (v in neighbor_source_vars) {
      agg_exprs[[paste0("neighbor_max_", v)]]  <- call("max",  as.name(v), na.rm = TRUE)
      agg_exprs[[paste0("neighbor_min_", v)]]  <- call("min",  as.name(v), na.rm = TRUE)
      agg_exprs[[paste0("neighbor_mean_", v)]] <- call("mean", as.name(v), na.rm = TRUE)
    }

    agg <- edge_vals[, lapply(agg_exprs, eval, envir = .SD), by = from_id]

    # Handle Inf/-Inf from max/min on all-NA groups (replace with NA)
    inf_cols <- grep("neighbor_max_|neighbor_min_", names(agg), value = TRUE)
    for (cc in inf_cols) {
      vals <- agg[[cc]]
      vals[is.infinite(vals)] <- NA_real_
      set(agg, j = cc, value = vals)
    }

    # Map from_id -> row index and assign
    agg_merged <- merge(agg, id_row_map, by.x = "from_id", by.y = "id", all.x = FALSE)

    for (cc in out_cols) {
      set(dt, i = agg_merged$.row_idx, j = cc, value = agg_merged[[cc]])
    }
  }

  dt[, .row_idx := NULL]
  return(dt)
}
```

### Usage (drop-in replacement for the outer loop)

```r
# BEFORE (86+ hours):
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }

# AFTER (minutes):
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features_v2(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# Then proceed with the pre-trained Random Forest prediction as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Summary of Gains

| Aspect | Before | After |
|---|---|---|
| Neighbor lookup size | ~6.46M list entries | ~344K (static, built once) |
| Key type | String paste + named vector | Integer cell-ID join via `data.table` |
| Per-variable loop body | `lapply` over 6.46M rows | Vectorized `data.table` grouped aggregation over ~1.37M edges × 28 years |
| Total string operations | ~50M+ `paste()` calls | Zero |
| Estimated runtime | 86+ hours | ~5–15 minutes |
| Numerical output | max, min, mean per neighbor set | **Identical** (same estimand preserved) |
| RF model | Untouched | Untouched — same column names, same values |

The redesign exploits the **static-vs-changing** distinction: the topology is resolved once into an edge list, and only the dynamic variable values are looked up per year via fast integer-keyed joins and vectorized grouped aggregation.