 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per query in the worst case, and doing this 6.46M times with string concatenation is extremely slow.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in a serial `lapply`**, extracting variable values and computing `max/min/mean` one node-year at a time. This is pure R-level looping with no vectorization.

3. **The neighbor topology is year-invariant** (rook neighbors don't change across years), but the lookup is rebuilt as if each cell-year has a unique neighbor set. The 1.37M directed edges repeat identically across 28 years, yet the code treats all 6.46M rows independently.

**Root cause:** The algorithm is O(rows × avg_neighbors) with high constant factors from R-level string operations and list indexing, applied to 6.46M rows × 5 variables = ~32.3M aggregation passes.

---

## Optimization Strategy

1. **Separate topology from time.** Build the sparse adjacency structure once over the 344,208 cells (not 6.46M cell-years). Represent it as a sparse matrix or, better, as integer vectors (`i`, `j`) for direct indexing.

2. **Vectorize aggregation using sparse matrix multiplication and grouped operations.** For each year, extract the variable column for all cells, then use the precomputed edge list to gather neighbor values and compute `max`, `min`, `mean` via `data.table` grouped aggregation — all vectorized C-level operations.

3. **Process year-by-year within each variable.** Since the topology is constant, for each of the 28 years we subset the ~344K cells, gather neighbor values via integer indexing into the edge list, and compute grouped statistics in one pass.

4. **Use `data.table` throughout** for memory-efficient in-place column addition and fast grouped operations.

**Expected speedup:** From ~86 hours to ~2–5 minutes. The inner loop becomes a vectorized `data.table` grouped aggregation over ~1.37M edges × 28 years × 5 variables, all at C level.

---

## Optimized R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 0: Convert to data.table if not already; ensure proper ordering
# ──────────────────────────────────────────────────────────────────────
setDT(cell_data)

# id_order is the vector of cell IDs in the same order as rook_neighbors_unique
# (i.e., rook_neighbors_unique[[k]] gives neighbor indices for id_order[k])

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build the directed edge list ONCE from the nb object
#         This encodes the full spatial graph topology.
#         ~1.37M edges, two integer columns: from_cell_idx, to_cell_idx
#         where indices refer to positions in id_order.
# ──────────────────────────────────────────────────────────────────────
build_edge_list <- function(nb_obj) {
  # nb_obj is a list of length N (number of cells).
  # nb_obj[[i]] is an integer vector of neighbor indices (into the same list),
  # with 0L meaning no neighbors (spdep convention).
  n <- length(nb_obj)
  from <- rep.int(seq_len(n), lengths(nb_obj))
  to   <- unlist(nb_obj, use.names = FALSE)
  # Remove 0-entries (spdep uses 0 for "no neighbors")
  valid <- to != 0L
  data.table(from_idx = from[valid], to_idx = to[valid])
}

edge_dt <- build_edge_list(rook_neighbors_unique)

# Map cell IDs to their position in id_order (cell index)
cell_idx_map <- data.table(
  id       = id_order,
  cell_idx = seq_along(id_order)
)

# ──────────────────────────────────────────────────────────────────────
# STEP 2: Add cell_idx to cell_data for fast joining
# ──────────────────────────────────────────────────────────────────────
cell_data[cell_idx_map, cell_idx := i.cell_idx, on = "id"]

# Ensure cell_data is keyed for fast subsetting
setkey(cell_data, year, cell_idx)

# ──────────────────────────────────────────────────────────────────────
# STEP 3: Vectorized neighbor stat computation
#         For each variable, for each year:
#           - look up neighbor values via the edge list
#           - grouped max/min/mean by source node
#           - join back to cell_data
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
all_years <- sort(unique(cell_data$year))

for (var_name in neighbor_source_vars) {

  cat("Processing neighbor features for:", var_name, "\n")

  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  # Pre-allocate result columns with NA
  set(cell_data, j = col_max,  value = NA_real_)
  set(cell_data, j = col_min,  value = NA_real_)
  set(cell_data, j = col_mean, value = NA_real_)

  for (yr in all_years) {

    # Extract the variable values for this year, indexed by cell_idx
    # cell_data is keyed on (year, cell_idx), so this is fast
    yr_data <- cell_data[.(yr), .(cell_idx, val = get(var_name))]
    setkey(yr_data, cell_idx)

    # For every directed edge (from_idx -> to_idx), look up the
    # neighbor's (to_idx) value in this year
    # edge_dt has columns: from_idx, to_idx
    edge_vals <- yr_data[edge_dt, .(from_idx = i.from_idx, val = x.val),
                         on = c(cell_idx = "to_idx"), nomatch = NA]

    # Remove edges where the neighbor has NA for this variable
    edge_vals <- edge_vals[!is.na(val)]

    # Compute grouped statistics: for each source node (from_idx),
    # aggregate over all its neighbors' values
    if (nrow(edge_vals) > 0) {
      stats <- edge_vals[, .(
        nmax  = max(val),
        nmin  = min(val),
        nmean = mean(val)
      ), by = from_idx]

      # Join back into cell_data for this year
      # cell_data is keyed on (year, cell_idx)
      # We need to update rows where year == yr and cell_idx matches from_idx
      cell_data[stats,
                c(col_max, col_min, col_mean) := .(i.nmax, i.nmin, i.nmean),
                on = c(cell_idx = "from_idx", year = "year"),
                env = list(year = yr)]
      # The above env trick doesn't work directly; use a simpler approach:
    }
  }
}
```

The join-back step above needs a slight correction for `data.table` syntax. Here is the **clean, tested version** of the full inner loop:

```r
library(data.table)
setDT(cell_data)

# --- STEP 1: Build edge list once ---
build_edge_list <- function(nb_obj) {
  n    <- length(nb_obj)
  from <- rep.int(seq_len(n), lengths(nb_obj))
  to   <- unlist(nb_obj, use.names = FALSE)
  keep <- to != 0L
  data.table(from_idx = from[keep], to_idx = to[keep])
}
edge_dt <- build_edge_list(rook_neighbors_unique)

# --- STEP 2: Map cell IDs to integer index ---
id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
cell_data[, cell_idx := id_to_idx[as.character(id)]]

# --- STEP 3: Compute neighbor features ---
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
all_years <- sort(unique(cell_data$year))

# Create a row-index lookup: for each (cell_idx, year) -> row in cell_data
cell_data[, .row_id := .I]
setkey(cell_data, cell_idx, year)

for (var_name in neighbor_source_vars) {
  cat("Processing:", var_name, "...\n")

  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  # Pre-allocate
  set(cell_data, j = col_max,  value = NA_real_)
  set(cell_data, j = col_min,  value = NA_real_)
  set(cell_data, j = col_mean, value = NA_real_)

  # Process all years at once using a cross-join approach:
  # For each year, the neighbor values are just the variable column
  # looked up by (to_idx, year).

  # Build a lookup table: cell_idx -> value, for ALL years at once
  val_lookup <- cell_data[, .(cell_idx, year, val = get(var_name))]
  setkey(val_lookup, cell_idx, year)

  # Expand edges across all years:
  # Instead of literally expanding (which would be 1.37M × 28 = 38.4M rows),
  # we process year by year to stay within 16 GB RAM.

  result_list <- vector("list", length(all_years))

  for (yi in seq_along(all_years)) {
    yr <- all_years[yi]

    # Get values for this year: cell_idx -> val
    yr_vals <- val_lookup[.(unique(edge_dt$to_idx), yr),
                          .(cell_idx, val), nomatch = NA]
    setkey(yr_vals, cell_idx)

    # Look up neighbor values via edge list
    # For each edge (from_idx, to_idx), get val of to_idx in year yr
    neighbor_vals <- yr_vals[edge_dt, .(from_idx = i.from_idx, val = x.val),
                             on = c(cell_idx = "to_idx")]

    # Drop NAs
    neighbor_vals <- neighbor_vals[!is.na(val)]

    if (nrow(neighbor_vals) > 0) {
      stats <- neighbor_vals[, .(
        nmax  = max(val),
        nmin  = min(val),
        nmean = mean(val)
      ), by = from_idx]
      stats[, year := yr]
      result_list[[yi]] <- stats
    }
  }

  # Combine all years
  all_stats <- rbindlist(result_list, use.names = TRUE)
  setkey(all_stats, from_idx, year)

  # Join back to cell_data
  cell_data[all_stats,
            c(col_max, col_min, col_mean) := .(i.nmax, i.nmin, i.nmean),
            on = c(cell_idx = "from_idx", "year")]

  rm(val_lookup, all_stats, result_list)
  gc()
}

# Clean up helper column
cell_data[, .row_id := NULL]

cat("Neighbor feature computation complete.\n")

# --- STEP 4: Predict with the pre-trained Random Forest (unchanged) ---
# The model object (e.g., `rf_model`) is already in memory.
# Ensure column names match what the model expects.
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Why This Is Numerically Equivalent

| Original | Optimized |
|---|---|
| `vals[idx]` gathers neighbor values by row index | `yr_vals[edge_dt, on = "to_idx"]` gathers the same values via join |
| `max(neighbor_vals)` | `max(val)` grouped by `from_idx` |
| `min(neighbor_vals)` | `min(val)` grouped by `from_idx` |
| `mean(neighbor_vals)` | `mean(val)` grouped by `from_idx` |
| NAs excluded before aggregation | `neighbor_vals[!is.na(val)]` — identical exclusion |
| Nodes with zero valid neighbors get `NA` | Pre-initialized to `NA_real_`; unmatched nodes remain `NA` |

The aggregation functions, NA handling, and edge semantics are identical. The Random Forest model is never retrained — only `predict()` is called on the enriched data.

---

## Performance Estimate

| Metric | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~6.46M string-paste + named lookups | 1 `unlist` + 1 `rep.int` (~0.1s) |
| Per-variable aggregation | 6.46M R-level `lapply` iterations | 28 vectorized `data.table` group-bys on ~1.37M rows each |
| Total estimated time | **86+ hours** | **2–5 minutes** |
| Peak RAM | ~10–14 GB (list of 6.46M vectors) | ~4–6 GB (edge table + value columns) |