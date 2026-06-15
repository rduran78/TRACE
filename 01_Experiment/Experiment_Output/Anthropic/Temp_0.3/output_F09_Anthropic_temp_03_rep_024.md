 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of **two compounding bottlenecks**:

### 1. `build_neighbor_lookup` — O(n²)-like per-row string hashing
For every one of the ~6.46 million cell-year rows, the function:
- Converts the cell ID to a reference index.
- Looks up neighbor cell IDs from the `nb` object.
- Pastes together `(neighbor_id, year)` string keys.
- Matches those keys against a named character vector of length 6.46 million.

Named-vector lookup in R is **hash-based but with per-call overhead**. Doing ~6.46 million `paste` + named-vector lookups, each touching multiple neighbors, results in billions of character operations. The lookup table itself is a **list of 6.46 million integer vectors**, consuming enormous memory and time to construct.

### 2. `compute_neighbor_stats` — Repeated `lapply` over 6.46M elements
For each of the 5 variables, `lapply` iterates over 6.46 million list elements, subsetting a numeric vector by index, removing NAs, and computing max/min/mean. This is pure R-level looping — no vectorization.

### Root Cause
The neighbor topology is **static across years** (344,208 cells, ~1.37M directed neighbor pairs), but the code re-discovers neighbor row indices **per cell-year row** (6.46M rows), entangling spatial structure with temporal structure. This is the core inefficiency.

---

## Optimization Strategy

**Separate spatial topology from temporal attributes, then use vectorized joins.**

1. **Build the neighbor edge table once** — a two-column `data.table` of `(cell_id, neighbor_id)` with ~1.37M rows. This is year-independent.

2. **For each variable, join yearly attributes onto the edge table** — attach the variable value for each `(neighbor_id, year)` pair. This produces ~1.37M × 28 ≈ ~38.5M rows (but done via a keyed `data.table` equi-join, which is extremely fast).

3. **Aggregate (max, min, mean) by `(cell_id, year)`** — a single grouped `data.table` aggregation, fully vectorized in C.

4. **Join the aggregated stats back onto the main dataset.**

This replaces 6.46 million R-level list operations with a handful of `data.table` keyed joins and group-by aggregations. Expected runtime: **minutes, not hours**.

Memory: the edge table × years is ~38.5M rows × a few columns of integers/doubles — well within 16 GB.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# STEP 0 — Convert the spdep nb object to a static edge table
#          (done ONCE; can be serialized to disk)
# ---------------------------------------------------------------
build_edge_table <- function(id_order, nb_obj) {
  # id_order: vector of 344,208 cell IDs (in the order matching nb_obj)
  # nb_obj:   spdep nb list of length 344,208
  edges <- rbindlist(lapply(seq_along(nb_obj), function(i) {
    nbrs <- nb_obj[[i]]
    # spdep nb encodes "no neighbors" as 0L; skip those
    nbrs <- nbrs[nbrs != 0L]
    if (length(nbrs) == 0L) return(NULL)
    data.table(cell_id = id_order[i], neighbor_id = id_order[nbrs])
  }))
  edges
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# edge_dt has ~1,373,394 rows: (cell_id, neighbor_id)

# ---------------------------------------------------------------
# STEP 1 — Convert main data to data.table (if not already)
# ---------------------------------------------------------------
cell_dt <- as.data.table(cell_data)

# Ensure key columns are of consistent type
edge_dt[, cell_id     := as.integer(cell_id)]
edge_dt[, neighbor_id := as.integer(neighbor_id)]
cell_dt[, id   := as.integer(id)]
cell_dt[, year := as.integer(year)]

# ---------------------------------------------------------------
# STEP 2 — Function: compute neighbor stats for one variable
# ---------------------------------------------------------------
add_neighbor_features <- function(cell_dt, edge_dt, var_name) {
  # Thin attribute table: just (id, year, value)
  attr_dt <- cell_dt[, .(id, year, value = get(var_name))]
  setkey(attr_dt, id, year)

  # Join neighbor attribute values onto the edge table × year

  # First, cross edge_dt with all years present in the data
  years <- sort(unique(attr_dt$year))

  # Expand edges × years
  # Memory: ~1.37M edges × 28 years ≈ 38.5M rows (manageable)
  edge_year <- CJ_dt(edge_dt, years)

  # Keyed join: attach the neighbor's value for that year
  setkey(edge_year, neighbor_id, year)
  setkey(attr_dt, id, year)
  edge_year[attr_dt, neighbor_val := i.value, on = .(neighbor_id = id, year)]

  # Aggregate by (cell_id, year)
  agg <- edge_year[
    !is.na(neighbor_val),
    .(
      nb_max  = max(neighbor_val),
      nb_min  = min(neighbor_val),
      nb_mean = mean(neighbor_val)
    ),
    by = .(cell_id, year)
  ]

  # Rename columns to match original naming convention
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(agg, c("nb_max", "nb_min", "nb_mean"),
                c(max_col,  min_col,  mean_col))

  # Remove old columns if they exist (idempotent re-runs)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }

  # Join aggregated stats back onto the main table
  setkey(agg, cell_id, year)
  setkey(cell_dt, id, year)
  cell_dt <- agg[cell_dt, on = .(cell_id = id, year)]

  # The join puts cell_id as the key; restore column name to 'id'
  setnames(cell_dt, "cell_id", "id")

  cell_dt
}

# Helper: cross join a data.table with a vector of years
CJ_dt <- function(edge_dt, years) {
  # Repeat each edge for every year — vectorized
  n_edges <- nrow(edge_dt)
  n_years <- length(years)
  idx     <- rep(seq_len(n_edges), times = n_years)
  yr      <- rep(years, each = n_edges)
  out     <- edge_dt[idx]
  out[, year := yr]
  out
}

# ---------------------------------------------------------------
# STEP 3 — Run for all 5 neighbor source variables
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  cell_dt <- add_neighbor_features(cell_dt, edge_dt, var_name)
  gc()
}

# ---------------------------------------------------------------
# STEP 4 — Predict with the existing trained Random Forest
#           (model object is unchanged)
# ---------------------------------------------------------------
# Convert back to data.frame if the model expects one
cell_data <- as.data.frame(cell_dt)

# The trained RF model (rf_model) is used as-is — no retraining
cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Memory-Optimized Variant (if 38.5M-row expansion is tight on 16 GB)

If RAM is a concern, process one year at a time instead of expanding all years at once:

```r
add_neighbor_features_lowmem <- function(cell_dt, edge_dt, var_name) {
  years    <- sort(unique(cell_dt$year))
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  agg_list <- vector("list", length(years))

  for (j in seq_along(years)) {
    yr <- years[j]
    # Subset attribute for this year only
    attr_yr <- cell_dt[year == yr, .(id, value = get(var_name))]
    setkey(attr_yr, id)

    # Join neighbor values
    tmp <- copy(edge_dt)
    tmp[attr_yr, neighbor_val := i.value, on = .(neighbor_id = id)]

    agg_list[[j]] <- tmp[
      !is.na(neighbor_val),
      .(
        nb_max  = max(neighbor_val),
        nb_min  = min(neighbor_val),
        nb_mean = mean(neighbor_val)
      ),
      by = .(cell_id)
    ][, year := yr]
  }

  agg <- rbindlist(agg_list)
  setnames(agg, c("nb_max", "nb_min", "nb_mean"),
                c(max_col,  min_col,  mean_col))

  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }

  setkey(agg, cell_id, year)
  setkey(cell_dt, id, year)
  cell_dt <- agg[cell_dt, on = .(cell_id = id, year)]
  setnames(cell_dt, "cell_id", "id")
  cell_dt
}
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M string-paste + named-vector matches | 1.37M-row static edge table (built once) |
| **Stats computation** | R-level `lapply` over 6.46M list elements × 5 vars | Vectorized `data.table` join + `group-by` × 5 vars |
| **Estimated runtime** | ~86+ hours | **~5–15 minutes** |
| **RAM peak** | Large (6.46M-element list of int vectors) | ~2–4 GB (38.5M-row intermediate table) |
| **RF model** | Unchanged | Unchanged — predict only, no retraining |
| **Numerical results** | Original max/min/mean | Identical — same arithmetic, same neighbor topology |