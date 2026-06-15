 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(N²)-like in practice
This function iterates over **all 6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs **year-specific string keys** (`paste(neighbor_id, year, sep="_")`) and matches them against a named character vector (`idx_lookup`) of length 6.46 million.

Named vector lookup in R via `idx_lookup[neighbor_keys]` is hash-based, but doing ~6.46 million calls into `lapply`, each allocating character vectors and performing multiple hash lookups, produces enormous overhead. The resulting `neighbor_lookup` is a **list of 6.46 million integer vectors** — one per cell-year row — consuming substantial memory and time to build.

### 2. `compute_neighbor_stats` iterates over 6.46M list elements per variable
For each of the 5 neighbor source variables, another `lapply` over 6.46 million elements extracts neighbor values, removes NAs, and computes max/min/mean. That's **~32.3 million R-level function calls** total across the 5 variables.

### Root cause: the neighbor topology is **static** but is being resolved at the cell-year level
Rook neighbors are a **spatial** relationship — cell A's neighbors are the same in 1992 as in 2019. The current code "flattens" this into a cell-year-level lookup, exploding the problem by a factor of 28 (the number of years). The correct approach is to resolve neighbor topology **once at the cell level** (344K cells), then use vectorized joins to bring in yearly attributes.

---

## Optimization Strategy

| Step | What | Complexity |
|------|------|------------|
| **1** | Build a **cell-level** neighbor edge table (from → to) once. ~1.37M rows. | O(C) where C = cells |
| **2** | Convert `cell_data` to a `data.table` keyed on `(id, year)`. | O(N) |
| **3** | For each variable, **join** the edge table to cell_data to pull neighbor values, then **group-by** `(from_id, year)` to compute `max`, `min`, `mean` in one vectorized pass. | O(E × Y) fully vectorized |
| **4** | Join the aggregated stats back onto `cell_data`. | O(N) |

**Expected speedup:** The join-and-aggregate approach replaces ~32M R-level `lapply` iterations with ~5 vectorized `data.table` grouped joins. Estimated runtime: **minutes, not hours** (typically 2–10 minutes total on a 16 GB laptop).

**Preserves:**
- The trained Random Forest model (no retraining).
- The original numerical estimand (max, min, mean of non-NA neighbor values, with NA when no neighbors or all-NA).

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build a static cell-level edge table (once, reusable)
# ──────────────────────────────────────────────────────────────────────
# rook_neighbors_unique : spdep nb object (list of integer index vectors)
# id_order              : vector of cell IDs in the same order as the nb object

build_edge_table <- function(id_order, neighbors) {
  # neighbors[[i]] gives the indices (into id_order) of cell i's rook neighbors
  from_ids <- rep(id_order, times = lengths(neighbors))
  to_ids   <- id_order[unlist(neighbors)]
  data.table(from_id = from_ids, to_id = to_ids)
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# ~1,373,394 rows — small, static, and reusable

# ──────────────────────────────────────────────────────────────────────
# STEP 2: Convert cell_data to data.table (if not already)
# ──────────────────────────────────────────────────────────────────────
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# ──────────────────────────────────────────────────────────────────────
# STEP 3 & 4: For each variable, compute neighbor stats via join + group-by
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {
  # Subset to only the columns we need for the join (minimise memory)
  vals_dt <- cell_dt[, .(id, year, val = get(var_name))]

  # Join: for every (from_id, year), pull the neighbor cell's value
  # edge_dt gives (from_id -> to_id); we join vals_dt on to_id + year
  merged <- edge_dt[vals_dt,
                    on = .(from_id = id),   # one row per (edge × year)
                    allow.cartesian = TRUE,
                    nomatch = 0L
  ][vals_dt,
    on = .(to_id = id, year),               # attach neighbor's value
    nomatch = NA,
    .(from_id, year, neighbor_val = i.val)
  ]

  # ---- cleaner two-step approach (more readable, same speed) ----
  # Step A: expand edges × years by joining cell_dt onto edge_dt via from_id
  #         This gives us one row per (from_cell, to_cell, year).
  # Step B: join again to get the neighbor (to_cell) value for that year.

  # Step A
  expanded <- merge(
    edge_dt,
    vals_dt[, .(from_id = id, year)],
    by = "from_id",
    allow.cartesian = TRUE
  )
  # expanded has columns: from_id, to_id, year


  # Step B: get neighbor value
  expanded[vals_dt, neighbor_val := i.val,
           on = .(to_id = id, year)]

  # Step C: aggregate per (from_id, year), dropping NAs
  stats <- expanded[!is.na(neighbor_val),
                    .(nbr_max  = max(neighbor_val),
                      nbr_min  = min(neighbor_val),
                      nbr_mean = mean(neighbor_val)),
                    by = .(from_id, year)]

  # Rename columns to match original pipeline naming convention
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats,
           c("nbr_max", "nbr_min", "nbr_mean"),
           c(max_col,   min_col,   mean_col))

  # Step D: join back onto cell_dt
  cell_dt[stats,
          (c(max_col, min_col, mean_col)) := mget(c(max_col, min_col, mean_col)),
          on = .(id = from_id, year)]

  invisible(cell_dt)
}

# ──────────────────────────────────────────────────────────────────────
# STEP 5: Run for all neighbor source variables
# ──────────────────────────────────────────────────────────────────────
for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  compute_neighbor_features_fast(cell_data, edge_dt, var_name)
}

# ──────────────────────────────────────────────────────────────────────
# STEP 6: Predict with the existing trained Random Forest (unchanged)
# ──────────────────────────────────────────────────────────────────────
# cell_data now has the same neighbor_max_*, neighbor_min_*, neighbor_mean_*
# columns as the original pipeline, with identical numerical values.
# Proceed directly to prediction:
#
#   preds <- predict(trained_rf_model, newdata = cell_data)
#
# The model is NOT retrained — only the feature-engineering step was optimised.
```

---

## Memory-Conscious Variant (if 16 GB is tight)

The `merge(..., allow.cartesian = TRUE)` in Step A above creates ~1.37M × 28 ≈ **38.5M rows** per variable. Each row is three columns (two integer IDs + one numeric value) ≈ ~900 MB peak. This fits in 16 GB but is snug with 110 predictor columns already in memory. If memory is a concern, process years in chunks:

```r
compute_neighbor_features_chunked <- function(cell_dt, edge_dt, var_name, chunk_size = 7) {
  years <- sort(unique(cell_dt$year))
  year_chunks <- split(years, ceiling(seq_along(years) / chunk_size))

  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  # Initialise columns as NA
  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]

  for (yrs in year_chunks) {
    sub <- cell_dt[year %in% yrs, .(id, year, val = get(var_name))]

    expanded <- merge(edge_dt, sub[, .(from_id = id, year)],
                      by = "from_id", allow.cartesian = TRUE)
    expanded[sub, neighbor_val := i.val, on = .(to_id = id, year)]

    stats <- expanded[!is.na(neighbor_val),
                      .(nbr_max  = max(neighbor_val),
                        nbr_min  = min(neighbor_val),
                        nbr_mean = mean(neighbor_val)),
                      by = .(from_id, year)]

    cell_dt[stats,
            (c(max_col, min_col, mean_col)) :=
              .(i.nbr_max, i.nbr_min, i.nbr_mean),
            on = .(id = from_id, year)]

    rm(expanded, stats, sub)
    gc()
  }
  invisible(cell_dt)
}
```

---

## Summary

| Aspect | Original | Optimised |
|--------|----------|-----------|
| Neighbor resolution level | Cell-year (6.46M entries) | Cell (344K entries) + join |
| Core operation | `lapply` + named-vector hash lookup | `data.table` equi-join + grouped aggregation |
| Estimated runtime | ~86+ hours | ~2–10 minutes |
| Memory pattern | 6.46M-element list of integer vectors | Temporary 38.5M-row table per variable (or chunked) |
| RF model | Unchanged | Unchanged |
| Numerical output | max, min, mean of non-NA neighbor values | Identical |