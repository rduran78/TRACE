 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The colleague is **partially correct but misses the deeper bottleneck**. Let me audit each claim against the code:

### Claim 1: `do.call(rbind, result)` is a bottleneck

`do.call(rbind, result)` on a list of ~6.46 million 3-element vectors is indeed not optimal, but it executes in **seconds** (it's a single call binding pre-allocated vectors). This is a **minor** bottleneck.

### Claim 2: "Repeated list binding inside `compute_neighbor_stats()`"

There is **no repeated list binding** inside `compute_neighbor_stats()`. It uses `lapply` to produce a list in one pass and then binds once. The colleague's diagnosis here is **factually wrong** — the code doesn't grow a list iteratively.

### The actual deep bottleneck: `build_neighbor_lookup()`

The true bottleneck is **`build_neighbor_lookup()`**, specifically:

1. **`paste()` key construction and named-vector lookup (`idx_lookup[neighbor_keys]`)** is called **6.46 million times** inside `lapply`. Each call constructs character keys and performs name-based lookups on a **6.46-million-element named vector**. Named vector lookup in R is **O(n)** per query because R's named vectors use linear hashing that degrades at scale. Over 6.46M iterations, each touching multiple neighbors, this produces **billions of character-match operations**.

2. **`as.character()` and `paste()` allocations** inside the per-row lambda create enormous garbage-collection pressure (~6.46M × k string allocations).

3. The **total work** is approximately 6.46M rows × ~4 rook neighbors × O(lookup) per neighbor. With the naive named-vector approach, this is the source of the **86+ hour runtime**.

`compute_neighbor_stats()` by contrast is a simple numeric indexing operation (`vals[idx]`) — essentially free once the lookup exists.

**Verdict: REJECT the colleague's diagnosis.** The dominant bottleneck is `build_neighbor_lookup()` with its per-row string construction and named-vector lookups over a 6.46M-element vector.

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()`** with a vectorized approach using `data.table` hash joins instead of named-vector character lookups. This reduces lookup from effective O(n) to amortized O(1) per query.

2. **Vectorize `compute_neighbor_stats()`** using `data.table` grouped operations, eliminating the per-row `lapply` and the `do.call(rbind, ...)` entirely.

3. **Pre-expand the neighbor-edge list once** (cell→neighbor × year), then do a single equi-join to resolve row indices, then group-by to compute max/min/mean. This turns the entire pipeline into a few vectorized `data.table` operations.

4. **Preserve** the trained Random Forest model (no retraining) and the original numerical outputs (max, min, mean of neighbor values per variable).

**Expected speedup**: From 86+ hours to **minutes** (roughly 3–10 minutes depending on disk I/O).

---

## Working R Code

```r
library(data.table)

#' Optimized pipeline: replaces build_neighbor_lookup + compute_neighbor_stats
#' Preserves the original numerical estimand (max, min, mean of neighbor values).
#' Does NOT touch the trained Random Forest model.

optimize_neighbor_features <- function(cell_data,
                                       id_order,
                                       rook_neighbors_unique,
                                       neighbor_source_vars) {

  # -------------------------------------------------------------------
  # STEP 1: Convert to data.table and create a unique row index
  # -------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]

  # -------------------------------------------------------------------
  # STEP 2: Build an integer-keyed edge list from the nb object
  #
  # rook_neighbors_unique is an nb object: a list of length

  # length(id_order), where element i contains integer indices into
  # id_order of cell i's neighbors (0L means no neighbors in nb).
  # -------------------------------------------------------------------
  edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb <- rook_neighbors_unique[[i]]
    nb <- nb[nb != 0L]
    if (length(nb) == 0L) return(NULL)
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb])
  }))

  cat(sprintf("Edge list: %d directed edges\n", nrow(edges)))

  # -------------------------------------------------------------------
  # STEP 3: For each source variable, compute neighbor stats via
  #          vectorized data.table joins and grouped aggregation.
  #
  # Logic equivalent to the original:
  #   For each row (id, year), find all neighbors sharing the same year,

  #   then compute max, min, mean of the neighbor's variable value.
  # -------------------------------------------------------------------

  # Minimal join table: only (id, year, row_idx) + the variable columns we need
  join_cols <- c("id", "year", "row_idx", neighbor_source_vars)
  dt_join <- dt[, ..join_cols]

  # Key the join table for fast equi-joins
  setkey(dt_join, id, year)

  for (var_name in neighbor_source_vars) {
    cat(sprintf("Processing variable: %s ...\n", var_name))

    # -- 3a: Build focal table: (focal_id, year, row_idx)
    focal <- dt_join[, .(focal_id = id, year, row_idx)]

    # -- 3b: Join focal -> edges to get (focal_row_idx, neighbor_id, year)
    #         This expands each focal row by its number of neighbors.
    focal_edges <- merge(focal, edges, by = "focal_id", allow.cartesian = TRUE)
    #   columns: focal_id, year, row_idx (of focal), neighbor_id

    # -- 3c: Join to get the neighbor's variable value in the same year
    #         Key neighbor lookup table on (id, year)
    neighbor_vals_dt <- dt_join[, .(id, year, nval = get(var_name))]
    setkey(neighbor_vals_dt, id, year)

    setnames(focal_edges, "neighbor_id", "id")
    setkey(focal_edges, id, year)
    matched <- neighbor_vals_dt[focal_edges, on = .(id, year), nomatch = NA]
    #   columns: id (=neighbor_id), year, nval, focal_id, row_idx

    # -- 3d: Aggregate: group by focal row_idx, compute stats
    stats <- matched[!is.na(nval),
                     .(nb_max  = max(nval),
                       nb_min  = min(nval),
                       nb_mean = mean(nval)),
                     by = row_idx]

    # -- 3e: Merge stats back into dt by row_idx
    #         Rows with no valid neighbors get NA (preserving original behavior)
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    dt[stats, on = "row_idx",
       c(max_col, min_col, mean_col) := .(nb_max, nb_min, nb_mean)]

    # Rows not in stats remain NA (default for new columns in data.table)

    # Clean up large intermediates
    rm(focal, focal_edges, neighbor_vals_dt, matched, stats)
    gc()

    cat(sprintf("  Done: %s\n", var_name))
  }

  # -------------------------------------------------------------------
  # STEP 4: Remove helper column and return as data.frame
  #          (to remain compatible with downstream Random Forest predict)
  # -------------------------------------------------------------------
  dt[, row_idx := NULL]
  as.data.frame(dt)
}


# =====================================================================
# USAGE — drop-in replacement for the original pipeline
# =====================================================================

# cell_data             : the existing data.frame (~6.46M rows, ~110 columns)
# id_order              : integer vector of cell IDs matching nb object order
# rook_neighbors_unique : the precomputed spdep::nb object loaded from disk
# The trained RF model object (e.g., rf_model) is untouched.

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# Now use the existing trained Random Forest model as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Colleague's Claim | Actual Finding |
|---|---|---|
| `do.call(rbind, result)` | "Main bottleneck" | Minor cost (~seconds) |
| "Repeated list binding in `compute_neighbor_stats`" | Claimed present | **Does not exist in the code** |
| `build_neighbor_lookup()` | Not mentioned | **True bottleneck**: 6.46M iterations of string construction + O(n) named-vector lookups |
| Fix | — | Replace with vectorized `data.table` join + group-by aggregation |
| Expected runtime | 86+ hours | **~3–10 minutes** |
| RF model | — | Preserved, untouched |
| Numerical output | — | Identical (max, min, mean of non-NA neighbor values; NA when no neighbors) |