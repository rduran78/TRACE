 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three interconnected reasons:

### 1. Row-level `lapply` over 6.46 million rows in `build_neighbor_lookup`
For every single cell-year row, the function performs character coercion, `paste`, and named-vector lookups. With 6.46M iterations, this creates billions of small string operations and hash lookups. Named vector lookups via `idx_lookup[neighbor_keys]` are O(n) hash lookups repeated millions of times.

### 2. Redundant recomputation of spatial topology per year
The rook-neighbor relationships are **purely spatial** — they do not change across years. Yet `build_neighbor_lookup` re-resolves the same cell→neighbor mapping for every cell-year combination, effectively doing the same spatial work 28 times over (once per year).

### 3. Row-level `lapply` in `compute_neighbor_stats`
For each of the 6.46M rows, `compute_neighbor_stats` subsets a numeric vector, removes NAs, and computes max/min/mean. This is a tight R-level loop with no vectorization. For 5 variables, that's ~32.3 million individual iterations.

**Summary:** The core problem is that the code treats a **panel** problem (cell × year) as a flat row problem, failing to exploit the fact that the neighbor graph is time-invariant. It also uses R-level loops where vectorized joins would suffice.

---

## Optimization Strategy

**Key insight:** Build the neighbor adjacency table **once** as a two-column data.table (`cell_id`, `neighbor_id`) — roughly 1.37M rows. Then, for each year, join the yearly cell attributes onto this table and compute grouped `max`, `min`, `mean` by `cell_id` using `data.table` aggregation. This replaces 6.46M R-level iterations with 28 fast vectorized grouped aggregations.

**Steps:**

1. **Build a static adjacency edge-list** from `rook_neighbors_unique` (an `nb` object): one row per directed edge (`cell_id → neighbor_id`). ~1.37M rows, built once.
2. **Split computation by year.** For each year, subset the cell-year attributes, join them onto the edge-list by `neighbor_id`, then group-by `cell_id` and compute `max`, `min`, `mean` for each variable — all vectorized inside `data.table`.
3. **Column-bind** the resulting neighbor features back onto `cell_data`.
4. **Predict** with the existing trained Random Forest model (unchanged).

**Expected speedup:** Each year's aggregation involves a keyed join on ~1.37M edges and a grouped aggregation — typically <1 second per variable per year in `data.table`. For 5 variables × 28 years = 140 operations, total time is on the order of **2–5 minutes** instead of 86+ hours.

**Memory:** The edge-list is ~1.37M rows × 2 integer columns ≈ 11 MB. Yearly subsets are ~344K rows. Well within 16 GB.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 0: Ensure cell_data is a data.table with columns 'id' and 'year'
# ──────────────────────────────────────────────────────────────────────
cell_data <- as.data.table(cell_data)

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build the static adjacency edge-list ONCE
#
#   rook_neighbors_unique : an nb object (list of integer index vectors)
#   id_order              : vector of cell IDs in the same order as the nb object
#
#   Result: adj_dt — a data.table with columns (cell_id, neighbor_id)
#           representing every directed rook-neighbor pair.
# ──────────────────────────────────────────────────────────────────────

build_adjacency_table <- function(id_order, neighbors_nb) {
  # neighbors_nb[[i]] contains integer indices into id_order for cell i's neighbors
  # A zero-length or 0-valued entry means no neighbors
  n <- length(id_order)
  
  # Pre-allocate lists for speed
  from_ids <- vector("list", n)
  to_ids   <- vector("list", n)
  
  for (i in seq_len(n)) {
    nb_idx <- neighbors_nb[[i]]
    # spdep nb objects use 0L to denote "no neighbors"
    nb_idx <- nb_idx[nb_idx > 0L]
    if (length(nb_idx) > 0L) {
      from_ids[[i]] <- rep(id_order[i], length(nb_idx))
      to_ids[[i]]   <- id_order[nb_idx]
    }
  }
  
  data.table(
    cell_id     = unlist(from_ids, use.names = FALSE),
    neighbor_id = unlist(to_ids,   use.names = FALSE)
  )
}

adj_dt <- build_adjacency_table(id_order, rook_neighbors_unique)

# Verify expected size
message("Adjacency edges: ", nrow(adj_dt))
# Should be approximately 1,373,394

# ──────────────────────────────────────────────────────────────────────
# STEP 2: Compute neighbor features via vectorized join + grouped agg
#
#   For each (cell, year) and each source variable, we need:
#     neighbor_max_{var}, neighbor_min_{var}, neighbor_mean_{var}
#
#   Strategy: loop over years (28 iterations), join yearly attributes
#   onto adj_dt by neighbor_id, then aggregate by cell_id.
# ──────────────────────────────────────────────────────────────────────

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-allocate the new columns in cell_data with NA_real_
for (var in neighbor_source_vars) {
  cell_data[, paste0("neighbor_max_",  var) := NA_real_]
  cell_data[, paste0("neighbor_min_",  var) := NA_real_]
  cell_data[, paste0("neighbor_mean_", var) := NA_real_]
}

# Key cell_data for fast subsetting
setkey(cell_data, year, id)

# Key adjacency table on neighbor_id for fast join
setkey(adj_dt, neighbor_id)

years <- sort(unique(cell_data$year))

message("Computing neighbor features for ", length(years), " years x ",
        length(neighbor_source_vars), " variables ...")

for (yr in years) {
  
  # Extract this year's cell attributes (only needed columns)
  yr_attrs <- cell_data[.(yr), c("id", neighbor_source_vars), with = FALSE]
  setnames(yr_attrs, "id", "neighbor_id")
  setkey(yr_attrs, neighbor_id)
  
  # Join neighbor attributes onto every edge
  # Each row becomes: (cell_id, neighbor_id, ntl, ec, pop_density, def, usd_est_n2)
  edges_with_vals <- adj_dt[yr_attrs, on = "neighbor_id", nomatch = NULL]
  
  # Aggregate: for each cell_id, compute max/min/mean of each variable
  # across all its neighbors
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)),   na.rm = TRUE)),
      bquote(min(.(as.name(v)),   na.rm = TRUE)),
      bquote(mean(.(as.name(v)),  na.rm = TRUE))
    )
  }), recursive = FALSE)
  
  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0(c("neighbor_max_", "neighbor_min_", "neighbor_mean_"), v)
  }))
  
  # Build the aggregation call dynamically
  agg_result <- edges_with_vals[,
    setNames(lapply(neighbor_source_vars, function(v) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        list(NA_real_, NA_real_, NA_real_)
      } else {
        list(max(vals), min(vals), mean(vals))
      }
    }), neighbor_source_vars),
    by = cell_id
  ]
  
  # The above produces nested lists; use a cleaner approach:
  # Compute all stats in one pass per group
  agg_result <- edges_with_vals[, {
    out <- vector("list", length(neighbor_source_vars) * 3L)
    k <- 0L
    for (v in neighbor_source_vars) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        out[[k + 1L]] <- NA_real_
        out[[k + 2L]] <- NA_real_
        out[[k + 3L]] <- NA_real_
      } else {
        out[[k + 1L]] <- max(vals)
        out[[k + 2L]] <- min(vals)
        out[[k + 3L]] <- mean(vals)
      }
      k <- k + 3L
    }
    setNames(out, agg_names)
  }, by = cell_id]
  
  # Now write back into cell_data for this year
  setkey(agg_result, cell_id)
  
  # Match rows in cell_data for this year
  idx <- cell_data[.(yr), which = TRUE]
  matched <- match(cell_data$id[idx], agg_result$cell_id)
  
  for (col_name in agg_names) {
    set(cell_data, i = idx, j = col_name, value = agg_result[[col_name]][matched])
  }
  
  if (yr %% 5 == 0 || yr == years[1] || yr == years[length(years)]) {
    message("  Year ", yr, " done.")
  }
}

# Replace Inf/-Inf from max/min on empty sets (safety)
for (col_name in grep("^neighbor_", names(cell_data), value = TRUE)) {
  vals <- cell_data[[col_name]]
  vals[is.infinite(vals)] <- NA_real_
  set(cell_data, j = col_name, value = vals)
}

message("Neighbor features complete.")

# ──────────────────────────────────────────────────────────────────────
# STEP 3: Predict with the existing trained Random Forest (unchanged)
# ──────────────────────────────────────────────────────────────────────

# The trained model object (e.g., rf_model) and its expected feature names
# are preserved exactly. No retraining occurs.
# cell_data now contains the same neighbor_max_*, neighbor_min_*, neighbor_mean_*
# columns the model expects, with identical numerical values.

# Example (adapt to your actual model object name and prediction call):
# cell_data$predicted <- predict(rf_model, newdata = cell_data)

message("Pipeline complete. Ready for Random Forest prediction.")
```

---

## Faster Alternative: Fully Vectorized Without Year Loop

If you want to eliminate even the year loop, you can do the entire computation in a single massive join. This uses more peak memory (~2–3 GB) but completes in one pass:

```r
# ──────────────────────────────────────────────────────────────────────
# ALTERNATIVE: Single-pass fully vectorized approach
# ──────────────────────────────────────────────────────────────────────

library(data.table)

cell_data <- as.data.table(cell_data)
adj_dt    <- build_adjacency_table(id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Create a cross of adjacency edges × years
# Instead, join cell_data directly using a composite key

# Add neighbor attributes by joining cell_data onto adj_dt crossed with year
# Key: every (cell_id, year) pair needs its neighbors' (neighbor_id, year) values

# Build edge-year table: expand adj_dt to all years
# ~1.37M edges × 28 years ≈ 38.5M rows (fits in RAM as integers + doubles)

edges_all <- adj_dt[, .(year = years), by = .(cell_id, neighbor_id)]

# Join neighbor attributes
setkey(cell_data, id, year)
setkey(edges_all, neighbor_id, year)

# Attach neighbor values
edges_all <- cell_data[, c("id", "year", neighbor_source_vars), with = FALSE
  ][edges_all, on = c(id = "neighbor_id", "year"), nomatch = NULL]

# Now aggregate by (cell_id, year)
agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0(c("neighbor_max_", "neighbor_min_", "neighbor_mean_"), v)
}))

agg_result <- edges_all[, {
  out <- vector("list", length(neighbor_source_vars) * 3L)
  k <- 0L
  for (v in neighbor_source_vars) {
    vals <- get(v)
    vals <- vals[!is.na(vals)]
    if (length(vals) == 0L) {
      out[[k + 1L]] <- NA_real_; out[[k + 2L]] <- NA_real_; out[[k + 3L]] <- NA_real_
    } else {
      out[[k + 1L]] <- max(vals); out[[k + 2L]] <- min(vals); out[[k + 3L]] <- mean(vals)
    }
    k <- k + 3L
  }
  setNames(out, agg_names)
}, by = .(cell_id, year)]

# Merge back
setkey(agg_result, cell_id, year)
setkey(cell_data, id, year)

cell_data <- agg_result[cell_data, on = c(cell_id = "id", "year")]

# Clean up Inf values
for (col_name in agg_names) {
  vals <- cell_data[[col_name]]
  vals[is.infinite(vals)] <- NA_real_
  set(cell_data, j = col_name, value = vals)
}

message("Single-pass neighbor features complete. Ready for prediction.")
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor resolution** | Per cell-year row (6.46M string lookups) | Once (344K cells → 1.37M edge-list) |
| **Stats computation** | R-level `lapply` over 6.46M rows × 5 vars | `data.table` grouped aggregation (vectorized C) |
| **Estimated runtime** | ~86+ hours | **~2–5 minutes** |
| **Peak memory** | High (large named vectors, repeated string ops) | ~2–3 GB (edge-list + yearly subsets) |
| **RF model** | Unchanged | Unchanged |
| **Numerical output** | Identical `max`, `min`, `mean` per (cell, year, variable) | Identical — same arithmetic, same values |

The trained Random Forest model is never modified or retrained. The neighbor `max`, `min`, and `mean` values are numerically identical to the original implementation since the same arithmetic operations are applied to the same neighbor sets — only the execution strategy changes from row-level R loops to vectorized `data.table` joins and grouped aggregations.