 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` — O(n) list of per-row lookups via `lapply`

- With ~6.46 million rows, this creates a list of 6.46M elements. Each iteration does string pasting, hash lookups in `idx_lookup`, and subsetting. The string-key approach (`paste(id, year, sep="_")`) is extremely expensive at this scale — it allocates millions of temporary character vectors and performs millions of named-vector lookups (which are O(n) in base R named vectors, not O(1) hash lookups).

### Bottleneck B: `compute_neighbor_stats` — per-row `lapply` over 6.46M entries

- For each of the 5 variables, another `lapply` iterates over 6.46M elements, subsetting a numeric vector by index, removing NAs, and computing `max`, `min`, `mean`. The `do.call(rbind, result)` on a 6.46M-element list of 3-element vectors is also very slow (repeated row-binding).

### Why 86+ hours?

- ~6.46M iterations × 2 passes (build + compute) × 5 variables for compute = ~38.8M R-level loop iterations, each with allocation-heavy string operations or subsetting. The named-vector lookup `idx_lookup[neighbor_keys]` is the single worst offender — it's a linear scan per lookup in base R.

### Why raster focal/kernel operations don't directly apply

The comment in the prompt asks whether raster focal operations are a useful analogy. They are conceptually analogous (computing neighborhood statistics), but the data is in **long panel format** (cell × year), not a raster stack. Converting to raster, applying `focal()` per year per variable, and converting back would require reshaping 28 years × 5 variables = 140 focal operations on 344K-cell rasters. This is feasible but introduces complexity around NA handling and irregular grid boundaries. The more direct and faithful optimization is to **vectorize the neighbor computation using sparse matrix multiplication and grouped operations**, which preserves the exact numerical results.

---

## 2. Optimization Strategy

### Step 1: Replace string-keyed lookup with integer indexing via `data.table`

Use `data.table` keyed joins to map `(id, year)` → row index in O(1) amortized time.

### Step 2: Build a sparse adjacency matrix (cell-level), then expand to cell-year level

- The rook neighbor structure is **time-invariant**: cell *i*'s neighbors are the same in every year. So we build a 344,208 × 344,208 sparse adjacency matrix **once**, then use it to compute all neighbor statistics via sparse matrix–vector products.
- For `mean`: neighbor mean of variable `x` for cell `i` = `(A %*% x) / (A %*% 1)` where `A` is the adjacency matrix (applied within each year).
- For `max` and `min`: sparse matrix multiplication gives the sum, not max/min. We handle these with a **grouped operation** using `data.table` — explode the neighbor pairs, join the variable values, and compute `max`/`min`/`mean` grouped by `(id, year)`.

### Step 3: Vectorized grouped computation with `data.table`

- Build an edge table: `(id, neighbor_id)` from the `nb` object (~1.37M directed edges).
- Cross-join with years to get `(id, year, neighbor_id)` — ~1.37M × 28 = ~38.5M rows.
- Join neighbor variable values from the main data.
- Group by `(id, year)`, compute `max`, `min`, `mean`.
- Join results back to the main data.

This replaces 6.46M R-level iterations with a single vectorized `data.table` grouped aggregation. Expected runtime: **minutes, not hours**.

### Memory check

- Edge table expanded: ~38.5M rows × 3 integer columns ≈ 460 MB
- With one double-precision variable column joined: +308 MB
- Total working memory per variable: ~800 MB — fits in 16 GB RAM.

---

## 3. Working R Code

```r
library(data.table)
library(Matrix)

# ──────────────────────────────────────────────────────────────────────
# STEP 0: Convert main data to data.table and key it
# ──────────────────────────────────────────────────────────────────────
cell_dt <- as.data.table(cell_data)
cell_dt[, row_idx := .I]
setkey(cell_dt, id, year)

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build the directed edge list from the nb object (time-invariant)
#
#   rook_neighbors_unique is an nb object of length 344,208.
#   id_order is the vector mapping position in the nb list → cell id.
# ──────────────────────────────────────────────────────────────────────
build_edge_list <- function(id_order, nb_obj) {
  # nb objects store neighbor indices as integer vectors; 0L means no neighbors
  from_ids <- rep(id_order, times = lengths(nb_obj))
  to_positions <- unlist(nb_obj, use.names = FALSE)

  # spdep uses 0L (integer 0) for cells with no neighbors — remove those
  valid <- to_positions > 0L
  from_ids <- from_ids[valid]
  to_ids <- id_order[to_positions[valid]]

  data.table(id = from_ids, neighbor_id = to_ids)
}

edges <- build_edge_list(id_order, rook_neighbors_unique)
cat("Edge list rows:", nrow(edges), "\n")
# Expected: ~1,373,394

# ──────────────────────────────────────────────────────────────────────
# STEP 2: Expand edges across all years and compute neighbor stats
# ──────────────────────────────────────────────────────────────────────
years <- sort(unique(cell_dt$year))  # 1992:2019

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-build the (id, year, neighbor_id) table once — ~38.5M rows
# To save memory, we process one year at a time inside the variable loop.

# Prepare a keyed lookup table for joining variable values
# We'll subset columns as needed.

compute_and_add_all_neighbor_features <- function(cell_dt, edges, years,
                                                   neighbor_source_vars) {
  # For each variable, process all years in a single vectorized operation
  setkey(cell_dt, id, year)

  for (var_name in neighbor_source_vars) {
    cat("Processing neighbor features for:", var_name, "\n")
    t0 <- proc.time()

    # Extract only the columns we need for the join
    val_dt <- cell_dt[, .(id, year, val = get(var_name))]
    setkey(val_dt, id, year)

    # Process in yearly chunks to limit peak memory (~1.37M edges per year)
    result_list <- vector("list", length(years))

    for (yi in seq_along(years)) {
      yr <- years[yi]

      # All edges for this year: join neighbor values
      # edges has (id, neighbor_id); we need val for each neighbor_id in this year
      yr_edges <- copy(edges)
      yr_edges[, year := yr]

      # Join to get neighbor's variable value
      setkey(yr_edges, neighbor_id, year)
      yr_edges[val_dt, neighbor_val := i.val, on = .(neighbor_id = id, year)]

      # Compute grouped stats: max, min, mean (excluding NAs)
      stats <- yr_edges[!is.na(neighbor_val),
                         .(nb_max = max(neighbor_val),
                           nb_min = min(neighbor_val),
                           nb_mean = mean(neighbor_val)),
                         by = .(id)]
      stats[, year := yr]
      result_list[[yi]] <- stats
    }

    # Combine all years
    all_stats <- rbindlist(result_list, use.names = TRUE)

    # Rename columns to match expected output names
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")
    setnames(all_stats, c("nb_max", "nb_min", "nb_mean"),
             c(max_col, min_col, mean_col))

    # Join back to main data
    setkey(all_stats, id, year)
    setkey(cell_dt, id, year)

    # Remove old columns if they exist (for idempotency)
    for (col in c(max_col, min_col, mean_col)) {
      if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
    }

    cell_dt <- all_stats[cell_dt, on = .(id, year)]

    elapsed <- (proc.time() - t0)[3]
    cat("  Done in", round(elapsed, 1), "seconds\n")
  }

  return(cell_dt)
}

cell_dt <- compute_and_add_all_neighbor_features(
  cell_dt, edges, years, neighbor_source_vars
)

# ──────────────────────────────────────────────────────────────────────
# STEP 3: Convert back to data.frame if needed for the RF predict step
# ──────────────────────────────────────────────────────────────────────
cell_dt[, row_idx := NULL]  # clean up helper column
cell_data <- as.data.frame(cell_dt)

# ──────────────────────────────────────────────────────────────────────
# STEP 4: Predict with the pre-trained Random Forest (unchanged)
# ──────────────────────────────────────────────────────────────────────
# The trained model object (e.g., `rf_model`) is used as-is:
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## 4. Further Optimization: Fully Vectorized (No Year Loop)

If memory permits (~2–3 GB working set), you can eliminate the year loop entirely:

```r
compute_neighbor_features_fully_vectorized <- function(cell_dt, edges,
                                                        neighbor_source_vars) {
  setkey(cell_dt, id, year)

  # Expand edges × years in one shot
  yr_dt <- CJ(year = sort(unique(cell_dt$year)),
               edge_idx = seq_len(nrow(edges)))
  yr_dt[, `:=`(id = edges$id[edge_idx],
               neighbor_id = edges$neighbor_id[edge_idx])]
  yr_dt[, edge_idx := NULL]
  # ~38.5M rows

  for (var_name in neighbor_source_vars) {
    cat("Processing:", var_name, "\n")
    t0 <- proc.time()

    val_dt <- cell_dt[, .(id, year, val = get(var_name))]
    setkey(val_dt, id, year)

    # Join neighbor values
    setkey(yr_dt, neighbor_id, year)
    yr_dt[val_dt, neighbor_val := i.val, on = .(neighbor_id = id, year)]

    # Grouped aggregation
    stats <- yr_dt[!is.na(neighbor_val),
                    .(nb_max  = max(neighbor_val),
                      nb_min  = min(neighbor_val),
                      nb_mean = mean(neighbor_val)),
                    by = .(id, year)]

    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")
    setnames(stats, c("nb_max", "nb_min", "nb_mean"),
             c(max_col, min_col, mean_col))

    setkey(stats, id, year)
    for (col in c(max_col, min_col, mean_col)) {
      if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
    }
    cell_dt <- stats[cell_dt, on = .(id, year)]

    yr_dt[, neighbor_val := NULL]  # free for next variable

    cat("  Done in", round((proc.time() - t0)[3], 1), "s\n")
  }

  return(cell_dt)
}
```

---

## 5. Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M string-paste + named-vector lookups | Integer edge list from `nb` object, O(n) |
| **Neighbor stats** | 6.46M `lapply` iterations × 5 vars | Vectorized `data.table` grouped aggregation |
| **`do.call(rbind, ...)`** | 6.46M-element list row-bind | `rbindlist` (C-level) |
| **Estimated runtime** | 86+ hours | **5–15 minutes** |
| **Peak memory** | Moderate (but slow) | ~2–4 GB (fits in 16 GB) |
| **Numerical results** | `max`, `min`, `mean` of rook neighbors | **Identical** — same operations, same NA handling |
| **RF model** | N/A | **Unchanged — no retraining** |

The key insight: the neighbor structure is **time-invariant**, so the expensive per-row string-key lookup is entirely unnecessary. By expressing the problem as a join between an edge table and the panel data, followed by a grouped aggregation, we replace millions of R-level loop iterations with a handful of vectorized C-level `data.table` operations.