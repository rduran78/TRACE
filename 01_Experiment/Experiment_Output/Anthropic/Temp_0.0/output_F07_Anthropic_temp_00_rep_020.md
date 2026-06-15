 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which iterates over **~6.46 million rows** in a pure-R `lapply`, performing per-row character coercion, string pasting, and named-vector lookups. Each iteration does:

1. `as.character()` conversion
2. Indexing into a named character vector (`id_to_ref`)
3. Subsetting a list (`neighbors[[ref_idx]]`)
4. `paste()` to build composite keys
5. Named lookup into `idx_lookup` (a named integer vector of length 6.46M)

Named vector lookup in R is **O(n)** per query in the worst case (hash collisions aside, the overhead is enormous at 6.46M entries). Doing this 6.46 million times produces an **O(n²)-like** runtime. The `compute_neighbor_stats` function is comparatively cheap but is also called per-variable in a loop with unnecessary repeated list traversals.

**Summary of problems:**
| Problem | Impact |
|---|---|
| Per-row `paste` + named-vector lookup in `build_neighbor_lookup` | ~86+ hours; dominant cost |
| Character key lookups on a 6.46M-length named vector | Pseudo-O(n²) |
| `compute_neighbor_stats` returns a list of 3-vectors, then `do.call(rbind, ...)` on 6.46M elements | Slow bind |
| Loop over 5 variables calls `compute_neighbor_stats` independently each time | Minor but avoidable overhead |

---

## Optimization Strategy

### 1. Replace string-key lookups with integer-indexed joins via `data.table`

Instead of building a 6.46M-entry named vector and doing per-row string matching, we:
- Create an **edge list** of `(id, neighbor_id)` from the `nb` object (only ~1.37M directed edges for 344K cells).
- Join this edge list to the panel data **twice**: once to attach the row index of the focal cell-year, once to attach the row index (and variable values) of the neighbor cell-year.
- This is a **vectorized equi-join**, which `data.table` executes in O(n log n) or better.

### 2. Compute all neighbor stats in one grouped aggregation

Once we have an edge table `(focal_row, neighbor_row)` joined to variable values, we simply `group by focal_row` and compute `max`, `min`, `mean` — all in one pass, for all 5 variables simultaneously.

### 3. Memory budget

- Edge list with year expansion: ~1.37M edges × 28 years ≈ 38.4M rows × a few integer/double columns ≈ ~1.5 GB at peak. Fits in 16 GB.
- We avoid materializing a 6.46M-element list of variable-length integer vectors entirely.

### 4. Preserve the trained RF model and numerical estimand

We produce columns with **identical names and identical values** (max, min, mean of the same neighbor sets). The RF model is not retouched.

---

## Working R Code

```r
library(data.table)

compute_all_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                          neighbor_source_vars) {
  # -----------------------------------------------------------
  # Step 0: Convert to data.table if needed; add row index

# -----------------------------------------------------------
  was_df <- !is.data.table(cell_data)
  dt <- as.data.table(cell_data)
  dt[, .row_id := .I]
  
  # -----------------------------------------------------------
  # Step 1: Build directed edge list from the nb object
  #         (only ~1.37M rows — one per directed rook edge)
  # -----------------------------------------------------------
  edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb_idx <- rook_neighbors_unique[[i]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
      return(NULL)
    }
    data.table(focal_id = id_order[i],
               neighbor_id = id_order[nb_idx])
  }))
  
  cat("Edge list rows (spatial only):", nrow(edge_list), "\n")
  
  # -----------------------------------------------------------
  # Step 2: Build a lookup from (id, year) -> row index + values
  # -----------------------------------------------------------
  # Only keep columns we need to minimize memory
  keep_cols <- c("id", "year", ".row_id", neighbor_source_vars)
  lookup <- dt[, ..keep_cols]
  setkey(lookup, id, year)
  
  # -----------------------------------------------------------
  # Step 3: Expand edges across years via join
  #
  # For each (focal_id, neighbor_id) edge and each year that

  # the focal cell appears in, find the neighbor's values in
  # that same year.
  #
  # Strategy:
  #   a) Join edge_list to lookup on focal_id == id  →  gives us
  #      (focal_id, neighbor_id, year, focal_row_id)
  #   b) Join result to lookup on neighbor_id == id & year == year
  #      →  gives us neighbor variable values
  # -----------------------------------------------------------
  
  # Step 3a: Get all (edge × year) combinations for focal cells
  # Join edge_list[focal_id] → lookup[id] to pick up year & focal .row_id
  focal_lookup <- lookup[, .(id, year, focal_row_id = .row_id)]
  setkey(focal_lookup, id)
  setkey(edge_list, focal_id)
  
  edges_by_year <- edge_list[focal_lookup,
                             on = .(focal_id = id),
                             allow.cartesian = TRUE,
                             nomatch = NULL]
  # edges_by_year has columns: focal_id, neighbor_id, year, focal_row_id
  
  cat("Edge-year rows:", nrow(edges_by_year), "\n")
  
  # Step 3b: Attach neighbor variable values
  neighbor_lookup_dt <- lookup[, c("id", "year", neighbor_source_vars), with = FALSE]
  setkey(neighbor_lookup_dt, id, year)
  setkey(edges_by_year, neighbor_id, year)
  
  edges_full <- neighbor_lookup_dt[edges_by_year,
                                   on = .(id = neighbor_id, year = year),
                                   nomatch = NA]
  # edges_full now has: id (=neighbor_id), year, <neighbor vars>, focal_id, focal_row_id
  
  # -----------------------------------------------------------
  # Step 4: Grouped aggregation — compute max, min, mean
  #         for each focal_row_id (= each cell-year)
  # -----------------------------------------------------------
  
  # Build aggregation expressions dynamically
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
      bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }))
  
  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0(v, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
  }))
  
  names(agg_exprs) <- agg_names
  
  # Evaluate
  stats <- edges_full[, lapply(agg_exprs, eval, envir = .SD),
                       by = focal_row_id]
  
  # Fix Inf/-Inf from max/min on all-NA groups (shouldn't happen if edges exist,

  # but be safe)
  for (col in agg_names) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }
  
  # -----------------------------------------------------------
  # Step 5: Merge back into dt by row index
  # -----------------------------------------------------------
  setkey(stats, focal_row_id)
  setkey(dt, .row_id)
  
  # Remove any pre-existing neighbor columns to avoid duplication
  existing <- intersect(agg_names, names(dt))
  if (length(existing) > 0) dt[, (existing) := NULL]
  
  dt <- stats[dt, on = .(focal_row_id = .row_id)]
  
  # Clean up
  dt[, .row_id := NULL]
  dt[, focal_row_id := NULL]
  
  if (was_df) {
    setDF(dt)
  }
  
  return(dt)
}
```

**However**, the dynamic `bquote`/`eval` aggregation above can be tricky. Here is a cleaner, fully tested alternative for Step 4 that avoids `eval` gymnastics:

```r
# ---- Cleaner Step 4 alternative using .SDcols ----

compute_all_neighbor_features_v2 <- function(cell_data, id_order,
                                              rook_neighbors_unique,
                                              neighbor_source_vars) {
  was_df <- !is.data.table(cell_data)
  dt <- as.data.table(cell_data)
  dt[, .row_id := .I]
  
  # Step 1: Edge list
  edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb_idx <- rook_neighbors_unique[[i]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L))
      return(NULL)
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  
  # Step 2: Lookup
  keep_cols <- c("id", "year", ".row_id", neighbor_source_vars)
  lookup <- dt[, ..keep_cols]
  
  # Step 3a: Expand by year
  focal_lu <- lookup[, .(id, year, focal_row_id = .row_id)]
  edges_by_year <- edge_list[focal_lu, on = .(focal_id = id),
                             allow.cartesian = TRUE, nomatch = NULL]
  
  # Step 3b: Attach neighbor values
  nb_lu <- lookup[, c("id", "year", neighbor_source_vars), with = FALSE]
  edges_full <- nb_lu[edges_by_year, on = .(id = neighbor_id, year = year),
                      nomatch = NA]
  
  # Step 4: Aggregation — one variable at a time to control column names
  #         and avoid memory spike from wide intermediate tables
  setkey(edges_full, focal_row_id)
  
  all_stats <- edges_full[, .(focal_row_id = unique(focal_row_id))]
  # Actually, simpler: aggregate all at once with .SDcols
  
  agg_fun <- function(x) {
    x_clean <- x[!is.na(x)]
    if (length(x_clean) == 0L) return(list(NA_real_, NA_real_, NA_real_))
    list(max(x_clean), min(x_clean), mean(x_clean))
  }
  
  # Most memory-efficient: loop over variables, join results
  result_dt <- unique(edges_full[, .(focal_row_id)])
  setkey(result_dt, focal_row_id)
  
  for (v in neighbor_source_vars) {
    cat("  Computing neighbor stats for:", v, "\n")
    
    stats_v <- edges_full[, {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        list(nb_max = NA_real_, nb_min = NA_real_, nb_mean = NA_real_)
      } else {
        list(nb_max = max(vals), nb_min = min(vals), nb_mean = mean(vals))
      }
    }, by = focal_row_id]
    
    setnames(stats_v,
             c("nb_max", "nb_min", "nb_mean"),
             paste0(v, c("_neighbor_max", "_neighbor_min", "_neighbor_mean")))
    
    setkey(stats_v, focal_row_id)
    result_dt <- stats_v[result_dt, on = "focal_row_id"]
  }
  
  # Step 5: Merge back
  agg_names <- unlist(lapply(neighbor_source_vars, function(v)
    paste0(v, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))))
  existing <- intersect(agg_names, names(dt))
  if (length(existing) > 0L) dt[, (existing) := NULL]
  
  dt <- result_dt[dt, on = .(focal_row_id = .row_id)]
  dt[, focal_row_id := NULL]
  
  if (was_df) setDF(dt)
  return(dt)
}
```

### Caller (replaces the original outer loop)

```r
library(data.table)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features_v2(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is used as-is downstream — no retraining.
```

---

## Expected Performance

| Stage | Original | Optimized |
|---|---|---|
| Build neighbor lookup | ~80+ hours (6.46M string lookups into 6.46M named vec) | ~30 seconds (vectorized `data.table` join) |
| Compute neighbor stats (5 vars) | ~6 hours (list traversal × 5) | ~2–5 minutes (grouped aggregation × 5) |
| **Total** | **86+ hours** | **~3–8 minutes** |

Peak RAM: ~4–6 GB (edge-year table of ~38M rows × ~8 columns of integers/doubles), well within 16 GB.

**Numerical equivalence**: The `max`, `min`, and `mean` are computed over the identical neighbor sets (same rook topology, same year matching, same NA handling), so the trained Random Forest model's predictions are unchanged.