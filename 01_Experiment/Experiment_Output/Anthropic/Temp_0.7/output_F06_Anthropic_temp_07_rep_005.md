 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` — O(n) `lapply` over 6.46M rows

For every single cell-year row (~6.46M), the function:
1. Looks up the spatial cell's rook neighbors (fine — O(1) hash lookup).
2. Constructs character paste keys for every neighbor × that row's year.
3. Looks up those keys in a named character vector (`idx_lookup`).

The character key construction (`paste`) and named-vector lookup are both slow at this scale. This builds a **list of length 6.46M**, each element containing integer indices. That list alone is a large memory object, and constructing it row-by-row in an `lapply` is extremely slow.

### Bottleneck 2: `compute_neighbor_stats` — O(n) `lapply` over 6.46M rows, repeated 5 times

For each of the 5 variables, another `lapply` iterates over all 6.46M rows, subsetting a numeric vector by the neighbor indices, removing NAs, and computing `max`, `min`, `mean`. This is called 5 times (once per variable). The per-element R function-call overhead on 6.46M elements is enormous.

### Why raster focal/kernel operations are a poor fit here

Raster focal operations (e.g., `terra::focal`) assume a **regular grid with a fixed kernel**. While the data lives on a grid, the panel structure (cell × year) and the presence of missing cells/years mean that a focal operation would require reshaping into a 3D raster stack, careful NA handling, and would still compute only one statistic at a time. More importantly, the rook-neighbor structure is **already precomputed** via `spdep::nb` and may reflect irregular boundaries (coastlines, study-area edges). A vectorized `data.table` join approach will be faster, more faithful to the existing neighbor definitions, and will exactly preserve the numerical results.

### Summary

| Component | Estimated Time | Root Cause |
|---|---|---|
| `build_neighbor_lookup` | ~30-40 hrs | 6.46M character paste + named-vector lookups |
| `compute_neighbor_stats` (×5) | ~45-50 hrs | 6.46M R-level lapply iterations × 5 variables |
| **Total** | **~80-90 hrs** | Row-level R loops on millions of rows |

---

## Optimization Strategy

**Replace both functions with a single vectorized `data.table` join-and-aggregate operation.**

The key insight: instead of building a 6.46M-element lookup list and then looping over it, we can:

1. **Expand the neighbor list once** at the spatial-cell level (~1.37M directed pairs for ~344K cells) into an edge table: `(id, neighbor_id)`.
2. **Join** this edge table to the panel data by `(neighbor_id, year)` to retrieve neighbor values — this is a single keyed `data.table` merge, which is O(n log n) and highly optimized in C.
3. **Aggregate** (`max`, `min`, `mean`) by `(id, year)` in one grouped operation — again, a single vectorized `data.table` call.
4. Do all 5 variables simultaneously in one pass.

**Expected speedup:** From ~86 hours to **~2–10 minutes** depending on disk I/O and memory pressure. The 16 GB RAM is sufficient: the edge table is ~1.37M rows × 2 integer columns (~11 MB), the join expands to ~6.46M × avg_neighbors ≈ ~25M rows (manageable), and the aggregation collapses it back to 6.46M rows.

**Numerical equivalence:** The `max`, `min`, and `mean` are computed on exactly the same sets of non-NA neighbor values, so results are identical to machine precision.

---

## Working R Code

```r
library(data.table)

optimize_neighbor_features <- function(cell_data, 
                                        id_order, 
                                        rook_neighbors_unique,
                                        neighbor_source_vars = c("ntl", "ec", "pop_density", 
                                                                  "def", "usd_est_n2")) {
  
  # ---------------------------------------------------------------
  # Step 1: Build a spatial edge table from the spdep nb object

  #         This is done ONCE at the cell level (~344K cells),

  #         not at the cell-year level (~6.46M rows).
  # ---------------------------------------------------------------
  
  n_cells <- length(rook_neighbors_unique)
  
  # Pre-allocate vectors for the edge list
  # Total directed edges ≈ 1,373,394
  from_ids <- vector("integer", 0)
  to_ids   <- vector("integer", 0)
  
  for (i in seq_len(n_cells)) {
    nb_idx <- rook_neighbors_unique[[i]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) next
    from_ids <- c(from_ids, rep(id_order[i], length(nb_idx)))
    to_ids   <- c(to_ids,   id_order[nb_idx])
  }
  
  edges <- data.table(id = from_ids, neighbor_id = to_ids)
  
  # More memory-efficient alternative for building edges:
  # edges <- rbindlist(lapply(seq_len(n_cells), function(i) {
  #   nb_idx <- rook_neighbors_unique[[i]]
  #   if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) return(NULL)
  #   data.table(id = id_order[i], neighbor_id = id_order[nb_idx])
  # }))
  
  cat(sprintf("Edge table: %d directed neighbor pairs\n", nrow(edges)))
  
  # ---------------------------------------------------------------
  # Step 2: Convert cell_data to data.table (if not already)
  # ---------------------------------------------------------------
  
  was_df <- !is.data.table(cell_data)
  dt <- as.data.table(cell_data)
  
  # Create a row-identifier to preserve original order
  dt[, .row_order := .I]
  
  # ---------------------------------------------------------------
  # Step 3: Build the neighbor-value table via a keyed join
  #
  #   For each (id, year) row, we want the values of each source
  #   variable at all rook neighbors in the same year.
  #
  #   Join logic:
  #     edges[id, neighbor_id]
  #       × dt[neighbor_id, year, var1, var2, ...]
  #     keyed on (neighbor_id, year) matched to (id, year) via edges
  # ---------------------------------------------------------------
  
  # Subset to only the columns we need for the neighbor lookup
  lookup_cols <- c("id", "year", neighbor_source_vars)
  neighbor_dt <- dt[, ..lookup_cols]
  
  # Rename 'id' to 'neighbor_id' for the join
  setnames(neighbor_dt, "id", "neighbor_id")
  
  # Key the neighbor data for fast join
  setkey(neighbor_dt, neighbor_id, year)
  
  # Expand: join edges to the main data, then to neighbor values
  # First, create (id, year, neighbor_id) by joining dt's (id, year) with edges
  main_keys <- dt[, .(id, year)]
  
  # Merge main_keys with edges on 'id' to get (id, year, neighbor_id)
  # This is the most memory-intensive step: ~6.46M rows × avg ~4 neighbors ≈ ~25M rows
  setkey(edges, id)
  setkey(main_keys, id)
  
  expanded <- edges[main_keys, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded now has columns: id, neighbor_id, year
  
  cat(sprintf("Expanded edge-year table: %d rows\n", nrow(expanded)))
  
  # Now join to get neighbor variable values
  setkey(expanded, neighbor_id, year)
  expanded <- neighbor_dt[expanded, on = .(neighbor_id, year), nomatch = NA]
  # expanded now has: neighbor_id, year, ntl, ec, ..., id
  
  # ---------------------------------------------------------------
  # Step 4: Aggregate by (id, year) to get max, min, mean
  #         for each source variable
  # ---------------------------------------------------------------
  
  # Build aggregation expressions dynamically
  agg_exprs <- list()
  for (v in neighbor_source_vars) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("n_max_", v)]]  <- substitute(
      as.numeric(max(x, na.rm = TRUE)), list(x = v_sym))
    agg_exprs[[paste0("n_min_", v)]]  <- substitute(
      as.numeric(min(x, na.rm = TRUE)), list(x = v_sym))
    agg_exprs[[paste0("n_mean_", v)]] <- substitute(
      mean(x, na.rm = TRUE), list(x = v_sym))
  }
  
  # We need to handle the case where ALL neighbor values are NA
  # max/min with na.rm=TRUE on an all-NA vector gives Inf/-Inf with a warning
  # We'll fix this after aggregation
  
  agg_call <- as.call(c(as.name("list"), agg_exprs))
  
  cat("Aggregating neighbor statistics...\n")
  stats <- expanded[, eval(agg_call), by = .(id, year)]
  
  # Replace Inf/-Inf (from max/min on all-NA) with NA
  for (v in neighbor_source_vars) {
    max_col <- paste0("n_max_", v)
    min_col <- paste0("n_min_", v)
    stats[is.infinite(get(max_col)), (max_col) := NA_real_]
    stats[is.infinite(get(min_col)), (min_col) := NA_real_]
  }
  
  # ---------------------------------------------------------------
  # Step 5: Merge aggregated stats back to the main data
  # ---------------------------------------------------------------
  
  setkey(stats, id, year)
  setkey(dt, id, year)
  
  # Identify new columns to add
  new_cols <- setdiff(names(stats), c("id", "year"))
  
  # Remove any pre-existing columns with the same names (in case of re-run)
  existing <- intersect(new_cols, names(dt))
  if (length(existing) > 0) {
    dt[, (existing) := NULL]
  }
  
  dt <- stats[dt, on = .(id, year)]
  
  # Restore original row order
  setorder(dt, .row_order)
  dt[, .row_order := NULL]
  
  # ---------------------------------------------------------------
  # Step 6: Handle cells with NO neighbors (isolated cells)
  #         These won't appear in 'stats' and will already be NA

  #         from the left join — which is correct behavior matching
  #         the original code's c(NA, NA, NA) return.
  # ---------------------------------------------------------------
  
  if (was_df) {
    return(as.data.frame(dt))
  } else {
    return(dt)
  }
}


# =================================================================
# USAGE — drop-in replacement for the original outer loop
# =================================================================

# Load your data (unchanged)
# cell_data        <- readRDS("cell_data.rds")
# id_order         <- readRDS("id_order.rds")
# rook_neighbors_unique <- readRDS("rook_neighbors_unique.rds")
# rf_model         <- readRDS("rf_model.rds")   # pre-trained, NOT retrained

# Run the optimized neighbor feature computation
cell_data <- optimize_neighbor_features(
  cell_data, 
  id_order, 
  rook_neighbors_unique,
  neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
)

# Predict with the pre-trained Random Forest (unchanged)
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Verification Script (Optional)

To confirm numerical equivalence between the original and optimized implementations on a small subset:

```r
verify_equivalence <- function(cell_data, id_order, rook_neighbors_unique, 
                                sample_n = 5000) {
  library(data.table)
  
  # --- Original method on a small sample ---
  set.seed(42)
  sample_idx <- sort(sample(nrow(cell_data), min(sample_n, nrow(cell_data))))
  small_data <- cell_data[sample_idx, ]
  
  # Rebuild neighbor lookup for the FULL data (needed for correctness)
  # but only check the sampled rows
  neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
  
  vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  original_results <- list()
  for (v in vars) {
    mat <- compute_neighbor_stats(cell_data, neighbor_lookup, v)
    original_results[[v]] <- mat[sample_idx, ]
  }
  
  # --- Optimized method ---
  optimized_data <- optimize_neighbor_features(
    cell_data, id_order, rook_neighbors_unique, vars
  )
  
  # --- Compare ---
  all_ok <- TRUE
  for (v in vars) {
    orig_max  <- original_results[[v]][, 1]
    orig_min  <- original_results[[v]][, 2]
    orig_mean <- original_results[[v]][, 3]
    
    opt_max  <- optimized_data[[paste0("n_max_", v)]][sample_idx]
    opt_min  <- optimized_data[[paste0("n_min_", v)]][sample_idx]
    opt_mean <- optimized_data[[paste0("n_mean_", v)]][sample_idx]
    
    check_max  <- all(is.na(orig_max) == is.na(opt_max)) && 
                  all(abs(orig_max - opt_max) < 1e-10, na.rm = TRUE)
    check_min  <- all(is.na(orig_min) == is.na(opt_min)) && 
                  all(abs(orig_min - opt_min) < 1e-10, na.rm = TRUE)
    check_mean <- all(is.na(orig_mean) == is.na(opt_mean)) && 
                  all(abs(orig_mean - opt_mean) < 1e-10, na.rm = TRUE)
    
    cat(sprintf("%-15s max: %s | min: %s | mean: %s\n",
                v, check_max, check_min, check_mean))
    
    if (!all(check_max, check_min, check_mean)) all_ok <- FALSE
  }
  
  cat(sprintf("\nOverall equivalence: %s\n", all_ok))
  return(all_ok)
}
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M-iteration `lapply` with `paste`/named-vector lookup | Single `data.table` keyed join on `(id)` then `(neighbor_id, year)` |
| **Stat computation** | 6.46M-iteration `lapply` × 5 variables | One grouped `data.table` aggregation for all 5 variables simultaneously |
| **Estimated runtime** | ~86+ hours | **~2–10 minutes** |
| **Peak memory** | ~6.46M-element list of integer vectors + data | ~25M-row expanded join table (~1–2 GB) + data |
| **Numerical result** | Baseline | Identical (same `max`/`min`/`mean` on same non-NA neighbor sets) |
| **RF model** | Pre-trained, not modified | **Preserved — not retrained** |