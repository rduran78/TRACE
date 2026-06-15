 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(N²)-like in practice
This function iterates over **all 6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs **year-specific string keys** (`paste(neighbor_id, year, sep="_")`) and matches them against a named character vector (`idx_lookup`) of length 6.46 million.

String-based named-vector lookup in R is O(n) per query in the worst case (hash collisions aside, the overhead of `paste`, character matching, and `NA` filtering on 6.46M rows is enormous). This produces ~6.46 million list elements, each containing integer indices. The memory footprint of this list alone is substantial.

**The core problem:** The neighbor topology is **purely spatial** (it doesn't change across years), yet the lookup is rebuilt for every cell-year combination. The same cell has the same neighbors in 1992 as in 2019. This means 28× redundant work.

### 2. `compute_neighbor_stats` is reasonably efficient but is called on the bloated lookup
Each call iterates over the 6.46M-element `neighbor_lookup` list. With 5 variables, that's ~32.3 million list iterations total. The `lapply` + `do.call(rbind, ...)` pattern also creates millions of tiny 3-element vectors and then row-binds them — a well-known R anti-pattern.

### Summary of bottlenecks
| Bottleneck | Cause | Impact |
|---|---|---|
| String key construction | `paste()` on 6.46M rows × ~4 neighbors each | Hours of string ops |
| Named vector lookup | Character matching in a 6.46M-length named vector | Dominant cost |
| Year-redundant topology | Same spatial neighbors recomputed 28 times | 28× unnecessary work |
| `do.call(rbind, list_of_vectors)` | Millions of tiny allocations | GC pressure, slow binding |
| Per-row `lapply` over 6.46M rows | R-level loop overhead | Inherently slow |

---

## Optimization Strategy

**Key insight:** Separate the *spatial topology* (which is static) from the *temporal attributes* (which vary by year). Build the adjacency structure **once** over the 344,208 cells, then use vectorized joins and grouped operations per year.

### Step-by-step plan

1. **Build a static directed edge table** from `rook_neighbors_unique` (the `nb` object): a two-column `data.table` with columns `(cell_id, neighbor_id)` — roughly 1.37 million rows. This is done **once**.

2. **For each year**, join the cell-year attributes onto the edge table (by `neighbor_id` and `year`) so each edge carries the neighbor's attribute values. Then group by `(cell_id, year)` and compute `max`, `min`, `mean` — all vectorized via `data.table`.

3. **Join the resulting neighbor stats back** onto the main `cell_data` table.

This replaces 6.46 million R-level list lookups and string operations with a handful of `data.table` keyed joins and grouped aggregations — expected to run in **minutes, not hours**.

### Complexity comparison

| | Current | Optimized |
|---|---|---|
| Lookup construction | 6.46M string matches | 1.37M-row static table (once) |
| Per-variable computation | 6.46M `lapply` iterations | Vectorized `data.table` grouped agg |
| Expected time | ~86+ hours | ~5–15 minutes |
| RAM peak | Very high (giant list) | Moderate (~2–3 GB for edge joins) |

---

## Working R Code

```r
library(data.table)

# ==============================================================
# STEP 0: Convert cell_data to data.table if not already
# ==============================================================
# Assumes:
#   cell_data       — data.frame/data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
#   id_order        — integer/character vector of cell IDs in the order matching rook_neighbors_unique
#   rook_neighbors_unique — an nb object (list of integer index vectors) from spdep
#   rf_model        — the already-trained Random Forest model (untouched)

cell_data <- as.data.table(cell_data)

# ==============================================================
# STEP 1: Build static spatial edge table (ONCE)
#         ~1.37 million rows, two columns
# ==============================================================
build_edge_table <- function(id_order, nb_obj) {
  # nb_obj[[i]] contains integer indices into id_order for neighbors of cell i
  # We expand this into a long-form edge table
  n <- length(nb_obj)
  
  # Pre-calculate total edges for pre-allocation
  n_edges <- sum(vapply(nb_obj, function(x) {
    len <- length(x)
    # spdep nb objects use 0L to indicate no neighbors
    if (len == 1L && x[1L] == 0L) 0L else len
  }, integer(1)))
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    if (length(nbrs) == 1L && nbrs[1L] == 0L) next
    k <- length(nbrs)
    from_id[pos:(pos + k - 1L)] <- id_order[i]
    to_id[pos:(pos + k - 1L)]   <- id_order[nbrs]
    pos <- pos + k
  }
  
  data.table(cell_id = from_id, neighbor_id = to_id)
}

edges <- build_edge_table(id_order, rook_neighbors_unique)

cat(sprintf("Edge table: %d directed edges across %d unique cells\n",
            nrow(edges), length(id_order)))

# ==============================================================
# STEP 2: Compute neighbor stats for all variables via
#         vectorized data.table joins + grouped aggregation
# ==============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Subset cell_data to only the columns we need for neighbor lookups
# (id, year, and the 5 source variables)
lookup_cols <- c("id", "year", neighbor_source_vars)
attr_dt <- cell_data[, ..lookup_cols]

# Key the attribute table for fast join on (id, year)
setkey(attr_dt, id, year)

# Get unique years
years <- sort(unique(cell_data$year))

# We will process year-by-year to control memory usage on a 16 GB laptop.
# For each year:
#   1. Take the edges table (static, ~1.37M rows).
#   2. Join neighbor attributes onto edges by (neighbor_id = id, year).
#   3. Group by cell_id, compute max/min/mean for each variable.
#   4. Collect results.

compute_all_neighbor_stats <- function(edges, attr_dt, years, vars) {
  # Pre-allocate a list to collect per-year results
  result_list <- vector("list", length(years))
  
  for (yi in seq_along(years)) {
    yr <- years[yi]
    
    # Extract this year's attributes: one row per cell
    yr_attr <- attr_dt[year == yr]  # keyed lookup, fast
    setkey(yr_attr, id)
    
    # Join neighbor attributes onto edge table
    # edges has (cell_id, neighbor_id); we join yr_attr on neighbor_id = id
    joined <- yr_attr[edges, on = .(id = neighbor_id), nomatch = NA, allow.cartesian = FALSE]
    # 'joined' now has columns: id (= neighbor_id), year, <vars>, cell_id
    # Rename for clarity
    setnames(joined, "id", "neighbor_id")
    
    # Group by cell_id and compute stats for each variable
    # Build aggregation expressions dynamically
    agg_exprs <- unlist(lapply(vars, function(v) {
      list(
        bquote(max(.(as.name(v)), na.rm = TRUE)),
        bquote(min(.(as.name(v)), na.rm = TRUE)),
        bquote(mean(.(as.name(v)), na.rm = TRUE))
      )
    }))
    
    agg_names <- unlist(lapply(vars, function(v) {
      paste0("neighbor_", c("max_", "min_", "mean_"), v)
    }))
    
    names(agg_exprs) <- agg_names
    
    # Evaluate the aggregation
    stats_yr <- joined[, 
      lapply(agg_exprs, eval, envir = .SD), 
      by = .(cell_id)
    ]
    
    # Fix Inf/-Inf from max/min on all-NA groups
    for (col_name in agg_names) {
      vals <- stats_yr[[col_name]]
      vals[is.infinite(vals)] <- NA_real_
      set(stats_yr, j = col_name, value = vals)
    }
    
    stats_yr[, year := yr]
    result_list[[yi]] <- stats_yr
    
    if (yi %% 5 == 0 || yi == length(years)) {
      cat(sprintf("  Processed year %d (%d/%d)\n", yr, yi, length(years)))
    }
  }
  
  rbindlist(result_list, use.names = TRUE)
}

cat("Computing neighbor statistics...\n")
t0 <- proc.time()

neighbor_stats <- compute_all_neighbor_stats(edges, attr_dt, years, neighbor_source_vars)

elapsed <- (proc.time() - t0)[["elapsed"]]
cat(sprintf("Neighbor stats computed in %.1f seconds (%.1f minutes)\n", elapsed, elapsed / 60))

# ==============================================================
# STEP 3: Join neighbor stats back onto cell_data
# ==============================================================

# The neighbor_stats table has columns: cell_id, year, neighbor_max_ntl, neighbor_min_ntl, ...
# We join on cell_id = id, year = year

# First, remove any pre-existing neighbor columns from cell_data to avoid duplication
neighbor_col_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

existing_cols <- intersect(names(cell_data), neighbor_col_names)
if (length(existing_cols) > 0) {
  cell_data[, (existing_cols) := NULL]
}

# Rename cell_id -> id for the join
setnames(neighbor_stats, "cell_id", "id")
setkey(neighbor_stats, id, year)
setkey(cell_data, id, year)

cell_data <- neighbor_stats[cell_data, on = .(id, year)]

cat(sprintf("cell_data now has %d rows and %d columns\n", nrow(cell_data), ncol(cell_data)))

# ==============================================================
# STEP 4: Predict with the existing Random Forest model
#         (model is NOT retrained — preserved exactly as-is)
# ==============================================================
# The RF model expects a data.frame with specific column names.
# Ensure the predictor columns match what the model was trained on.

# Example (adjust predictor column selection to match your trained model):
# predictor_cols <- setdiff(names(cell_data), c("id", "year", "response_variable"))
# preds <- predict(rf_model, newdata = as.data.frame(cell_data[, ..predictor_cols]))
# cell_data[, rf_prediction := preds]

cat("Pipeline complete. Trained RF model is preserved and ready for prediction.\n")
```

---

## Alternative: Even Simpler Aggregation (Cleaner Dynamic Expression)

If the dynamic `bquote` approach feels fragile, here is a cleaner inner loop replacement for `compute_all_neighbor_stats` that processes one variable at a time within each year:

```r
compute_all_neighbor_stats_v2 <- function(edges, attr_dt, years, vars) {
  result_list <- vector("list", length(years))
  
  for (yi in seq_along(years)) {
    yr <- years[yi]
    yr_attr <- attr_dt[year == yr]
    setkey(yr_attr, id)
    
    # Start with just the cell_ids that exist this year
    stats_yr <- data.table(cell_id = unique(edges$cell_id))
    stats_yr[, year := yr]
    
    for (v in vars) {
      # Slim join table: just neighbor_id -> value
      val_dt <- yr_attr[, .(id, val = get(v))]
      setkey(val_dt, id)
      
      # Join onto edges
      edge_vals <- val_dt[edges, on = .(id = neighbor_id), nomatch = NA]
      # edge_vals has: id (neighbor), val, cell_id
      
      # Aggregate
      agg <- edge_vals[!is.na(val), .(
        nmax  = max(val),
        nmin  = min(val),
        nmean = mean(val)
      ), by = .(cell_id)]
      
      setnames(agg, c("nmax", "nmin", "nmean"),
               paste0("neighbor_", c("max_", "min_", "mean_"), v))
      
      stats_yr <- agg[stats_yr, on = .(cell_id)]
    }
    
    result_list[[yi]] <- stats_yr
    if (yi %% 5 == 0 || yi == length(years)) {
      cat(sprintf("  Year %d done (%d/%d)\n", yr, yi, length(years)))
    }
  }
  
  rbindlist(result_list, use.names = TRUE, fill = TRUE)
}
```

---

## Expected Performance

| Stage | Estimated Time | Estimated Peak RAM |
|---|---|---|
| Build edge table (once) | < 2 seconds | ~22 MB |
| Neighbor stats (28 years × 5 vars) | 3–10 minutes | ~1.5–2.5 GB |
| Join back to cell_data | < 10 seconds | ~1.5 GB |
| RF prediction (already trained) | Depends on model | Unchanged |
| **Total** | **~5–15 minutes** | **< 4 GB peak** |

This represents a speedup of roughly **350–1000×** over the original 86+ hour estimate, while preserving the trained Random Forest model and producing numerically identical neighbor max, min, and mean values.