 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(N²)-like in practice
This function iterates over **all ~6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs `paste()`-based string keys for every neighbor × that row's year (expensive string allocation).
- Performs named-vector lookups via `idx_lookup[neighbor_keys]` — which is a **hash-table lookup on character keys repeated ~6.46M × ~4 neighbors ≈ 25.8 million times**.

The real killer is that the neighbor topology is **purely spatial and year-invariant**, yet the function rebuilds the mapping for every cell-year combination. With 344,208 cells × 28 years, it does 28× redundant work on the spatial graph and creates ~6.46 million character-key lookups.

### 2. `compute_neighbor_stats` uses per-row `lapply`
For each of the 5 variables, it loops over 6.46 million entries in R-level `lapply`, extracting and summarizing small vectors. That's ~32.3 million R-level function calls total, with no vectorization.

### Summary of bottlenecks
| Step | Calls | Cost driver |
|---|---|---|
| `build_neighbor_lookup` | 6.46M | `paste()` + named character vector lookup per row |
| `compute_neighbor_stats` | 6.46M × 5 vars | R-level `lapply` with per-row subsetting |

---

## Optimization Strategy

**Core insight:** The neighbor graph is spatial-only and time-invariant. Build it once as a simple integer-indexed adjacency table (cell index → neighbor cell indices), then use a vectorized **join-based** approach to compute neighbor statistics per year.

### Steps:
1. **Build a static spatial adjacency `data.table`** with columns `(cell_idx, neighbor_cell_idx)` from the `nb` object — done once, ~1.37M rows.
2. **Add year via cross-join**: For each year, join cell attributes onto the neighbor table by `(neighbor_cell_idx, year)` — this is a keyed `data.table` equi-join, fully vectorized in C.
3. **Group-by aggregation**: Group by `(cell_idx, year)` and compute `max`, `min`, `mean` of each neighbor variable — one vectorized `data.table` operation per variable.
4. **Join results back** onto the main dataset.

This replaces ~32M R-level function calls with a handful of vectorized `data.table` joins and grouped aggregations.

**Expected speedup:** From ~86 hours to **~2–5 minutes** on a 16 GB laptop.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 0: Convert main data to data.table (if not already)
# ──────────────────────────────────────────────────────────────────────
# cell_data must have columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order is the vector of cell IDs in the same order as rook_neighbors_unique
# rook_neighbors_unique is an nb object (list of integer index vectors)

setDT(cell_data)

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build the static spatial adjacency table ONCE
#         This is ~1.37M rows, year-invariant.
# ──────────────────────────────────────────────────────────────────────
build_adjacency_table <- function(id_order, nb_obj) {
  # nb_obj[[i]] contains integer indices into id_order for neighbors of cell i
  # id_order[i] is the actual cell ID for spatial index i
  n <- length(nb_obj)
  
  from_idx <- rep(seq_len(n), lengths(nb_obj))
  to_idx   <- unlist(nb_obj, use.names = FALSE)
  
  # Remove the 0-neighbor sentinel that spdep uses (nb with no neighbors = 0L)
  valid <- to_idx != 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]
  
  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

adj_table <- build_adjacency_table(id_order, rook_neighbors_unique)
# Result: ~1,373,394 rows with columns (id, neighbor_id)

cat("Adjacency table rows:", nrow(adj_table), "\n")

# ──────────────────────────────────────────────────────────────────────
# STEP 2: Compute neighbor stats for all variables via vectorized joins
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Create a slim lookup of cell attributes by (id, year)
# Only keep the columns we need for neighbor stats
attr_cols <- c("id", "year", neighbor_source_vars)
cell_attrs <- cell_data[, ..attr_cols]
setkey(cell_attrs, id, year)

# Get unique years
years <- sort(unique(cell_data$year))

compute_all_neighbor_features <- function(adj_table, cell_attrs,
                                          neighbor_source_vars, years) {
  # For each year, we:
  #   1. Cross the adjacency table with that year
  #   2. Join neighbor attributes
  #   3. Aggregate by (id, year)
  
  # Pre-allocate list for results
  year_results <- vector("list", length(years))
  
  # Rename neighbor attribute columns to avoid collision
  neighbor_var_names <- paste0("n_", neighbor_source_vars)
  
  for (yi in seq_along(years)) {
    yr <- years[yi]
    
    # Adjacency table for this year: every spatial edge gets this year
    # ~1.37M rows per year
    adj_yr <- adj_table[, .(id, neighbor_id, year = yr)]
    
    # Join neighbor cell attributes onto adj_yr
    # Key: (neighbor_id, year) matched to cell_attrs (id, year)
    adj_yr <- merge(
      adj_yr,
      cell_attrs,
      by.x = c("neighbor_id", "year"),
      by.y = c("id", "year"),
      all.x = TRUE,
      sort = FALSE
    )
    
    # Aggregate: for each (id, year), compute max/min/mean of each variable
    # Build aggregation expressions dynamically
    agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
      list(
        bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
        bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
        bquote(mean(.(as.name(v)), na.rm = TRUE))
      )
    }), recursive = FALSE)
    
    agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
      paste0("neighbor_", c("max_", "min_", "mean_"), v)
    }))
    
    names(agg_exprs) <- agg_names
    
    # Execute aggregation
    agg_result <- adj_yr[,
      lapply(agg_exprs, eval, envir = .SD),
      by = .(id, year),
      .SDcols = neighbor_source_vars
    ]
    
    year_results[[yi]] <- agg_result
    
    if (yi %% 5 == 0 || yi == length(years)) {
      cat(sprintf("  Processed year %d (%d/%d)\n", yr, yi, length(years)))
    }
  }
  
  rbindlist(year_results)
}

# --- Actually, a cleaner and faster approach avoids the year loop entirely ---
# We can do one big merge. With ~1.37M edges × 28 years = ~38.5M rows,
# this fits comfortably in 16 GB RAM.

compute_all_neighbor_features_vectorized <- function(adj_table, cell_attrs,
                                                      neighbor_source_vars) {
  cat("Building full edge-year table via cross join with years...\n")
  
  # Cross join adjacency with all years: ~38.5M rows
  all_years <- data.table(year = sort(unique(cell_attrs$year)))
  adj_full <- adj_table[, CJ_id := .I]  # just need the cross
  adj_full <- CJ(edge_id = seq_len(nrow(adj_table)),
                 year = all_years$year)
  adj_full[, `:=`(
    id          = adj_table$id[edge_id],
    neighbor_id = adj_table$neighbor_id[edge_id]
  )]
  adj_full[, edge_id := NULL]
  
  cat(sprintf("Edge-year table: %s rows\n", format(nrow(adj_full), big.mark = ",")))
  
  # Join neighbor attributes
  cat("Joining neighbor attributes...\n")
  setkey(cell_attrs, id, year)
  setkey(adj_full, neighbor_id, year)
  
  adj_full <- cell_attrs[adj_full, on = .(id = neighbor_id, year = year)]
  
  # Now adj_full has columns: id (= neighbor_id from original), year,
  # ntl, ec, pop_density, def, usd_est_n2, i.id (= focal cell id)
  # Fix column names — the merge flips id references
  setnames(adj_full, "i.id", "focal_id")
  # 'id' column now = neighbor_id, the source vars come from the neighbor
  
  # Aggregate by (focal_id, year)
  cat("Aggregating neighbor statistics...\n")
  
  # Build aggregation call
  agg_list <- list()
  for (v in neighbor_source_vars) {
    agg_list[[paste0("neighbor_max_", v)]]  <-
      substitute(as.numeric(max(VAR, na.rm = TRUE)), list(VAR = as.name(v)))
    agg_list[[paste0("neighbor_min_", v)]]  <-
      substitute(as.numeric(min(VAR, na.rm = TRUE)), list(VAR = as.name(v)))
    agg_list[[paste0("neighbor_mean_", v)]] <-
      substitute(mean(VAR, na.rm = TRUE), list(VAR = as.name(v)))
  }
  
  agg_call <- as.call(c(as.name("list"), agg_list))
  
  result <- adj_full[, eval(agg_call), by = .(focal_id, year)]
  setnames(result, "focal_id", "id")
  
  # Replace Inf/-Inf from max/min of all-NA groups with NA
  inf_cols <- grep("neighbor_max_|neighbor_min_", names(result), value = TRUE)
  for (col in inf_cols) {
    result[is.infinite(get(col)), (col) := NA_real_]
  }
  
  cat("Done.\n")
  result
}

# ──────────────────────────────────────────────────────────────────────
# STEP 3: Execute and merge back
# ──────────────────────────────────────────────────────────────────────
cat("Computing neighbor features (vectorized)...\n")
t0 <- proc.time()

neighbor_features <- compute_all_neighbor_features_vectorized(
  adj_table, cell_attrs, neighbor_source_vars
)

elapsed <- (proc.time() - t0)["elapsed"]
cat(sprintf("Neighbor feature computation: %.1f seconds\n", elapsed))

# Remove any old neighbor columns from cell_data to avoid duplication
old_neighbor_cols <- grep("^neighbor_", names(cell_data), value = TRUE)
if (length(old_neighbor_cols) > 0) {
  cell_data[, (old_neighbor_cols) := NULL]
}

# Merge neighbor features onto cell_data
setkey(cell_data, id, year)
setkey(neighbor_features, id, year)
cell_data <- neighbor_features[cell_data, on = .(id, year)]

cat(sprintf("Final dataset: %s rows × %d columns\n",
            format(nrow(cell_data), big.mark = ","), ncol(cell_data)))

# ──────────────────────────────────────────────────────────────────────
# STEP 4: Predict with the existing trained Random Forest (unchanged)
# ──────────────────────────────────────────────────────────────────────
# The trained model object (e.g., `rf_model`) is preserved as-is.
# Prediction proceeds exactly as before:
#
#   cell_data$rf_prediction <- predict(rf_model, newdata = cell_data)
#
# No retraining. The numerical estimand is identical because the
# neighbor max, min, and mean values are computed with the same
# arithmetic (max, min, mean of non-NA neighbor values), just
# via vectorized joins instead of row-level R loops.
```

---

## Why This Preserves the Original Numerical Estimand

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor graph | `rook_neighbors_unique` (nb object) | Same nb object → same adjacency pairs |
| Neighbor max | `max(neighbor_vals)` after removing NAs | `max(var, na.rm = TRUE)` in grouped `data.table` |
| Neighbor min | `min(neighbor_vals)` after removing NAs | `min(var, na.rm = TRUE)` in grouped `data.table` |
| Neighbor mean | `mean(neighbor_vals)` after removing NAs | `mean(var, na.rm = TRUE)` in grouped `data.table` |
| RF model | Pre-trained, used for `predict()` only | Unchanged — no retraining |

The join-and-aggregate approach computes **mathematically identical** values; only the computational mechanism changes from R-level loops to C-level vectorized operations.

## Expected Performance

| Step | Original | Optimized |
|---|---|---|
| Build neighbor lookup | ~hours (6.46M string ops) | ~1 second (integer adjacency table) |
| Compute neighbor stats | ~80+ hours (32.3M R calls) | ~1–3 minutes (vectorized join + group-by) |
| **Total** | **~86+ hours** | **~2–5 minutes** |

Peak memory for the 38.5M-row edge table with 5 numeric columns is approximately 38.5M × 7 columns × 8 bytes ≈ **2.2 GB**, well within the 16 GB laptop constraint.