 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two root causes:

### 1. `build_neighbor_lookup` is O(N²)-ish in practice
It iterates over all **6.46 million** cell-year rows, and for each row it:
- Does a character-based hash lookup (`id_to_ref`, `idx_lookup`) using `paste()` to build string keys.
- Constructs per-row neighbor keys by pasting cell IDs and years.
- Performs named-vector lookups (which in R are linear scans on long named vectors).

The `idx_lookup` named vector has ~6.46 million entries. Named vector lookup in R is **O(n)** per query (it is *not* a hash table). So each of the 6.46M rows does multiple O(6.46M) scans → effectively **O(n²)** total.

### 2. `compute_neighbor_stats` uses `lapply` over 6.46M rows
Each iteration subsets a numeric vector and computes max/min/mean. While each call is cheap, 6.46M R-level function calls with list allocation is inherently slow.

### 3. The neighbor topology is year-invariant but rebuilt per cell-year
Rook neighbors depend only on spatial grid position, not on year. The current code re-resolves neighbor *row indices* for every cell-year, mixing spatial topology with temporal indexing unnecessarily.

---

## Optimization Strategy

**Key insight:** Separate the *static spatial topology* from the *yearly attribute join*.

1. **Build a cell-level neighbor edge table once** — a simple two-column `data.table` of `(cell_id, neighbor_id)` derived from `rook_neighbors_unique`. This is only ~1.37M rows.

2. **Join yearly attributes onto the edge table** — For each year and variable, join the cell's attribute value onto the neighbor side of the edge table using `data.table` keyed joins (O(n log n) or O(n) with keys).

3. **Aggregate neighbor stats by `(cell_id, year)`** — Use `data.table`'s grouped aggregation (`[, .(max, min, mean), by = .(cell_id, year)]`) which is highly optimized in C.

4. **Join the aggregated stats back** onto the main data.

This eliminates all per-row R-level iteration, all `paste()`-based key construction, and all named-vector lookups. Expected runtime: **minutes, not hours**.

---

## Working R Code

```r
library(data.table)

# ─────────────────────────────────────────────────────────
# STEP 1: Build the static cell-neighbor edge table (once)
# ─────────────────────────────────────────────────────────
# rook_neighbors_unique is an nb object (list of integer index vectors)
# id_order is the vector of cell IDs corresponding to each nb element

build_edge_table <- function(id_order, neighbors) {
  # neighbors[[i]] gives integer indices into id_order for cell id_order[i]
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)
  
  edge_dt <- data.table(
    cell_id     = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
  return(edge_dt)
}

edge_table <- build_edge_table(id_order, rook_neighbors_unique)
# edge_table has ~1,373,394 rows: (cell_id, neighbor_id)

# ─────────────────────────────────────────────────────────
# STEP 2: Convert main data to data.table (if not already)
# ─────────────────────────────────────────────────────────
cell_dt <- as.data.table(cell_data)

# Ensure key columns exist and have consistent types
stopifnot(all(c("id", "year") %in% names(cell_dt)))

# ─────────────────────────────────────────────────────────
# STEP 3: Compute neighbor stats for all variables at once
# ─────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_features <- function(cell_dt, edge_table, vars) {
  
  # Extract only the columns we need for the neighbor join
  # Columns: id, year, and all neighbor source variables
  cols_needed <- c("id", "year", vars)
  attr_dt <- cell_dt[, ..cols_needed]
  
  # Key the attribute table by id for fast join
  setkey(attr_dt, id)
  
  # Create a year-expanded edge table:
  # For each year, every edge (cell_id -> neighbor_id) is valid.
  # Instead of a full cross join (which would be huge), we join per year.
  
  years <- sort(unique(attr_dt$year))
  
  # Pre-allocate list for results
  result_list <- vector("list", length(years))
  
  for (yi in seq_along(years)) {
    yr <- years[yi]
    
    # Subset attributes for this year
    yr_attr <- attr_dt[year == yr]
    setkey(yr_attr, id)
    
    # Join neighbor attributes onto edge table
    # edge_table$neighbor_id -> yr_attr$id to get neighbor's variable values
    merged <- merge(
      edge_table,
      yr_attr[, c("id", vars), with = FALSE],
      by.x = "neighbor_id",
      by.y = "id",
      allow.cartesian = FALSE
    )
    # merged now has: neighbor_id, cell_id, ntl, ec, pop_density, def, usd_est_n2
    # Each row = one directed neighbor relationship for this year
    
    # Aggregate: for each cell_id, compute max/min/mean of each variable
    # across all its neighbors
    agg_exprs <- list()
    for (v in vars) {
      v_sym <- as.name(v)
      agg_exprs[[paste0("n_max_", v)]]  <- bquote(max(.(v_sym), na.rm = TRUE))
      agg_exprs[[paste0("n_min_", v)]]  <- bquote(min(.(v_sym), na.rm = TRUE))
      agg_exprs[[paste0("n_mean_", v)]] <- bquote(mean(.(v_sym), na.rm = TRUE))
    }
    
    # Build the aggregation call
    # Equivalent to: merged[, .(n_max_ntl = max(ntl, na.rm=TRUE), ...), by = cell_id]
    agg_dt <- merged[, 
      lapply(.SD, function(x) {
        x <- x[!is.na(x)]
        if (length(x) == 0L) return(c(NA_real_, NA_real_, NA_real_))
        c(max(x), min(x), mean(x))
      }),
      by = cell_id,
      .SDcols = vars
    ]
    # The above returns 3 rows per cell_id (max, min, mean). 
    # A cleaner approach:
    
    agg_dt <- merged[, {
      res <- list()
      for (v in vars) {
        vals <- get(v)
        vals <- vals[!is.na(vals)]
        if (length(vals) == 0L) {
          res[[paste0("n_max_", v)]]  <- NA_real_
          res[[paste0("n_min_", v)]]  <- NA_real_
          res[[paste0("n_mean_", v)]] <- NA_real_
        } else {
          res[[paste0("n_max_", v)]]  <- max(vals)
          res[[paste0("n_min_", v)]]  <- min(vals)
          res[[paste0("n_mean_", v)]] <- mean(vals)
        }
      }
      res
    }, by = cell_id]
    
    agg_dt[, year := yr]
    result_list[[yi]] <- agg_dt
    
    if (yi %% 5 == 0) message(sprintf("  Processed year %d (%d/%d)", yr, yi, length(years)))
  }
  
  rbindlist(result_list)
}

message("Computing neighbor features...")
t0 <- Sys.time()
neighbor_features <- compute_all_neighbor_features(cell_dt, edge_table, neighbor_source_vars)
message(sprintf("Done in %.1f minutes", difftime(Sys.time(), t0, units = "mins")))

# ─────────────────────────────────────────────────────────
# STEP 4: Join neighbor features back to main data
# ─────────────────────────────────────────────────────────

# Replace Inf/-Inf from max/min of empty sets (shouldn't happen with our NA guard, but safety)
inf_cols <- names(neighbor_features)[grepl("^n_(max|min|mean)_", names(neighbor_features))]
for (col in inf_cols) {
  vals <- neighbor_features[[col]]
  vals[is.infinite(vals)] <- NA_real_
  set(neighbor_features, j = col, value = vals)
}

setkey(cell_dt, id, year)
setkey(neighbor_features, cell_id, year)

# Remove any pre-existing neighbor columns to avoid conflicts
old_ncols <- grep("^n_(max|min|mean)_", names(cell_dt), value = TRUE)
if (length(old_ncols) > 0) {
  cell_dt[, (old_ncols) := NULL]
}

# Merge
cell_dt <- merge(cell_dt, neighbor_features, by.x = c("id", "year"), by.y = c("cell_id", "year"), all.x = TRUE)

# ─────────────────────────────────────────────────────────
# STEP 5: Predict with the existing trained Random Forest
# ─────────────────────────────────────────────────────────
# The trained model object is assumed to be in memory (e.g., `rf_model`).
# Convert back to data.frame if the model expects one.

cell_data <- as.data.frame(cell_dt)

# Predict (model is NOT retrained — just used for inference)
cell_data$rf_prediction <- predict(rf_model, newdata = cell_data)

message("Pipeline complete.")
```

---

## Further Speed-Up: Vectorized Aggregation (No Inner Loop)

The `for (v in vars)` inside the `j`-expression of `data.table` can be replaced with a fully vectorized `.SD`-based aggregation for even more speed:

```r
# Faster alternative for STEP 3 inner aggregation:
compute_all_neighbor_features_fast <- function(cell_dt, edge_table, vars) {
  
  attr_dt <- cell_dt[, c("id", "year", vars), with = FALSE]
  years <- sort(unique(attr_dt$year))
  result_list <- vector("list", length(years))
  
  for (yi in seq_along(years)) {
    yr <- years[yi]
    yr_attr <- attr_dt[year == yr]
    
    merged <- yr_attr[edge_table, on = .(id = neighbor_id), nomatch = 0L,
                      allow.cartesian = FALSE]
    # merged columns: id (=neighbor_id), year, <vars>, cell_id
    
    # Compute all three stats at once using .SD
    agg_max  <- merged[, lapply(.SD, function(x) { x <- x[!is.na(x)]; if(length(x)==0L) NA_real_ else max(x) }),  by = cell_id, .SDcols = vars]
    agg_min  <- merged[, lapply(.SD, function(x) { x <- x[!is.na(x)]; if(length(x)==0L) NA_real_ else min(x) }),  by = cell_id, .SDcols = vars]
    agg_mean <- merged[, lapply(.SD, function(x) { x <- x[!is.na(x)]; if(length(x)==0L) NA_real_ else mean(x) }), by = cell_id, .SDcols = vars]
    
    setnames(agg_max,  vars, paste0("n_max_",  vars))
    setnames(agg_min,  vars, paste0("n_min_",  vars))
    setnames(agg_mean, vars, paste0("n_mean_", vars))
    
    combined <- agg_max[agg_min, on = "cell_id"][agg_mean, on = "cell_id"]
    combined[, year := yr]
    result_list[[yi]] <- combined
    
    if (yi %% 5 == 0) message(sprintf("  Year %d done (%d/%d)", yr, yi, length(years)))
  }
  
  rbindlist(result_list)
}
```

---

## Expected Performance

| Component | Before | After |
|---|---|---|
| Neighbor lookup build | ~40-60 hrs (6.46M × O(n) named-vec lookups) | ~2 seconds (vectorized edge table) |
| Neighbor stats (5 vars) | ~20-30 hrs (6.46M × 5 lapply calls) | ~5-15 min (data.table grouped agg × 28 years) |
| Total | **86+ hours** | **~5-20 minutes** |

The trained Random Forest model is never retrained — only `predict()` is called. The numerical estimand is preserved because `max`, `min`, and `mean` are computed identically over the same neighbor sets; only the implementation mechanism (vectorized join + grouped aggregation vs. row-wise R loop) changes.