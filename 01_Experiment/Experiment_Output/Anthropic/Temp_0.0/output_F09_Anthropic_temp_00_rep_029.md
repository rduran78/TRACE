 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(N²)-like in practice
This function iterates over **all ~6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs **character key strings** (`paste(id, year)`) and performs **named-vector lookups** (`idx_lookup[neighbor_keys]`) — these are hash-table lookups but done 6.46 million times, each creating temporary character vectors and subsetting a 6.46-million-entry named vector.

The result is a **list of 6.46 million integer vectors**, one per row. This is both slow to build and memory-heavy.

### 2. `compute_neighbor_stats` iterates row-by-row again
For each of the 6.46 million rows, it subsets a numeric vector by the neighbor indices, removes NAs, and computes `max/min/mean`. This is repeated **5 times** (once per source variable), totaling ~32.3 million R-level loop iterations with per-element allocation.

### Root cause
The neighbor topology is **static across years** — each cell has the same rook neighbors every year. But the lookup is rebuilt at the cell-year level, exploding a ~344K-cell spatial problem into a ~6.46M-row problem. The key insight: **spatial adjacency is time-invariant; only the attribute values change by year.**

---

## Optimization Strategy

**Build the adjacency table once at the cell level (344K rows), then use a vectorized merge/join per year to compute neighbor statistics.**

Specifically:

1. **Convert the `nb` object to a two-column edge table** (`id`, `neighbor_id`) — ~1.37M rows. Do this once.
2. **For each year**, join the cell attributes onto the edge table (by `neighbor_id`), then `group_by(id)` and summarize `max`, `min`, `mean` for each variable. This is a standard grouped aggregation — extremely fast in `data.table`.
3. **Join the resulting neighbor stats back** onto the main cell-year data frame.

This replaces 6.46M R-level list operations with a handful of vectorized `data.table` joins and grouped aggregations per year. Expected speedup: **~200–500x**, bringing runtime to **minutes** instead of days.

**The trained Random Forest model is untouched.** The output columns are numerically identical (same `max`, `min`, `mean` of the same neighbor values), preserving the original estimand.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build a static spatial edge table from the nb object (once)
# ──────────────────────────────────────────────────────────────────────
# id_order is the vector of cell IDs corresponding to positions in the nb list.
# rook_neighbors_unique is the spdep::nb object (list of integer index vectors).

build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate: count total edges
  n_edges <- sum(vapply(neighbors, length, integer(1)))
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_idx <- neighbors[[i]]
    n      <- length(nb_idx)
    if (n > 0L) {
      from_id[pos:(pos + n - 1L)] <- id_order[i]
      to_id[pos:(pos + n - 1L)]   <- id_order[nb_idx]
      pos <- pos + n
    }
  }
  
  data.table(id = from_id, neighbor_id = to_id)
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# edge_dt has ~1,373,394 rows: (id, neighbor_id)

# ──────────────────────────────────────────────────────────────────────
# STEP 2: Convert main data to data.table (if not already)
# ──────────────────────────────────────────────────────────────────────
cell_dt <- as.data.table(cell_data)

# Ensure key columns exist
stopifnot(all(c("id", "year") %in% names(cell_dt)))

# Define the neighbor source variables and the output column names
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# ──────────────────────────────────────────────────────────────────────
# STEP 3: Compute all neighbor stats via vectorized join + group-by
# ──────────────────────────────────────────────────────────────────────
# Strategy: process one year at a time to limit peak memory.
# For each year:
#   - Extract that year's cell attributes (344K rows)
#   - Join onto edge_dt by neighbor_id to get neighbor attribute values
#   - Group by id, compute max/min/mean for each variable
#   - Store results

years <- sort(unique(cell_dt$year))

# Pre-create output columns (initialized to NA) for speed
for (var in neighbor_source_vars) {
  col_max  <- paste0("neighbor_max_", var)
  col_min  <- paste0("neighbor_min_", var)
  col_mean <- paste0("neighbor_mean_", var)
  cell_dt[, (col_max)  := NA_real_]
  cell_dt[, (col_min)  := NA_real_]
  cell_dt[, (col_mean) := NA_real_]
}

# Set key on cell_dt for fast subsetting and updating
setkey(cell_dt, id, year)

cat("Processing", length(years), "years x", length(neighbor_source_vars), "variables\n")

for (yr in years) {
  
  # Extract this year's attribute values: only id + source vars needed
  yr_attrs <- cell_dt[year == yr, c("id", neighbor_source_vars), with = FALSE]
  
  # Join neighbor attributes onto edge table:
  # edge_dt has (id, neighbor_id); we want the neighbor's attribute values
  # Join: edge_dt.neighbor_id == yr_attrs.id
  setkey(yr_attrs, id)
  edges_with_vals <- yr_attrs[edge_dt, on = .(id = neighbor_id), nomatch = NA,
                               allow.cartesian = FALSE]
  # Result columns: id (= neighbor_id), <vars>, neighbor_id (from edge_dt)
  # But data.table renames: the 'id' from yr_attrs matched to 'neighbor_id',
  # and edge_dt$id becomes i.id (or we need to be explicit).
  
  # Let's be more explicit to avoid column name confusion:
  edges_with_vals <- merge(
    edge_dt,
    yr_attrs,
    by.x = "neighbor_id",
    by.y = "id",
    all.x = TRUE,    # keep all edges even if neighbor has NA
    sort = FALSE
  )
  # edges_with_vals: (id, neighbor_id, ntl, ec, pop_density, def, usd_est_n2)
  # where the variable columns are the NEIGHBOR's values
  
  # Group by focal cell id, compute stats for each variable
  agg_exprs <- list()
  for (var in neighbor_source_vars) {
    col_max  <- paste0("neighbor_max_", var)
    col_min  <- paste0("neighbor_min_", var)
    col_mean <- paste0("neighbor_mean_", var)
    var_sym  <- as.name(var)
    agg_exprs[[col_max]]  <- bquote(
      if (all(is.na(.(var_sym)))) NA_real_ else max(.(var_sym), na.rm = TRUE)
    )
    agg_exprs[[col_min]]  <- bquote(
      if (all(is.na(.(var_sym)))) NA_real_ else min(.(var_sym), na.rm = TRUE)
    )
    agg_exprs[[col_mean]] <- bquote(
      if (all(is.na(.(var_sym)))) NA_real_ else mean(.(var_sym), na.rm = TRUE)
    )
  }
  
  # Build the j-expression for data.table
  # Simpler approach: compute in a straightforward grouped operation
  stats_yr <- edges_with_vals[, {
    out <- list()
    for (v in neighbor_source_vars) {
      vals <- .SD[[v]]
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        out[[paste0("neighbor_max_", v)]]  <- NA_real_
        out[[paste0("neighbor_min_", v)]]  <- NA_real_
        out[[paste0("neighbor_mean_", v)]] <- NA_real_
      } else {
        out[[paste0("neighbor_max_", v)]]  <- max(vals)
        out[[paste0("neighbor_min_", v)]]  <- min(vals)
        out[[paste0("neighbor_mean_", v)]] <- mean(vals)
      }
    }
    out
  }, by = id, .SDcols = neighbor_source_vars]
  
  # Now update cell_dt for this year with the computed stats
  stat_cols <- names(stats_yr)[names(stats_yr) != "id"]
  stats_yr[, year := yr]
  setkey(stats_yr, id, year)
  
  cell_dt[stats_yr, (stat_cols) := mget(paste0("i.", stat_cols)),
          on = .(id, year)]
  
  cat("  Year", yr, "done\n")
}

# ──────────────────────────────────────────────────────────────────────
# STEP 4: Convert back to data.frame if needed for the RF predict call
# ──────────────────────────────────────────────────────────────────────
cell_data <- as.data.frame(cell_dt)

# ──────────────────────────────────────────────────────────────────────
# STEP 5: Predict with the existing trained Random Forest (unchanged)
# ──────────────────────────────────────────────────────────────────────
# The trained model object (e.g., rf_model) is used as-is:
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Further Optimization: Eliminate the Per-Year R Loop Inside Groups

The grouped `.SD` loop above is still somewhat slow because `data.table` calls an R function per group. A faster approach uses **column-wise vectorized aggregation** by pre-computing per-variable stats in separate calls:

```r
# ──────────────────────────────────────────────────────────────────────
# FASTER STEP 3: Fully vectorized, no R-level per-group function
# ──────────────────────────────────────────────────────────────────────

edge_dt_keyed <- copy(edge_dt)
setkey(edge_dt_keyed, neighbor_id)

all_stats <- vector("list", length(years))

for (yr_i in seq_along(years)) {
  yr <- years[yr_i]
  
  # This year's attributes
  yr_attrs <- cell_dt[year == yr, c("id", neighbor_source_vars), with = FALSE]
  setkey(yr_attrs, id)
  
  # Join: get neighbor values for every edge
  joined <- yr_attrs[edge_dt_keyed, on = .(id = neighbor_id), nomatch = NA]
  # Columns: id (=neighbor_id that matched), <vars>, i.id (=focal cell from edge_dt)
  # Rename for clarity
  setnames(joined, "i.id", "focal_id")
  
  # Compute stats per focal_id, one variable at a time (fully vectorized)
  stats_list <- list(focal_id = joined[, sort(unique(focal_id))])
  
  # Actually, let data.table do it all at once with built-in functions:
  stat_expr_parts <- character(0)
  for (var in neighbor_source_vars) {
    vmax  <- paste0("neighbor_max_", var)
    vmin  <- paste0("neighbor_min_", var)
    vmean <- paste0("neighbor_mean_", var)
    stat_expr_parts <- c(stat_expr_parts,
      sprintf('"%s" = fifelse(all(is.na(%s)), NA_real_, max(%s, na.rm=TRUE))', vmax, var, var),
      sprintf('"%s" = fifelse(all(is.na(%s)), NA_real_, min(%s, na.rm=TRUE))', vmin, var, var),
      sprintf('"%s" = fifelse(all(is.na(%s)), NA_real_, mean(%s, na.rm=TRUE))', vmean, var, var)
    )
  }
  
  expr_str <- paste0("list(", paste(stat_expr_parts, collapse = ", "), ")")
  stats_yr <- joined[, eval(parse(text = expr_str)), by = focal_id]
  setnames(stats_yr, "focal_id", "id")
  stats_yr[, year := yr]
  
  all_stats[[yr_i]] <- stats_yr
  cat("  Year", yr, "done\n")
}

# Bind all years and update cell_dt in one shot
all_stats_dt <- rbindlist(all_stats)
setkey(all_stats_dt, id, year)
setkey(cell_dt, id, year)

stat_cols <- setdiff(names(all_stats_dt), c("id", "year"))
cell_dt[all_stats_dt, (stat_cols) := mget(paste0("i.", stat_cols)), on = .(id, year)]

cell_data <- as.data.frame(cell_dt)
```

---

## Expected Performance

| Component | Original | Optimized |
|---|---|---|
| Neighbor lookup build | ~hours (6.46M R iterations, string hashing) | ~1 second (vectorized edge table from nb) |
| Neighbor stats (5 vars) | ~hours (32.3M R iterations) | ~5–15 min (28 data.table grouped aggregations on ~1.37M-row edge table) |
| Peak RAM | High (6.46M-element list of integer vectors) | Moderate (~1.37M-row edge table + one year's join at a time) |
| **Total** | **~86+ hours** | **~5–20 minutes** |

The trained Random Forest model is never modified. The neighbor `max`, `min`, and `mean` values are computed from exactly the same neighbor sets and the same variable values, preserving the original numerical estimand identically.