 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of three compounding problems:

### 1. Row-Level `lapply` Over 6.46 Million Rows in `build_neighbor_lookup`
The function iterates row-by-row across all 6.46 million cell-year observations. For each row, it:
- Looks up the cell's reference index via a named character vector (`id_to_ref`).
- Retrieves neighbor cell IDs from the `nb` object.
- Pastes together `(neighbor_id, year)` keys as strings.
- Performs named-vector lookups on `idx_lookup` (a 6.46M-element named vector — each lookup is O(n) hash probing on a massive vector).

This produces a **list of 6.46 million integer vectors**, consuming enormous memory and time. The string-pasting and named-vector lookups are the dominant bottleneck.

### 2. Row-Level `lapply` Over 6.46 Million Rows in `compute_neighbor_stats`
For each of the 5 variables, the function iterates over the 6.46M-element neighbor lookup list, subsetting a numeric vector and computing `max`, `min`, `mean`. That is **5 × 6.46M = 32.3 million R-level function calls**, each with per-element overhead.

### 3. The Neighbor Topology Is Invariant Across Years, But Is Rebuilt Per Cell-Year
The rook-neighbor relationships are purely spatial — they don't change from year to year. Yet the current code embeds year into the lookup, effectively duplicating the same spatial adjacency structure 28 times and doing all the string work 28 times.

---

## Optimization Strategy

**Core insight:** Separate the *time-invariant spatial adjacency* from the *time-varying cell attributes*. Build the adjacency table once (344K cells × ~4 neighbors each ≈ 1.37M edges), then for each year, join attributes onto that edge table and compute grouped aggregates using `data.table`, which is vectorized in C.

### Steps:

1. **Build a `data.table` edge list once** from the `nb` object: columns `(cell_id, neighbor_id)` — ~1.37M rows. This is done once and is year-independent.

2. **For each year (or all years at once via a keyed join):** join the cell-year attributes onto the edge list by `(neighbor_id, year)`, then group by `(cell_id, year)` to compute `max`, `min`, `mean` of each neighbor variable.

3. **Join the resulting neighbor stats back** onto the main `cell_data` table.

This replaces 6.46M R-level list iterations with vectorized `data.table` keyed joins and grouped aggregations, reducing runtime from ~86 hours to **minutes**.

### Complexity comparison:

| Step | Current | Optimized |
|---|---|---|
| Build lookup | 6.46M string-paste + hash lookups | 1.37M-row edge table (once) |
| Compute stats (per var) | 6.46M R `lapply` calls | One vectorized `data.table` join + `groupby` over 1.37M × 28 ≈ 38.4M rows |
| Total R-level iterations | ~38.8M | ~0 (vectorized C) |

### Memory estimate:
- Edge table: 1.37M rows × 2 int cols ≈ 11 MB
- Expanded edge table (with year): 38.4M rows × 3 cols ≈ 920 MB
- Neighbor values joined: +1 double col ≈ +307 MB per variable
- Well within 16 GB RAM.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 1: Build the time-invariant spatial edge table ONCE
# ==============================================================================
# rook_neighbors_unique: an nb object (list of integer vectors of neighbor indices)
# id_order: vector of cell IDs corresponding to indices 1..344208 in the nb object

build_edge_table <- function(id_order, neighbors_nb) {
  # neighbors_nb is a list where neighbors_nb[[i]] gives integer indices of
  # neighbors of cell i (in id_order space). 0L means no neighbors in nb objects.
  
  n <- length(id_order)
  
  # Pre-calculate total edges for memory pre-allocation
  n_edges <- sum(vapply(neighbors_nb, function(x) {
    len <- length(x)
    # nb objects use integer(0) for no neighbors, or may contain 0L
    if (len == 1L && x[1] == 0L) 0L else len
  }, integer(1)))
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    nb_idx <- neighbors_nb[[i]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) next
    nb_len <- length(nb_idx)
    from_id[pos:(pos + nb_len - 1L)] <- id_order[i]
    to_id[pos:(pos + nb_len - 1L)]   <- id_order[nb_idx]
    pos <- pos + nb_len
  }
  
  data.table(cell_id = from_id, neighbor_id = to_id)
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

cat("Edge table rows:", nrow(edge_dt), "\n")
# Expected: ~1,373,394

# ==============================================================================
# STEP 2: Convert cell_data to data.table (if not already) and set keys
# ==============================================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure id and year columns are present
stopifnot(all(c("id", "year") %in% names(cell_data)))

# ==============================================================================
# STEP 3: Compute all neighbor features via vectorized joins
# ==============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_features <- function(cell_dt, edge_dt, source_vars) {
  
  # Create a lookup table: (neighbor_id aliased as cell_id, year) -> attribute values
  # We only need id, year, and the source variable columns for the join
  lookup_cols <- c("id", "year", source_vars)
  attr_dt <- cell_dt[, ..lookup_cols]
  
  # Rename 'id' to 'neighbor_id' for joining onto edge table
  setnames(attr_dt, "id", "neighbor_id")
  
  # Key the attribute table for fast join
  setkey(attr_dt, neighbor_id, year)
  
  # Get all unique years
  all_years <- sort(unique(cell_dt$year))
  
  # Expand edge table across all years:
  # CJ (cross join) of edge rows × years
  # More memory-efficient: use the edge table and cross join with years
  year_dt <- data.table(year = all_years)
  
  # Cross join edges with years
  # edge_dt has ~1.37M rows, 28 years => ~38.4M rows
  edge_year_dt <- edge_dt[, .(year = all_years), by = .(cell_id, neighbor_id)]
  
  cat("Edge-year table rows:", nrow(edge_year_dt), "\n")
  
  # Key for join
  setkey(edge_year_dt, neighbor_id, year)
  
  # Join neighbor attributes onto edge-year table
  edge_year_dt <- attr_dt[edge_year_dt, on = .(neighbor_id, year)]
  
  # Now edge_year_dt has columns: neighbor_id, year, ntl, ec, ..., cell_id
  # Group by (cell_id, year) and compute max, min, mean for each variable
  
  # Build aggregation expressions dynamically
  agg_exprs <- list()
  for (var in source_vars) {
    var_sym <- as.name(var)
    agg_exprs[[paste0("neighbor_max_", var)]]  <- bquote(
      as.numeric(max(.(var_sym), na.rm = TRUE))
    )
    agg_exprs[[paste0("neighbor_min_", var)]]  <- bquote(
      as.numeric(min(.(var_sym), na.rm = TRUE))
    )
    agg_exprs[[paste0("neighbor_mean_", var)]] <- bquote(
      mean(.(var_sym), na.rm = TRUE)
    )
  }
  
  # Convert to a single call for data.table's j argument
  agg_call <- as.call(c(as.name("list"), agg_exprs))
  
  cat("Computing grouped neighbor statistics...\n")
  
  neighbor_stats <- edge_year_dt[, eval(agg_call), by = .(cell_id, year)]
  
  # Handle Inf/-Inf from max/min on all-NA groups: replace with NA
  inf_cols <- names(neighbor_stats)[-(1:2)]
  for (col in inf_cols) {
    set(neighbor_stats, 
        i = which(is.infinite(neighbor_stats[[col]])), 
        j = col, 
        value = NA_real_)
  }
  
  # Rename cell_id back to id for merging
  setnames(neighbor_stats, "cell_id", "id")
  setkey(neighbor_stats, id, year)
  
  return(neighbor_stats)
}

neighbor_stats <- compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)

cat("Neighbor stats rows:", nrow(neighbor_stats), "\n")

# ==============================================================================
# STEP 4: Merge neighbor features onto cell_data
# ==============================================================================

# Remove any pre-existing neighbor columns to avoid duplication
new_cols <- setdiff(names(neighbor_stats), c("id", "year"))
existing_overlap <- intersect(names(cell_data), new_cols)
if (length(existing_overlap) > 0) {
  cell_data[, (existing_overlap) := NULL]
}

setkey(cell_data, id, year)
cell_data <- neighbor_stats[cell_data, on = .(id, year)]

cat("Final cell_data rows:", nrow(cell_data), "\n")
cat("Final cell_data cols:", ncol(cell_data), "\n")

# ==============================================================================
# STEP 5: Predict with the existing trained Random Forest (unchanged)
# ==============================================================================
# The trained RF model object (e.g., `rf_model`) is used as-is.
# The column names produced above (neighbor_max_ntl, neighbor_min_ntl, 
# neighbor_mean_ntl, etc.) must match the names expected by the model.
#
# If the original code used different naming conventions (e.g., "ntl_neighbor_max"),
# rename accordingly:

# Example renaming block (adjust to match your trained model's expected names):
# old_to_new <- c(
#   "neighbor_max_ntl"  = "ntl_neighbor_max",
#   "neighbor_min_ntl"  = "ntl_neighbor_min",
#   "neighbor_mean_ntl" = "ntl_neighbor_mean",
#   ...
# )
# setnames(cell_data, names(old_to_new), old_to_new)

# Then predict:
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Summary of Expected Gains

| Metric | Current | Optimized |
|---|---|---|
| **`build_neighbor_lookup`** | ~6.46M string operations + hash lookups | One loop over 344K cells to build 1.37M-row edge table (~seconds) |
| **`compute_neighbor_stats`** (per var) | 6.46M R-level `lapply` iterations | One `data.table` keyed join + grouped aggregation (~38.4M rows, vectorized C) |
| **Total for 5 variables** | 5 × 6.46M = 32.3M R iterations | 5 aggregates computed in a single grouped operation |
| **Estimated wall time** | ~86+ hours | **~2–10 minutes** on a 16 GB laptop |
| **Peak RAM** | Large (6.46M-element list of integer vectors) | ~3–5 GB (edge-year table + attributes) |
| **RF model** | Unchanged ✓ | Unchanged ✓ |
| **Numerical results** | Original estimand preserved ✓ | Identical `max`/`min`/`mean` values ✓ |

The key transformation is: **stop iterating over cell-year rows in R; instead, express the neighbor relationship as a static edge table and let `data.table`'s C-level join and grouping engine do the work vectorized.** The trained Random Forest model is never retrained — only the feature-engineering step is accelerated.