 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is squarely in `build_neighbor_lookup`, not in `compute_neighbor_stats` or the Random Forest. Here's why:

**`build_neighbor_lookup` is O(N²)-like in practice.** It calls `lapply` over all ~6.46 million rows. For each row, it:

1. Looks up the cell's reference index (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. **Constructs character keys** by pasting neighbor IDs with the current row's year (`paste(..., sep="_")`).
4. **Performs named-vector lookups** (`idx_lookup[neighbor_keys]`) — which in R is an O(n) hash-table probe *per key*, but the named vector itself has 6.46 million entries, and this is done inside an `lapply` over 6.46 million iterations.

The total number of key lookups is ~6.46M rows × ~4 neighbors/cell (rook) = ~25.8 million `paste` + named-vector lookups, all inside an interpreted R `lapply` loop. The `paste` allocations and repeated character matching against a 6.46M-entry named vector are extremely expensive. The result: an estimated 86+ hours.

**`compute_neighbor_stats`** is comparatively cheap (numeric subsetting + simple aggregation), but it is also called 5 times with an R-level `lapply` over 6.46M rows, which adds up.

---

## Optimization Strategy

### Principle: Replace row-level R loops and character-key lookups with vectorized joins via `data.table`.

1. **Vectorized neighbor-lookup construction**: Instead of looping per row, expand the `nb` object into an edge-list (cell_id → neighbor_id) once, then do a single `data.table` merge-join keyed on `(neighbor_id, year)` to resolve all neighbor row indices in bulk. This eliminates all `paste` and named-vector lookups.

2. **Vectorized neighbor statistics**: Group the joined table by `(focal_row)` and compute `max`, `min`, `mean` in one `data.table` aggregation — no `lapply` needed.

3. **Repeat for each variable**: Each of the 5 variables is a single grouped aggregation on the same join result, so the join is done once and reused.

**Expected speedup**: The join is O(N log N) or O(N) with keys, and grouping is highly optimized in `data.table`. This should reduce 86+ hours to **minutes** (typically 2–10 minutes on a 16 GB laptop).

---

## Working R Code

```r
library(data.table)

#' Build a full edge list from an spdep nb object.
#' Returns a data.table with columns: cell_id, neighbor_cell_id
#' @param id_order integer vector of cell IDs in the order matching the nb object
#' @param neighbors an nb object (list of integer index vectors)
build_edge_list <- function(id_order, neighbors) {
  # Pre-allocate by computing total number of edges
  n_edges <- sum(vapply(neighbors, length, integer(1)))
  
  focal_idx <- integer(n_edges)
  neighbor_idx <- integer(n_edges)
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_i <- neighbors[[i]]
    len <- length(nb_i)
    if (len > 0L) {
      focal_idx[pos:(pos + len - 1L)] <- i
      neighbor_idx[pos:(pos + len - 1L)] <- nb_i
      pos <- pos + len
    }
  }
  
  data.table(
    cell_id          = id_order[focal_idx],
    neighbor_cell_id = id_order[neighbor_idx]
  )
}

#' Compute neighbor features for all source variables at once, fully vectorized.
#' Preserves the trained RF model — only adds columns to cell_data.
#'
#' @param cell_data data.frame/data.table with columns: id, year, and all neighbor_source_vars
#' @param id_order integer vector of cell IDs matching the nb object
#' @param neighbors spdep nb object (rook_neighbors_unique)
#' @param neighbor_source_vars character vector of variable names
#' @return cell_data (data.table) with new neighbor feature columns appended
compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          neighbors,
                                          neighbor_source_vars) {
  
  # Convert to data.table if needed (modifies in place for efficiency)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  # Assign a row index to every row in cell_data
  cell_data[, .row_idx := .I]
  
  # --- Step 1: Build edge list (once) ---
  edges <- build_edge_list(id_order, neighbors)
  
  # --- Step 2: Create a keyed lookup: (id, year) -> row_idx ---
  # We also need the source variable values for the neighbor rows.
  # Strategy: join edges with cell_data twice —
  #   (a) to get the focal row index (for grouping), and
  #   (b) to get the neighbor row's variable values.
  
  # Keyed lookup for focal rows: (cell_id, year) -> .row_idx
  # We expand edges by year via a join on cell_data's (id, year).
  
  # Focal side: for every (cell_id, year) in cell_data, get .row_idx
  focal <- cell_data[, .(focal_id = id, year, focal_row_idx = .row_idx)]
  setkey(focal, focal_id, year)
  
  # Expand edges × years: join edges on focal_id = cell_id
  # This gives us (focal_row_idx, neighbor_cell_id, year) for every edge × year
  setkey(edges, cell_id)
  
  # Merge: for each edge (cell_id -> neighbor_cell_id), attach all years
  # of the focal cell, yielding (focal_row_idx, neighbor_cell_id, year)
  expanded <- edges[focal, on = .(cell_id = focal_id), allow.cartesian = TRUE,
                    nomatch = NULL,
                    .(focal_row_idx = i.focal_row_idx,
                      neighbor_cell_id = x.neighbor_cell_id,
                      year = i.year)]
  
  # --- Step 3: Resolve neighbor rows ---
  # Build neighbor-side lookup: (id, year) -> row_idx + variable values
  keep_cols <- c("id", "year", ".row_idx", neighbor_source_vars)
  neighbor_lkp <- cell_data[, ..keep_cols]
  setnames(neighbor_lkp, "id", "neighbor_id")
  setnames(neighbor_lkp, ".row_idx", "neighbor_row_idx")
  setkey(neighbor_lkp, neighbor_id, year)
  
  # Join to resolve neighbor variable values
  joined <- neighbor_lkp[expanded,
                          on = .(neighbor_id = neighbor_cell_id, year),
                          nomatch = NA]
  # joined now has columns:
  #   neighbor_id, year, neighbor_row_idx, <source_vars>, focal_row_idx
  
  # --- Step 4: Grouped aggregation per (focal_row_idx) per variable ---
  for (var in neighbor_source_vars) {
    cat("Computing neighbor features for:", var, "\n")
    
    # Subset to non-NA values of this variable
    sub <- joined[!is.na(get(var)), .(focal_row_idx, val = get(var))]
    
    if (nrow(sub) == 0L) {
      cell_data[, paste0(var, "_neighbor_max")  := NA_real_]
      cell_data[, paste0(var, "_neighbor_min")  := NA_real_]
      cell_data[, paste0(var, "_neighbor_mean") := NA_real_]
      next
    }
    
    agg <- sub[, .(nb_max  = max(val),
                    nb_min  = min(val),
                    nb_mean = mean(val)),
               by = focal_row_idx]
    
    # Initialize new columns with NA
    max_col  <- paste0(var, "_neighbor_max")
    min_col  <- paste0(var, "_neighbor_min")
    mean_col <- paste0(var, "_neighbor_mean")
    
    cell_data[, (max_col)  := NA_real_]
    cell_data[, (min_col)  := NA_real_]
    cell_data[, (mean_col) := NA_real_]
    
    # Assign aggregated values back by row index
    cell_data[agg$focal_row_idx, (max_col)  := agg$nb_max]
    cell_data[agg$focal_row_idx, (min_col)  := agg$nb_min]
    cell_data[agg$focal_row_idx, (mean_col) := agg$nb_mean]
  }
  
  # Clean up helper column
  cell_data[, .row_idx := NULL]
  
  return(cell_data)
}

# ============================================================
# USAGE — drop-in replacement for the original outer loop
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  neighbors            = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# cell_data now has 15 new columns (3 per variable: _neighbor_max, _min, _mean)
# matching the original estimand exactly.
# The trained Random Forest model is untouched — just call predict() as before.
```

---

## Why This Preserves the Original Numerical Estimand

| Original operation | Replacement | Equivalence |
|---|---|---|
| `vals[idx]` where `idx` = neighbor row indices for a focal row | `data.table` join on `(neighbor_id, year)` resolving to the same rows | Identical row resolution; same values |
| `max`, `min`, `mean` over non-NA neighbor values | `data.table` grouped `max`, `min`, `mean` after filtering `!is.na` | Identical aggregation functions |
| Rows with no valid neighbors → `NA, NA, NA` | Rows absent from `agg` remain `NA_real_` | Identical NA handling |

The column names follow the pattern `{var}_neighbor_max/min/mean`. Adjust the naming convention if the original `compute_and_add_neighbor_features` used different suffixes — the RF model's `predict()` call will need matching column names.

---

## Expected Performance

| Stage | Original (est.) | Optimized (est.) |
|---|---|---|
| Neighbor lookup construction | ~60–80 hrs | ~30 sec (edge-list build + one keyed join) |
| Neighbor stats (5 vars) | ~6–10 hrs | ~2–5 min (5 grouped aggregations) |
| **Total** | **86+ hrs** | **~3–8 min** |

Peak RAM will spike during the `expanded` join (~25.8M edges × 28 years ≈ ~720M rows of 3 integer columns ≈ ~8.6 GB), which fits within 16 GB. If RAM is tight, process years in batches:

```r
# Optional: batch by year to reduce peak RAM
years <- sort(unique(cell_data$year))
for (yr in years) {
  # filter expanded and joined to this year only
  # ... aggregate and assign back ...
}
```

This trades a small amount of speed for significantly lower peak memory.