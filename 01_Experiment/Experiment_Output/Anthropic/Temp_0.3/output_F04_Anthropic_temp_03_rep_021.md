 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Looks up the cell's reference index via character coercion and named-vector lookup.
2. Retrieves neighbor cell IDs.
3. Constructs `paste`-based string keys for each neighbor–year combination.
4. Matches those keys against a named character vector (`idx_lookup`) of length ~6.46M.

Named-vector lookup in R is O(n) per query in the worst case (hash collisions aside, the overhead of repeated `paste`, character conversion, and subsetting on a 6.46M-name vector millions of times is enormous). With ~6.46M rows and an average of ~4 rook neighbors each, this produces roughly **25.8 million string constructions and lookups**, each against a 6.46M-element named vector. The `compute_neighbor_stats` function is comparatively cheap (just numeric subsetting), but `do.call(rbind, ...)` on 6.46M small vectors is also unnecessarily slow.

**Secondary issue:** `compute_neighbor_stats` is called 5 times (once per variable), each time iterating over 6.46M entries. This is redundant since the neighbor index sets don't change across variables.

## Optimization Strategy

1. **Replace the per-row `lapply` in `build_neighbor_lookup` with a vectorized, merge-based approach using `data.table`.** Instead of building a list of neighbor row indices per row, build a two-column edge table `(focal_row, neighbor_row)` in one vectorized pass. This eliminates millions of `paste` calls and named-vector lookups.

2. **Compute all neighbor stats in one grouped `data.table` aggregation** over the edge table, for all 5 variables simultaneously, avoiding 5 separate `lapply` passes over 6.46M list elements.

3. **Eliminate `do.call(rbind, ...)`** on millions of small vectors (which is O(n²) in memory copies).

**Expected speedup:** From 86+ hours to roughly 5–15 minutes on the same laptop.

## Optimized Working R Code

```r
library(data.table)

build_neighbor_features_fast <- function(cell_data, id_order, rook_neighbors_unique,
                                         neighbor_source_vars) {
  # Convert to data.table if not already; add a row index
  dt <- as.data.table(cell_data)
  dt[, .row_id := .I]

  # ---- Step 1: Build a complete (focal_row, neighbor_row) edge table ----

  # Map each cell id to its position in id_order
  id_to_ref <- data.table(
    id      = id_order,
    ref_idx = seq_along(id_order)
  )

  # Expand the nb object into an edge list: (focal_ref_idx, neighbor_ref_idx)
  # rook_neighbors_unique is a list of integer vectors (spdep nb object)
  focal_ref <- rep(seq_along(rook_neighbors_unique),
                   lengths(rook_neighbors_unique))
  neighbor_ref <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Remove 0-neighbor entries (nb encodes no-neighbor as integer(0), already handled by rep/unlist)
  # Map ref indices back to cell ids
  edges <- data.table(
    focal_id    = id_order[focal_ref],
    neighbor_id = id_order[neighbor_ref]
  )

  # ---- Step 2: Join edges with data to get (focal_row, neighbor_row) per year ----

  # Create a keyed lookup: (id, year) -> row_id
  lookup <- dt[, .(id, year, .row_id)]

  # For each edge (focal_id, neighbor_id), we need every year that the focal cell appears in.
  # Then find the neighbor's row in the same year.

  # First, get focal rows with their year
  # Merge edges with focal lookup to get (focal_row, focal_year, neighbor_id)
  setkey(lookup, id)
  focal_expanded <- lookup[edges, on = .(id = focal_id),
                           .(focal_row = .row_id,
                             year      = year,
                             neighbor_id = i.neighbor_id),
                           nomatch = NULL,
                           allow.cartesian = TRUE]

  # Now merge to get neighbor_row in the same year
  setnames(lookup, ".row_id", "neighbor_row")
  setkey(lookup, id, year)
  setkey(focal_expanded, neighbor_id, year)

  edge_table <- lookup[focal_expanded,
                       on = .(id = neighbor_id, year = year),
                       .(focal_row    = i.focal_row,
                         neighbor_row = neighbor_row),
                       nomatch = NA_integer_]

  # Drop edges where neighbor had no matching row (missing year)
  edge_table <- edge_table[!is.na(neighbor_row)]

  # ---- Step 3: Compute all neighbor stats in one vectorized pass ----

  # Extract neighbor values for all source variables at once
  # Build a matrix of neighbor values indexed by edge_table$neighbor_row
  for (var in neighbor_source_vars) {
    set(edge_table, j = var, value = dt[[var]][edge_table$neighbor_row])
  }

  # Group by focal_row and compute max, min, mean for each variable
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)),   na.rm = TRUE)),
      bquote(min(.(as.name(v)),   na.rm = TRUE)),
      bquote(mean(.(as.name(v)),  na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  # Build the aggregation call
  # Using a simpler, robust approach:
  stats <- edge_table[, {
    out <- list()
    for (v in neighbor_source_vars) {
      vals <- get(v)
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
  }, by = focal_row]

  # ---- Step 4: Merge stats back onto the original data ----

  # Rows with no neighbors at all won't appear in stats; they get NA (correct).
  dt[stats, on = .(.row_id = focal_row),
     (agg_names) := mget(agg_names)]

  # Replace infinite values from max/min of empty sets (safety)
  for (col in agg_names) {
    dt[is.infinite(get(col)), (col) := NA_real_]
  }

  # Remove helper column and return as data.frame if original was data.frame
  dt[, .row_id := NULL]

  return(as.data.frame(dt))
}

# ---- Usage (drop-in replacement for the original outer loop) ----

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_fast(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is used as-is downstream — no retraining needed.
# predict(rf_model, newdata = cell_data)
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | The same `max`, `min`, `mean` are computed over exactly the same neighbor sets (same rook contiguity, same year matching, same NA handling). |
| **Trained RF model** | No model object is touched; only the feature columns fed to `predict()` are constructed. Column names match the originals. |
| **Edge cases** | Cells with no neighbors (or no neighbor present in that year) receive `NA`, identical to the original `is.na` / `length == 0` guards. |

## Performance Accounting

| Step | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~6.46M `paste` + named-vector lookups → hours | One vectorized `data.table` join (~25.8M edge rows) → seconds |
| Stat computation | 5 × `lapply` over 6.46M list elements | Single grouped `data.table` aggregation → minutes |
| Result binding | `do.call(rbind, 6.46M)` × 5 | In-place `:=` column assignment |
| **Estimated total** | **86+ hours** | **~5–15 minutes** |

The key insight is replacing millions of scalar R-level operations (string construction, named-vector lookup, list element creation) with a small number of bulk `data.table` keyed joins and grouped aggregations that execute in compiled C code.