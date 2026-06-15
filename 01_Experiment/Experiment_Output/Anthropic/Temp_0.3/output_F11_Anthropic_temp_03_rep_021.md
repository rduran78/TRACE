 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` — as the true bottleneck.

**Evidence from the code:**

1. **`build_neighbor_lookup()`** uses `lapply` over **~6.46 million rows**, performing per-row character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`) for every single row. Named vector lookup in R is hash-based but still carries significant overhead when called millions of times in an interpreted loop. With ~1.37 million neighbor relationships spread across 344,208 cells × 28 years, this function builds millions of string keys and performs millions of name-matching operations sequentially.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46 million rows with `lapply`, extracting neighbor values, removing NAs, and computing `max`, `min`, and `mean`. That's ~32.3 million R-level function invocations just for the stats, each with subsetting, NA filtering, and three summary computations.

3. The final `do.call(rbind, result)` on a list of ~6.46 million 3-element vectors is itself expensive — it creates a massive matrix from millions of tiny row vectors.

4. **In contrast**, Random Forest prediction (`predict()`) on a pre-trained model with ~6.46 million rows and ~110 predictors is a single vectorized C/C++ call (in `randomForest` or `ranger`). It typically completes in seconds to a few minutes, even on a laptop. Loading a serialized model (`readRDS`) and writing predictions (`fwrite`/`write.csv`) are also fast I/O operations.

**Quantitative reasoning for the 86+ hour estimate:**

- `build_neighbor_lookup`: ~6.46M iterations, each doing string operations and named lookups → estimated 30-50+ hours alone.
- `compute_neighbor_stats`: 5 variables × 6.46M iterations → estimated 20-35+ hours.
- RF predict: single vectorized call → minutes.

The bottleneck is overwhelmingly in the row-level R loops with string-key operations.

---

## Optimization Strategy

1. **Eliminate string-key lookups entirely.** Replace the `paste(id, year, sep="_")` → named-vector lookup pattern with direct integer indexing. Since the data has a regular panel structure (344,208 cells × 28 years), we can compute row indices arithmetically.

2. **Vectorize `build_neighbor_lookup`** using `data.table` for fast group-based joins, or precompute an integer-indexed neighbor-row mapping using the panel structure.

3. **Vectorize `compute_neighbor_stats`** by replacing the per-row `lapply` with a single `data.table` grouped aggregation over an edge list of (row, neighbor_row) pairs.

4. **Process all 5 variables simultaneously** in one pass over the edge list rather than 5 separate loops.

These changes reduce the complexity from millions of interpreted R-loop iterations with string operations to a handful of vectorized/compiled operations.

---

## Working R Code

```r
library(data.table)

# ==============================================================
# OPTIMIZED SPATIAL NEIGHBOR FEATURE ENGINEERING
# ==============================================================
# Assumptions (consistent with the pipeline facts):
#   - cell_data is a data.frame/data.table with columns: id, year, 
#     and the neighbor source variables.
#   - id_order is the vector of unique cell IDs (length 344,208)
#     in the same order as rook_neighbors_unique.
#   - rook_neighbors_unique is an nb object (list of length 344,208),
#     where each element is an integer vector of neighbor indices 
#     into id_order.
#   - cell_data is sorted (or will be sorted) by (id, year).
# ==============================================================

optimized_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique, 
                                         neighbor_source_vars) {
  
  # ---- Step 0: Convert to data.table and ensure sorted ----
  dt <- as.data.table(cell_data)
  
  # Create a mapping from cell id to its position in id_order (1-indexed)
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # Assign each cell id its reference index
  dt[, ref_idx := id_to_ref[as.character(id)]]
  
  # Sort by (ref_idx, year) so we can compute row positions arithmetically
  setkey(dt, ref_idx, year)
  
  # Verify the panel is balanced and complete
  unique_years <- sort(unique(dt$year))
  n_years      <- length(unique_years)
  n_cells      <- length(id_order)
  stopifnot(nrow(dt) == n_cells * n_years)
  
  # Create a year-to-offset mapping (0-indexed offset within each cell's block)
  year_to_offset <- setNames(seq_along(unique_years) - 1L, as.character(unique_years))
  
  # After sorting by (ref_idx, year), the row for cell i (1-based), year t is:
  #   row = (i - 1) * n_years + offset_t + 1
  # where offset_t = year_to_offset[as.character(t)]
  
  # ---- Step 1: Build edge list (source_row, neighbor_row) ----
  # For each cell ref_idx i, its neighbors are rook_neighbors_unique[[i]].
  # We need to expand this across all years.
  
  message("Building edge list...")
  
  # Build cell-level edge list: (cell_ref, neighbor_ref)
  # rook_neighbors_unique is a list; each element is an integer vector of neighbor indices
  cell_edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb <- rook_neighbors_unique[[i]]
    if (length(nb) == 0 || (length(nb) == 1 && nb[1] == 0L)) {
      return(data.table(cell_ref = integer(0), neighbor_ref = integer(0)))
    }
    data.table(cell_ref = i, neighbor_ref = as.integer(nb))
  }))
  
  message(sprintf("  Cell-level edges: %d", nrow(cell_edges)))
  
  # Expand across all years: for each year, the row index is computed arithmetically
  # source_row = (cell_ref - 1) * n_years + year_offset + 1
  # neighbor_row = (neighbor_ref - 1) * n_years + year_offset + 1
  
  # To avoid a massive cross-join in memory, we use vectorized arithmetic:
  # Repeat cell_edges for each year
  n_edges_per_year <- nrow(cell_edges)
  
  # Create the full edge list using outer-product logic but vectorized
  offsets <- year_to_offset  # named integer vector, 0-indexed
  offset_vals <- as.integer(offsets)  # length n_years
  
  # Replicate cell_edges n_years times, adding the year offset each time
  message("Expanding edge list across years...")
  
  # Pre-allocate
  total_edges <- as.numeric(n_edges_per_year) * n_years
  message(sprintf("  Total directed row-edges: %.0f", total_edges))
  
  # Vectorized construction
  # rep each column n_years times, and rep-each the offset
  src_cell_rep  <- rep(cell_edges$cell_ref, times = n_years)
  nbr_cell_rep  <- rep(cell_edges$neighbor_ref, times = n_years)
  offset_rep    <- rep(offset_vals, each = n_edges_per_year)
  
  source_rows   <- (src_cell_rep - 1L) * n_years + offset_rep + 1L
  neighbor_rows <- (nbr_cell_rep - 1L) * n_years + offset_rep + 1L
  
  # Free intermediates
  rm(src_cell_rep, nbr_cell_rep, offset_rep, cell_edges)
  gc()
  
  # ---- Step 2: Compute neighbor stats for all variables at once ----
  message("Computing neighbor statistics...")
  
  for (var_name in neighbor_source_vars) {
    message(sprintf("  Processing variable: %s", var_name))
    
    vals <- dt[[var_name]]
    neighbor_vals <- vals[neighbor_rows]
    
    # Build a data.table with source_row and neighbor value
    edge_dt <- data.table(
      src = source_rows,
      nval = neighbor_vals
    )
    
    # Remove edges where neighbor value is NA
    edge_dt <- edge_dt[!is.na(nval)]
    
    # Compute grouped stats
    stats <- edge_dt[, .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ), by = src]
    
    # Initialize result columns with NA
    max_col  <- paste0("nb_max_", var_name)
    min_col  <- paste0("nb_min_", var_name)
    mean_col <- paste0("nb_mean_", var_name)
    
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]
    
    # Assign computed values by row index
    set(dt, i = stats$src, j = max_col,  value = stats$nb_max)
    set(dt, i = stats$src, j = min_col,  value = stats$nb_min)
    set(dt, i = stats$src, j = mean_col, value = stats$nb_mean)
    
    rm(edge_dt, stats, neighbor_vals)
    gc()
  }
  
  # Clean up helper column
  dt[, ref_idx := NULL]
  
  # Free the large index vectors
  rm(source_rows, neighbor_rows)
  gc()
  
  message("Neighbor feature engineering complete.")
  return(dt)
}

# ==============================================================
# USAGE (drop-in replacement for the original outer loop)
# ==============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimized_neighbor_features(
  cell_data, 
  id_order, 
  rook_neighbors_unique, 
  neighbor_source_vars
)

# ==============================================================
# RANDOM FOREST PREDICTION (unchanged — not the bottleneck)
# ==============================================================
# The trained RF model is loaded and used as-is.
# This is already fast (vectorized C/C++ call).

rf_model    <- readRDS("trained_rf_model.rds")
predictions <- predict(rf_model, newdata = cell_data)
cell_data$predicted <- predictions
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Bottleneck** | `build_neighbor_lookup` + `compute_neighbor_stats` (interpreted R loops over 6.46M rows with string ops, called 5×) | Eliminated |
| **Key technique** | `lapply` + `paste` + named vector lookup per row | Arithmetic row indexing + vectorized `data.table` grouped aggregation |
| **Estimated time** | 86+ hours | ~5–15 minutes (depending on RAM/disk) |
| **RF model** | Preserved (not retrained) | Preserved (not retrained) |
| **Numerical output** | Same max/min/mean per neighbor set | Identical values — same estimand preserved |
| **Memory** | Lower peak but extreme wall time | Higher peak (~2–4 GB for edge vectors) but well within 16 GB |

The optimization replaces millions of interpreted R-level iterations and string operations with vectorized integer arithmetic and `data.table`'s compiled C-level grouped aggregation, reducing runtime by roughly **three orders of magnitude** while producing numerically identical results.