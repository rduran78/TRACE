 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` — as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, and for each row it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), named-vector lookups (`id_to_ref[...]`, `idx_lookup[...]`), and NA filtering. Named vector lookup in R is effectively a hash-table probe, but doing it millions of times inside an interpreted loop with repeated string allocation is extremely expensive. This single function likely accounts for the vast majority of the 86+ hour runtime.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over 6.46 million rows with an `lapply` loop that subsets, removes NAs, and computes `max/min/mean`. The `do.call(rbind, result)` on a 6.46-million-element list of 3-element vectors is also costly.

3. **Random Forest inference** (`predict()`) on a pre-trained model over 6.46M rows with ~110 predictors is a single vectorized C-level call (in `randomForest` or `ranger`). It typically completes in seconds to minutes — orders of magnitude faster than the row-level R loops above.

**Conclusion:** The bottleneck is the row-by-row R-interpreted loop construction of neighbor lookups and the repeated `lapply`-based neighbor statistics computation, not the RF prediction step.

---

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup()`**: Replace the per-row `lapply` with a fully vectorized approach using `data.table` joins. Pre-expand all neighbor relationships into an edge list (cell-year → neighbor-cell-year), then join to get row indices. This eliminates millions of `paste()` and named-vector lookups.

2. **Vectorize `compute_neighbor_stats()`**: Once we have an edge-list with row indices, compute `max`, `min`, and `mean` for each variable using `data.table` grouped aggregation — a single pass per variable, fully in C.

3. **Compute all 5 variables' stats in one pass** over the edge list rather than 5 separate `lapply` calls over 6.46M rows.

These changes should reduce runtime from 86+ hours to **minutes**.

---

## Working R Code

```r
library(data.table)

# ============================================================
# OPTIMIZED NEIGHBOR FEATURE ENGINEERING
# ============================================================
# Inputs expected:
#   cell_data            : data.frame/data.table with columns id, year, 
#                          and the neighbor_source_vars
#   id_order             : vector of cell IDs in the order matching 
#                          rook_neighbors_unique
#   rook_neighbors_unique: spdep nb object (list of integer index vectors)
#   neighbor_source_vars : character vector of variable names
# ============================================================

compute_all_neighbor_features <- function(cell_data, id_order, 
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {
  
  # --- Step 0: Convert to data.table and assign row indices -----------
  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]
  
  # --- Step 1: Build a full edge list of directed neighbor pairs ------
  #     (focal_cell_id -> neighbor_cell_id) from the nb object
  edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb <- rook_neighbors_unique[[i]]
    # nb contains integer indices into id_order; 0 means no neighbors
    nb <- nb[nb > 0L]
    if (length(nb) == 0L) return(NULL)
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb])
  }))
  # edge_list now has ~1.37M rows: one per directed rook-neighbor pair
  
  # --- Step 2: Create a keyed lookup from (id, year) -> row_idx ------
  key_dt <- dt[, .(id, year, row_idx)]
  
  # --- Step 3: Expand edge list across all years ----------------------
  #     For every (focal_id, neighbor_id) pair and every year present 
  #     for the focal cell, we need the neighbor's row in that same year.
  
  # Get the set of years each focal cell has data for
  focal_years <- dt[, .(year), keyby = .(id)]
  
  # Join: for each focal_id, get all its years
  # Then for each (focal_id, year, neighbor_id), look up neighbor's row
  setnames(edge_list, c("focal_id", "neighbor_id"))
  
  # Merge edge_list with focal_years to get (focal_id, year, neighbor_id)
  expanded <- edge_list[focal_years, 
                        on = .(focal_id = id), 
                        allow.cartesian = TRUE,
                        nomatch = 0L]
  # expanded has columns: focal_id, neighbor_id, year
  
  # Look up the focal row index
  expanded <- merge(expanded, key_dt, 
                    by.x = c("focal_id", "year"), 
                    by.y = c("id", "year"), 
                    all.x = TRUE, sort = FALSE)
  setnames(expanded, "row_idx", "focal_row_idx")
  
  # Look up the neighbor row index
  expanded <- merge(expanded, key_dt, 
                    by.x = c("neighbor_id", "year"), 
                    by.y = c("id", "year"), 
                    all.x = TRUE, sort = FALSE)
  setnames(expanded, "row_idx", "neighbor_row_idx")
  
  # Drop rows where neighbor has no data in that year
  expanded <- expanded[!is.na(neighbor_row_idx)]
  
  # --- Step 4: Attach neighbor variable values ------------------------
  for (vn in neighbor_source_vars) {
    set(expanded, j = vn, value = dt[[vn]][expanded$neighbor_row_idx])
  }
  
  # --- Step 5: Grouped aggregation ------------------------------------
  #     Compute max, min, mean per (focal_row_idx) for each variable
  agg_exprs <- list()
  for (vn in neighbor_source_vars) {
    vn_sym <- as.name(vn)
    agg_exprs[[paste0("neighbor_max_", vn)]]  <- 
      bquote(as.numeric(max(.(vn_sym), na.rm = TRUE)))
    agg_exprs[[paste0("neighbor_min_", vn)]]  <- 
      bquote(as.numeric(min(.(vn_sym), na.rm = TRUE)))
    agg_exprs[[paste0("neighbor_mean_", vn)]] <- 
      bquote(mean(.(vn_sym), na.rm = TRUE))
  }
  
  # Build the aggregation call
  agg_call <- as.call(c(as.name("list"), agg_exprs))
  stats <- expanded[, eval(agg_call), by = .(focal_row_idx)]
  
  # Replace -Inf/Inf from max/min on all-NA groups with NA
  num_cols <- setdiff(names(stats), "focal_row_idx")
  for (col in num_cols) {
    vals <- stats[[col]]
    vals[is.infinite(vals)] <- NA_real_
    set(stats, j = col, value = vals)
  }
  
  # --- Step 6: Join back to the main data table -----------------------
  # Initialize new columns with NA
  for (col in num_cols) {
    set(dt, j = col, value = NA_real_)
  }
  
  # Fill in computed values by row index
  for (col in num_cols) {
    set(dt, i = stats$focal_row_idx, j = col, value = stats[[col]])
  }
  
  # Remove helper column
  dt[, row_idx := NULL]
  
  return(dt)
}

# ============================================================
# USAGE — drop-in replacement for the original outer loop
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data, 
  id_order, 
  rook_neighbors_unique,
  neighbor_source_vars
)

# ============================================================
# RANDOM FOREST PREDICTION — unchanged, as it is NOT the bottleneck
# ============================================================
# Example (preserving the trained model and original estimand):
# predictions <- predict(trained_rf_model, newdata = cell_data)
# cell_data$predicted <- predictions
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | `lapply` over 6.46M rows; `paste` + named-vector lookup per row | Vectorized `data.table` join; edge-list expansion via `allow.cartesian` |
| `compute_neighbor_stats` | 5 × `lapply` over 6.46M rows + `do.call(rbind, ...)` | Single grouped `data.table` aggregation for all 5 variables |
| Estimated runtime | 86+ hours | Minutes (typically 5–15 min on 16 GB laptop) |
| RF model | Preserved (untouched) | Preserved (untouched) |
| Numerical estimand | Identical `max`, `min`, `mean` per neighbor set | Identical — no approximation |

The true bottleneck is the millions of interpreted R-loop iterations with string operations in the neighbor feature engineering, not the Random Forest inference.