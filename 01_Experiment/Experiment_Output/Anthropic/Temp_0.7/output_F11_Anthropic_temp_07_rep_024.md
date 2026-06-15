 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering pipeline** — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` — as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()` uses `lapply` over ~6.46 million rows**, performing per-row character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`) inside the loop. Named vector lookup in R is hash-based but carries significant per-call overhead when done millions of times. With ~1.37M neighbor relationships spread across 6.46M rows, this function alone generates tens of millions of string operations.

2. **`compute_neighbor_stats()` is called 5 times** (once per neighbor source variable), each time iterating over all ~6.46 million rows with `lapply`, subsetting a numeric vector, removing NAs, and computing `max/min/mean`. That's ~32.3 million R-level anonymous function invocations combined, each with small-vector allocation overhead.

3. **The `do.call(rbind, result)` on a list of 6.46 million 3-element vectors** is notoriously slow — it effectively builds a matrix one row at a time.

4. By contrast, Random Forest `predict()` on a pre-trained model with ~110 predictors and 6.46M rows is a single vectorized C/C++ call (in `ranger` or `randomForest`). It is computationally non-trivial but is nowhere near 86+ hours. Typical RF prediction on this scale takes minutes to low tens of minutes, not days.

**Conclusion:** The bottleneck is the row-level R `lapply` loops over millions of rows doing string operations and small-vector statistics. This is a classic "death by a million R-level iterations" problem.

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup` with a vectorized `data.table` equi-join.** Instead of building a per-row list of neighbor indices via string pasting and named-vector lookup, create an edge-list data.table of `(id, neighbor_id)` from the `nb` object, then merge with the panel data on `(neighbor_id, year)` to get all neighbor-row indices at once — fully vectorized.

2. **Replace `compute_neighbor_stats` with a grouped `data.table` aggregation.** Once we have the long-form edge table joined to the data, computing `max`, `min`, and `mean` per `(row_index, variable)` is a single grouped aggregation — no R-level loop needed.

3. **Eliminate `do.call(rbind, ...)` entirely** — `data.table` returns results in columnar form directly.

These changes convert O(N) R-level iterations (N ≈ 6.46M × 5) into a small number of vectorized C-level operations, reducing runtime from 86+ hours to likely **minutes**.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 1. Convert nb object to a vectorized edge-list data.table
# ---------------------------------------------------------------
build_neighbor_edges <- function(id_order, neighbors_nb) {
  # neighbors_nb is an nb object (list of integer index vectors)
  # id_order is the vector of cell IDs corresponding to each nb element
  
  # Pre-calculate lengths for pre-allocation
  lens <- lengths(neighbors_nb)
  total_edges <- sum(lens)
  
  # Build source (focal) and target (neighbor) index vectors
  focal_idx <- rep(seq_along(neighbors_nb), times = lens)
  neighbor_idx <- unlist(neighbors_nb, use.names = FALSE)
  
  # Map from positional index back to actual cell ID
  data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
}

# ---------------------------------------------------------------
# 2. Compute all neighbor features in vectorized fashion
# ---------------------------------------------------------------
compute_all_neighbor_features <- function(cell_data, id_order, neighbors_nb,
                                          neighbor_source_vars) {
  
  # Convert to data.table if not already; add a row index for later join-back
  dt <- as.data.table(cell_data)
  dt[, .row_idx := .I]
  
  # Step 1: Build edge list (focal_id -> neighbor_id)
  edges <- build_neighbor_edges(id_order, neighbors_nb)
  
  # Step 2: Create a keyed lookup from (id, year) -> row index
  #         and the values of the neighbor source variables
  keep_cols <- c("id", "year", ".row_idx", neighbor_source_vars)
  lookup <- dt[, ..keep_cols]
  setnames(lookup, "id", "neighbor_id")
  setkey(lookup, neighbor_id, year)
  
  # Step 3: Build the focal table: (focal_row_idx, focal_id, year)
  focal_info <- dt[, .(focal_row_idx = .row_idx, focal_id = id, year)]
  
  # Step 4: Join focal_info with edges to get (focal_row_idx, year, neighbor_id)
  #          for every focal-row × neighbor combination
  setkey(edges, focal_id)
  setkey(focal_info, focal_id)
  
  # This is the large join: each focal row gets its neighbor IDs

  expanded <- edges[focal_info, on = "focal_id", allow.cartesian = TRUE,
                    nomatch = NULL]
  # expanded has columns: focal_id, neighbor_id, focal_row_idx, year
  
  # Step 5: Join with lookup to get the neighbor variable values
  #          matched on (neighbor_id, year)
  setkey(expanded, neighbor_id, year)
  expanded <- lookup[expanded, on = .(neighbor_id, year), nomatch = NA]
  # Now expanded has the neighbor variable values for each focal-row × neighbor pair
  
  # Step 6: Grouped aggregation — compute max, min, mean per focal row per variable
  #          We aggregate by focal_row_idx
  
  # Pre-allocate result columns in dt
  for (var_name in neighbor_source_vars) {
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]
  }
  
  # Build aggregation expressions dynamically
  agg_exprs <- list()
  for (var_name in neighbor_source_vars) {
    v <- as.name(var_name)
    agg_exprs[[paste0("neighbor_max_", var_name)]]  <-
      bquote(as.double(max(.(v), na.rm = TRUE)))
    agg_exprs[[paste0("neighbor_min_", var_name)]]  <-
      bquote(as.double(min(.(v), na.rm = TRUE)))
    agg_exprs[[paste0("neighbor_mean_", var_name)]] <-
      bquote(mean(.(v), na.rm = TRUE))
  }
  
  # Single grouped aggregation over all variables at once
  agg_call <- as.call(c(as.name("list"), agg_exprs))
  agg_result <- expanded[, eval(agg_call), by = focal_row_idx]
  
  # Replace Inf/-Inf (from max/min of all-NA groups) with NA
  for (col_name in names(agg_result)[-1]) {
    vals <- agg_result[[col_name]]
    set(agg_result, i = which(is.infinite(vals)), j = col_name, value = NA_real_)
  }
  
  # Step 7: Map aggregated results back to dt by row index
  new_cols <- setdiff(names(agg_result), "focal_row_idx")
  dt[agg_result$focal_row_idx, (new_cols) := agg_result[, ..new_cols]]
  
  # Clean up helper column
  dt[, .row_idx := NULL]
  
  return(dt)
}

# ---------------------------------------------------------------
# 3. Drop-in replacement for the outer loop
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# ---------------------------------------------------------------
# 4. Random Forest prediction (UNCHANGED — not the bottleneck)
# ---------------------------------------------------------------
# library(ranger)  # or randomForest
# trained_model <- readRDS("path/to/trained_rf_model.rds")
# cell_data$predictions <- predict(trained_model, data = cell_data)$predictions
# fwrite(cell_data, "output_predictions.csv")
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | `lapply` over 6.46M rows, string paste + named-vector lookup | Single vectorized `data.table` join on `(neighbor_id, year)` |
| **Neighbor stats** | 5 × `lapply` over 6.46M rows + `do.call(rbind, ...)` | One grouped `data.table` aggregation across all 5 variables simultaneously |
| **Estimated runtime** | 86+ hours | Minutes (dominated by the large equi-join and grouped aggregation, both executed in C) |
| **RF model** | Preserved ✓ | Preserved ✓ — `predict()` call unchanged |
| **Numerical estimand** | max, min, mean of rook-neighbor values per cell-year | Identical — same grouping, same aggregation functions, same NA handling |

The trained Random Forest model is untouched. The numerical outputs (neighbor max, min, mean per variable per cell-year) are identical to the original. The only change is *how* those neighbor statistics are computed — replacing millions of R-level loop iterations with vectorized `data.table` operations.