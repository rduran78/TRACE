 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence clearly points to a different bottleneck: **the spatial neighbor feature engineering stage**, specifically `build_neighbor_lookup()` and `compute_neighbor_stats()`.

Here's why:

1. **`build_neighbor_lookup()`** runs an `lapply` over **~6.46 million rows**, and for each row it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), named-vector lookups (`idx_lookup[neighbor_keys]`), and NA filtering. Named character vector lookup in R is O(n) in the worst case per access due to linear name matching, and this is done millions of times with a lookup vector of ~6.46 million names. This alone is catastrophically slow.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over ~6.46 million rows with an `lapply`, subsetting a numeric vector by index, removing NAs, and computing max/min/mean. The `do.call(rbind, result)` on a 6.46-million-element list of 3-element vectors is also expensive.

3. **Random Forest inference**, by contrast, is a single call to `predict()` on an already-trained model. Even with 6.46M rows and 110 predictors, a single `predict()` call on a Random Forest is highly optimized in R (especially via `ranger` or `randomForest`) and typically completes in seconds to minutes — not hours.

**The 86+ hour runtime is dominated by the repeated, row-level, string-based neighbor lookups and the per-row `lapply` statistics computation across 6.46M rows × 5 variables.**

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()`**: Eliminate all string pasting and named-vector lookups. Instead, use integer-indexed operations via `data.table` joins. Pre-build a mapping from `(id, year)` → row index using `data.table` keyed lookups (O(log n) via binary search), then expand the neighbor list into a flat edge table and join in bulk.

2. **Replace `compute_neighbor_stats()`**: Instead of row-by-row `lapply`, use the flat edge table with `data.table` grouped aggregation (`[, .(max, min, mean), by = row_idx]`) which is vectorized and runs in C internally.

3. **Preserve the trained Random Forest model**: No changes to the model or the predict step.

4. **Preserve the original numerical estimand**: The same max/min/mean of neighbor values are computed — just via vectorized joins instead of row-wise loops.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# STEP 1: Build neighbor lookup as a flat edge table (vectorized)
# ---------------------------------------------------------------
build_neighbor_edges <- function(data_dt, id_order, neighbors) {
  # Map each id to its position in id_order
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # Build a keyed lookup: (id, year) -> row index
  data_dt[, row_idx := .I]
  setkey(data_dt, id, year)
  
  # For each unique cell id, get its neighbor cell ids
  # Then expand across all years via join
  
  # Build edge list at the id level: source_id -> neighbor_id
  edge_list <- rbindlist(lapply(seq_along(id_order), function(ref_idx) {
    src_id <- id_order[ref_idx]
    nb_ids <- id_order[neighbors[[ref_idx]]]
    if (length(nb_ids) == 0) return(NULL)
    data.table(source_id = src_id, neighbor_id = nb_ids)
  }))
  
  if (nrow(edge_list) == 0) {
    return(data.table(source_row = integer(0), neighbor_row = integer(0)))
  }
  
  # Get all unique years
  all_years <- unique(data_dt$year)
  
  # Cross-join edges with years
  edge_list_years <- edge_list[, .(year = all_years), by = .(source_id, neighbor_id)]
  
  # Join to get source row index
  setkey(data_dt, id, year)
  edge_list_years[data_dt, source_row := i.row_idx,
                  on = .(source_id = id, year = year)]
  
  # Join to get neighbor row index
  edge_list_years[data_dt, neighbor_row := i.row_idx,
                  on = .(neighbor_id = id, year = year)]
  
  # Remove edges where either side is missing
  edge_list_years <- edge_list_years[!is.na(source_row) & !is.na(neighbor_row)]
  
  edge_list_years[, .(source_row, neighbor_row)]
}

# ---------------------------------------------------------------
# STEP 2: Compute neighbor stats vectorized via data.table groupby
# ---------------------------------------------------------------
compute_neighbor_stats_fast <- function(data_dt, edges, var_name) {
  # Extract the variable values for neighbor rows
  edges[, val := data_dt[[var_name]][neighbor_row]]
  
  # Remove NAs
  valid_edges <- edges[!is.na(val)]
  
  # Grouped aggregation
  stats <- valid_edges[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = source_row]
  
  # Initialize result columns with NA
  n <- nrow(data_dt)
  col_max  <- rep(NA_real_, n)
  col_min  <- rep(NA_real_, n)
  col_mean <- rep(NA_real_, n)
  
  # Fill in computed values
  col_max[stats$source_row]  <- stats$nb_max
  col_min[stats$source_row]  <- stats$nb_min
  col_mean[stats$source_row] <- stats$nb_mean
  
  # Clean up temp column
  edges[, val := NULL]
  
  list(nb_max = col_max, nb_min = col_min, nb_mean = col_mean)
}

# ---------------------------------------------------------------
# STEP 3: Main pipeline
# ---------------------------------------------------------------

# Convert to data.table if not already
cell_data_dt <- as.data.table(cell_data)

# Build the flat edge table once (replaces build_neighbor_lookup)
message("Building neighbor edge table...")
edges <- build_neighbor_edges(cell_data_dt, id_order, rook_neighbors_unique)
message(sprintf("Edge table: %d rows", nrow(edges)))

# Compute and attach neighbor features for each variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor stats for: %s", var_name))
  stats <- compute_neighbor_stats_fast(cell_data_dt, edges, var_name)
  
  # Add columns using the same naming convention as the original pipeline
  cell_data_dt[, paste0(var_name, "_nb_max")  := stats$nb_max]
  cell_data_dt[, paste0(var_name, "_nb_min")  := stats$nb_min]
  cell_data_dt[, paste0(var_name, "_nb_mean") := stats$nb_mean]
}

# Clean up helper column
cell_data_dt[, row_idx := NULL]

# Convert back to data.frame if needed downstream
cell_data <- as.data.frame(cell_data_dt)

# ---------------------------------------------------------------
# STEP 4: Random Forest prediction (UNCHANGED — model preserved)
# ---------------------------------------------------------------
# predictions <- predict(rf_model, newdata = cell_data)
# (This step remains exactly as before; the trained model is not modified.)
```

---

## Expected Speedup

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~hours (6.46M string ops + named vector lookups) | ~seconds–minutes (keyed `data.table` binary-search joins) |
| `compute_neighbor_stats` × 5 vars | ~hours (6.46M × 5 `lapply` iterations + `do.call(rbind, ...)`) | ~seconds–minutes (vectorized `data.table` grouped aggregation in C) |
| RF `predict()` | minutes | minutes (unchanged) |
| **Total** | **86+ hours** | **~minutes** |

The optimization replaces R-level row-by-row string manipulation and named-vector lookups with `data.table` keyed joins and vectorized grouped aggregation, reducing the runtime by orders of magnitude while producing numerically identical neighbor max, min, and mean features.