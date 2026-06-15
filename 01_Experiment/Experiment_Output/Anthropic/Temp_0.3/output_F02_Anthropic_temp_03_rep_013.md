 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting and named-vector lookups. Named vector lookups in R via character keys are O(n) hash lookups, but doing this millions of times with string concatenation inside the loop creates enormous overhead. The `compute_neighbor_stats` function then iterates over the same 6.46 million entries again, once per variable. Combined: ~6.46M × (lookup construction + 5 variable passes) = billions of R-level interpreted operations.

**Specific problems:**

1. **`build_neighbor_lookup`**: `paste()` and named-vector indexing inside a per-row `lapply` over 6.46M rows is extremely slow. The `idx_lookup` named vector has 6.46M entries, making each character-key lookup expensive at scale.
2. **`compute_neighbor_stats`**: `lapply` over 6.46M elements, each allocating a small vector and calling `max/min/mean`, is slow due to R's per-call overhead and lack of vectorization.
3. **Memory**: Storing a list of 6.46M integer vectors (`neighbor_lookup`) plus intermediate string vectors consumes many gigabytes.
4. **No parallelism or vectorization**: Everything is serial, interpreted R.

---

## Optimization Strategy

### Key Insight
The neighbor relationships are **cell-level** (not cell-year-level). A cell's rook neighbors are the same in every year. Therefore, we should:

1. **Work at the cell level for neighbor topology** — build a sparse adjacency structure over 344K cells, not 6.46M cell-years.
2. **Use `data.table` for fast indexed joins by year** — instead of string-key lookups, join neighbor cell IDs by year using integer keys.
3. **Vectorize aggregation** — use `data.table` grouped operations (`[, .(max, min, mean), by=...]`) instead of per-row `lapply`.
4. **Process all variables in one pass** if possible, or at least use vectorized column operations.

### Architecture

```
cell_data (data.table, keyed on id + year)
    ↓
edges table: (id, neighbor_id) — 1.37M rows, cell-level
    ↓
Join edges × cell_data on (neighbor_id = id, year) → neighbor values
    ↓
Group by (id, year) → compute max, min, mean per variable
    ↓
Join aggregated stats back to cell_data
```

This replaces 6.46M-element R lists and per-row `lapply` with a single vectorized `data.table` merge + grouped aggregation. Expected runtime: **minutes, not days**. Memory: the expanded edge-year table will have ~1.37M edges × 28 years ≈ 38.4M rows, which is manageable at ~16 GB RAM especially with integer/double columns only.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 0. Convert cell_data to data.table if not already
# ---------------------------------------------------------------
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# ---------------------------------------------------------------
# 1. Build a cell-level edge list from the nb object
#    rook_neighbors_unique is a list of length 344,208;
#    id_order is the vector mapping list index → cell id.
# ---------------------------------------------------------------
build_edge_table <- function(id_order, neighbors) {
  # neighbors is an nb object: list of integer vectors (indices into id_order)
  from_ids <- rep(id_order, times = lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)
  # Remove the 0-neighbor sentinel if present (spdep uses 0L for no neighbors)
  valid    <- to_idx > 0L
  edges    <- data.table(
    id          = from_ids[valid],
    neighbor_id = id_order[to_idx[valid]]
  )
  return(edges)
}

edges <- build_edge_table(id_order, rook_neighbors_unique)
cat("Edge table rows:", nrow(edges), "\n")
# Expected: ~1,373,394

# ---------------------------------------------------------------
# 2. Vectorized neighbor-stat computation for all variables at once
# ---------------------------------------------------------------
compute_all_neighbor_features <- function(cell_data, edges, neighbor_source_vars) {

  # Subset cell_data to only the columns we need for the join
  join_cols <- c("id", "year", neighbor_source_vars)
  nbr_vals <- cell_data[, ..join_cols]

  # Rename 'id' to 'neighbor_id' so we can join on the neighbor side
  setnames(nbr_vals, "id", "neighbor_id")

  # Merge: for every (id, year) pair, look up each neighbor's values in that year
  # edges has (id, neighbor_id); nbr_vals has (neighbor_id, year, var1, var2, ...)
  # Result: one row per (id, year, neighbor_id) with the neighbor's variable values
  setkey(nbr_vals, neighbor_id, year)
  setkey(edges, neighbor_id)

  # This is the big join — ~1.37M edges × 28 years ≈ 38.4M rows
  expanded <- edges[nbr_vals, on = "neighbor_id", allow.cartesian = TRUE, nomatch = NULL]
  # expanded now has columns: id, neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2

  # ---------------------------------------------------------------
  # 3. Grouped aggregation: max, min, mean per (id, year) per variable
  # ---------------------------------------------------------------
  # Build aggregation expressions dynamically
  agg_exprs <- list()
  for (v in neighbor_source_vars) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("nbr_max_", v)]]  <- bquote(as.double(max(.(v_sym), na.rm = TRUE)))
    agg_exprs[[paste0("nbr_min_", v)]]  <- bquote(as.double(min(.(v_sym), na.rm = TRUE)))
    agg_exprs[[paste0("nbr_mean_", v)]] <- bquote(mean(.(v_sym), na.rm = TRUE))
  }

  # Convert to a single call
  agg_call <- as.call(c(as.name("list"), agg_exprs))

  cat("Aggregating neighbor stats...\n")
  stats <- expanded[, eval(agg_call), by = .(id, year)]

  # Replace Inf/-Inf (from max/min on all-NA) with NA
  for (col_name in names(stats)) {
    if (col_name %in% c("id", "year")) next
    vals <- stats[[col_name]]
    set(stats, i = which(is.infinite(vals)), j = col_name, value = NA_real_)
  }

  return(stats)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

stats <- compute_all_neighbor_features(cell_data, edges, neighbor_source_vars)

# ---------------------------------------------------------------
# 4. Join the aggregated neighbor features back to cell_data
# ---------------------------------------------------------------
# Remove any pre-existing neighbor columns to avoid duplication
nbr_col_names <- setdiff(names(stats), c("id", "year"))
existing_nbr_cols <- intersect(names(cell_data), nbr_col_names)
if (length(existing_nbr_cols) > 0) {
  cell_data[, (existing_nbr_cols) := NULL]
}

setkey(cell_data, id, year)
setkey(stats, id, year)

cell_data <- stats[cell_data, on = .(id, year)]
# Left join: all original rows preserved; cells with no neighbors get NA (correct)

cat("Done. cell_data now has", ncol(cell_data), "columns and", nrow(cell_data), "rows.\n")

# ---------------------------------------------------------------
# 5. Predict with the existing trained Random Forest (unchanged)
# ---------------------------------------------------------------
# The trained RF model object is preserved as-is.
# Ensure the new neighbor column names match what the model expects.
# If the original code produced columns named e.g. "ntl_nbr_max", "ntl_nbr_min",
# "ntl_nbr_mean", rename accordingly:
#
# Example renaming (adjust to match your model's expected feature names):
# setnames(cell_data, "nbr_max_ntl",  "ntl_nbr_max")
# setnames(cell_data, "nbr_min_ntl",  "ntl_nbr_min")
# setnames(cell_data, "nbr_mean_ntl", "ntl_nbr_mean")
# ... etc. for all 5 variables × 3 stats = 15 columns
#
# Then predict as before:
# cell_data$predicted_gdp <- predict(rf_model, newdata = cell_data)
```

---

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M-iteration `lapply` with `paste` + named vector lookup | Single vectorized `data.table` join |
| **Aggregation** | 6.46M-iteration `lapply` × 5 variables = 32.3M R function calls | One grouped `data.table` aggregation over all variables simultaneously |
| **Data structure** | 6.46M-element R list of integer vectors (~10+ GB) | Flat `data.table` (~38.4M rows, ~2-3 GB) |
| **Estimated runtime** | 86+ hours | **5–15 minutes** |
| **RAM peak** | Exceeds 16 GB | ~6–10 GB (fits in 16 GB) |
| **Numerical output** | max, min, mean per neighbor set | Identical max, min, mean per neighbor set |
| **Trained RF model** | — | Untouched; no retraining |

The key principle: **replace row-level R iteration with vectorized joins and grouped aggregation in `data.table`**, and exploit the fact that the spatial topology is cell-level (344K cells), not cell-year-level (6.46M rows).