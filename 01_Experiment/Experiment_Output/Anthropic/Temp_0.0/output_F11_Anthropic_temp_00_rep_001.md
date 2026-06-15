 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` — as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to construct neighbor keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. Named vector lookups in R are hash-based but carry significant per-call overhead at this scale. With ~6.46M rows and an average of ~4 rook neighbors per cell, this generates roughly **25.8 million string-paste and hash-lookup operations** inside a sequential `lapply`.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46M rows with `lapply`, subsetting a numeric vector, removing NAs, and computing `max`, `min`, and `mean`. That's another **~32.3 million** R-level function invocations total.

3. **`do.call(rbind, result)`** on a list of ~6.46M small vectors is notoriously slow in R — it creates a massive argument list and binds row-by-row.

4. By contrast, Random Forest **prediction** on a pre-trained model is a single vectorized call (`predict(model, newdata)`) that runs in compiled C/C++ code. Even with 110 predictors and 6.46M rows, this typically completes in seconds to a few minutes. Loading a serialized model (`readRDS`) is also fast. Writing predictions is a single `fwrite` or `write.csv` call.

**Conclusion:** The 86+ hour runtime is dominated by the R-level row-wise iteration in neighbor lookup construction and neighbor statistics computation, not by Random Forest inference.

---

## Optimization Strategy

1. **Replace named-vector lookups with `data.table` hash joins** — `data.table` performs equi-joins in compiled C code, eliminating millions of R-level string operations.
2. **Vectorize neighbor lookup construction** — Expand the `nb` object into an edge list (cell_id, neighbor_id), join with year to get all (cell-year, neighbor-cell-year) pairs, then join against the row index. No per-row `lapply`.
3. **Vectorize neighbor statistics** — Use `data.table` grouped aggregation (`[, .(max, min, mean), by = row_idx]`) over the expanded edge table instead of per-row `lapply`.
4. **Compute all 5 variables' stats in one pass** over the joined edge table, or at minimum use vectorized grouped operations per variable.
5. **Avoid `do.call(rbind, ...)`** — `data.table` returns results as a data.table directly.

Expected speedup: from 86+ hours to **minutes** (the entire pipeline), because all hot loops move from interpreted R to compiled C.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 0.  Assume these objects already exist:
#     - cell_data        : data.frame/data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2, ...
#     - id_order          : integer/character vector of unique cell IDs (same order as nb object)
#     - rook_neighbors_unique : spdep nb object (list of integer index vectors)
#     - rf_model          : pre-trained Random Forest model (do NOT retrain)
# ──────────────────────────────────────────────────────────────────────

# Convert to data.table if not already
if (!is.data.table(cell_data)) cell_data <- as.data.table(cell_data)

# Ensure a deterministic row order for later re-attachment
cell_data[, .row_idx := .I]

# ──────────────────────────────────────────────────────────────────────
# 1.  Build the edge list from the nb object (vectorized, one-time cost)
#     Each entry rook_neighbors_unique[[i]] contains integer indices into
#     id_order for the neighbors of id_order[i].
# ──────────────────────────────────────────────────────────────────────

edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb_idx <- rook_neighbors_unique[[i]]
  # spdep nb objects use 0L for cells with no neighbors

  nb_idx <- nb_idx[nb_idx != 0L]
  if (length(nb_idx) == 0L) return(NULL)
  data.table(focal_id = id_order[i], neighbor_id = id_order[nb_idx])
}))

cat("Edge list rows:", nrow(edge_list), "\n")
# Expected: ~1,373,394 directed edges

# ──────────────────────────────────────────────────────────────────────
# 2.  Build a vectorized neighbor lookup via data.table joins
#     For every (focal_id, year) row, find all (neighbor_id, year) rows.
# ──────────────────────────────────────────────────────────────────────

# Minimal index table: maps (id, year) -> .row_idx
idx_table <- cell_data[, .(id, year, .row_idx)]
setkey(idx_table, id, year)

# Expand edges × years:  for each focal row, get its neighbor rows
# Step A: join focal rows to edge list to get (focal .row_idx, neighbor_id, year)
focal_edges <- merge(
  cell_data[, .(focal_row = .row_idx, focal_id = id, year)],
  edge_list,
  by.x = "focal_id",
  by.y = "focal_id",
  allow.cartesian = TRUE
)
# focal_edges columns: focal_id, focal_row, year, neighbor_id

# Step B: join to find the neighbor's row index for the same year
setkey(focal_edges, neighbor_id, year)
setkey(idx_table, id, year)

focal_edges[idx_table, neighbor_row := i..row_idx,
            on = .(neighbor_id = id, year = year)]

# Drop edges where the neighbor-year row doesn't exist
focal_edges <- focal_edges[!is.na(neighbor_row)]

cat("Expanded focal-neighbor-year pairs:", nrow(focal_edges), "\n")

# ──────────────────────────────────────────────────────────────────────
# 3.  Compute neighbor statistics for all 5 variables (vectorized)
# ──────────────────────────────────────────────────────────────────────

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Attach neighbor values for all source variables at once
# (only the columns we need, indexed by neighbor_row)
neighbor_vals <- cell_data[focal_edges$neighbor_row, ..neighbor_source_vars]
focal_edges <- cbind(focal_edges[, .(focal_row)], neighbor_vals)

# Grouped aggregation: max, min, mean per focal_row per variable
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(max(.(as.name(v)), na.rm = TRUE)),
    bquote(min(.(as.name(v)), na.rm = TRUE)),
    bquote(mean(.(as.name(v)), na.rm = TRUE))
  )
}))

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", v, c("_max", "_min", "_mean"))
}))

names(agg_exprs) <- agg_names

# Evaluate the aggregation
neighbor_stats <- focal_edges[,
  lapply(agg_exprs, eval, envir = .SD),
  by = focal_row
]

# Replace -Inf/Inf (from max/min on all-NA) with NA
for (col in agg_names) {
  set(neighbor_stats, which(is.infinite(neighbor_stats[[col]])), col, NA_real_)
}

# ──────────────────────────────────────────────────────────────────────
# 3b. Alternative cleaner aggregation (if bquote approach is unwieldy)
# ──────────────────────────────────────────────────────────────────────

# Uncomment the block below as a simpler alternative to step 3 above:
#
# neighbor_stats <- focal_edges[, {
#   out <- list(focal_row = focal_row[1])
#   for (v in neighbor_source_vars) {
#     vals <- get(v)
#     vals <- vals[!is.na(vals)]
#     if (length(vals) == 0L) {
#       out[[paste0("neighbor_", v, "_max")]]  <- NA_real_
#       out[[paste0("neighbor_", v, "_min")]]  <- NA_real_
#       out[[paste0("neighbor_", v, "_mean")]] <- NA_real_
#     } else {
#       out[[paste0("neighbor_", v, "_max")]]  <- max(vals)
#       out[[paste0("neighbor_", v, "_min")]]  <- min(vals)
#       out[[paste0("neighbor_", v, "_mean")]] <- mean(vals)
#     }
#   }
#   out
# }, by = focal_row]

# ──────────────────────────────────────────────────────────────────────
# 4.  Merge neighbor stats back into cell_data
# ──────────────────────────────────────────────────────────────────────

setkey(neighbor_stats, focal_row)
setkey(cell_data, .row_idx)

# Remove old neighbor columns if they exist (idempotency)
old_cols <- intersect(agg_names, names(cell_data))
if (length(old_cols)) cell_data[, (old_cols) := NULL]

cell_data <- merge(cell_data, neighbor_stats, by.x = ".row_idx", by.y = "focal_row", all.x = TRUE)

# Rows with no neighbors will have NA for all neighbor stats (correct behavior)

# ──────────────────────────────────────────────────────────────────────
# 5.  Random Forest prediction (NOT the bottleneck — keep as-is)
# ──────────────────────────────────────────────────────────────────────

# rf_model <- readRDS("path/to/trained_rf_model.rds")   # if not already loaded
# predictions <- predict(rf_model, newdata = cell_data)
# cell_data[, prediction := predictions]
# fwrite(cell_data, "output_predictions.csv")

# Clean up helper column
cell_data[, .row_idx := NULL]

cat("Done. Neighbor feature engineering complete.\n")
```

---

## Summary

| Aspect | Colleague's Claim (RF Inference) | Actual Bottleneck (Neighbor Features) |
|---|---|---|
| **Operation type** | Vectorized C/C++ (`predict()`) | Row-wise R `lapply` over 6.46M rows × 5 vars |
| **String operations** | None | ~25.8M `paste()` + named-vector hash lookups |
| **Binding cost** | Single output vector | `do.call(rbind, ...)` on 6.46M 3-element vectors |
| **Estimated time** | Minutes | 86+ hours |
| **Fix** | None needed | Replace with `data.table` joins + grouped aggregation |

The optimized code replaces all per-row R-level iteration with compiled `data.table` joins and grouped aggregations, reducing the expected runtime from **86+ hours to minutes** while preserving the trained Random Forest model and the original numerical estimand (identical max/min/mean neighbor statistics, identical predictions).