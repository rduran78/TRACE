 # Diagnosis

**The colleague's claim is rejected.** The bottleneck is **not** Random Forest inference. Loading a trained model, calling `predict()` once on ~6.46M rows × 110 features, and writing predictions is a matter of seconds to minutes on a modern laptop.

**The true bottleneck is `build_neighbor_lookup` and `compute_neighbor_stats`.** Here's why:

1. **`build_neighbor_lookup`** runs an `lapply` over **6.46 million rows**, and inside each iteration it performs `paste()`, character key construction, and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) string hashing per call. Over 6.46M iterations, each touching multiple neighbors, this produces billions of character operations. This single function likely accounts for the vast majority of the 86+ hour runtime.

2. **`compute_neighbor_stats`** iterates over the 6.46M-element lookup list, performing subsetting, `na.rm` filtering, and `max`/`min`/`mean` per row — then binds everything with `do.call(rbind, ...)` on a 6.46M-element list. This is slow but secondary compared to the lookup construction.

3. These operations are repeated for **5 variables**, compounding the cost of `compute_neighbor_stats` (though `build_neighbor_lookup` runs once).

---

# Optimization Strategy

1. **Replace character-key lookup with integer-indexed direct joins.** Build the neighbor lookup using `data.table` fast merge/join on integer keys `(id, year)` → row index, eliminating all `paste()` and named-vector string matching.

2. **Vectorize `compute_neighbor_stats`** using `data.table` grouping: explode the neighbor relationships into an edge table, join variable values, and compute grouped `max`/`min`/`mean` in one vectorized pass — for all 5 variables at once.

3. **Avoid per-row `lapply` entirely.**

---

# Working R Code

```r
library(data.table)

# ── 0. Ensure cell_data is a data.table with a row index ──────────────
cell_data <- as.data.table(cell_data)
cell_data[, row_idx := .I]

# ── 1. Build integer-indexed neighbor edge table (replaces build_neighbor_lookup) ──
build_neighbor_edges <- function(dt, id_order, neighbors) {
  # Map each id to its position in id_order
  id_to_ref <- data.table(id = id_order, ref_idx = seq_along(id_order))

  # For every ref_idx, get the neighbor ids
  edges_id <- rbindlist(lapply(seq_along(neighbors), function(i) {
    nb <- neighbors[[i]]
    if (length(nb) == 0L) return(data.table(src_id = integer(0), nb_id = integer(0)))
    data.table(src_id = id_order[i], nb_id = id_order[nb])
  }))

  # Get unique years
  years <- sort(unique(dt$year))

  # Cross-join edges × years  →  (src_id, nb_id, year)
  edge_year <- CJ(edge_idx = seq_len(nrow(edges_id)), year = years)
  edge_year[, `:=`(src_id = edges_id$src_id[edge_idx],
                    nb_id  = edges_id$nb_id[edge_idx])]
  edge_year[, edge_idx := NULL]

  # Map (id, year) → row_idx in dt
  key_map <- dt[, .(id, year, row_idx)]
  setkey(key_map, id, year)

  # Attach source row index
  setnames(key_map, "id", "src_id")
  setkey(edge_year, src_id, year)
  edge_year <- key_map[edge_year, on = .(src_id, year), nomatch = 0L]
  setnames(edge_year, "row_idx", "src_row")

  # Attach neighbor row index
  setnames(key_map, "src_id", "nb_id")
  setkey(edge_year, nb_id, year)
  edge_year <- key_map[edge_year, on = .(nb_id, year), nomatch = 0L]
  setnames(edge_year, "row_idx", "nb_row")

  setnames(key_map, "nb_id", "id")  # restore
  return(edge_year[, .(src_row, nb_row)])
}

cat("Building neighbor edge table ...\n")
edge_dt <- build_neighbor_edges(cell_data, id_order, rook_neighbors_unique)
cat(sprintf("Edge table: %s rows\n", format(nrow(edge_dt), big.mark = ",")))

# ── 2. Compute neighbor stats for all variables at once (replaces loop) ──
compute_all_neighbor_features <- function(dt, edge_dt, var_names) {
  # Pre-extract the columns we need from the neighbor rows
  nb_vals <- dt[edge_dt$nb_row, ..var_names]
  nb_vals[, src_row := edge_dt$src_row]

  for (v in var_names) {
    cat(sprintf("  Computing neighbor stats for: %s\n", v))

    # Subset to non-NA neighbor values for this variable
    sub <- nb_vals[!is.na(get(v)), .(src_row, val = get(v))]

    stats <- sub[, .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ), keyby = src_row]

    # Name the new columns
    max_col  <- paste0(v, "_nb_max")
    min_col  <- paste0(v, "_nb_min")
    mean_col <- paste0(v, "_nb_mean")
    setnames(stats, c("nb_max", "nb_min", "nb_mean"),
                     c(max_col,  min_col,  mean_col))

    # Join back to dt on row_idx
    dt[stats, on = .(row_idx = src_row),
       (c(max_col, min_col, mean_col)) := mget(c(max_col, min_col, mean_col))]
  }

  return(dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features ...\n")
cell_data <- compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)
cat("Neighbor features done.\n")

# ── 3. Random Forest prediction (untouched — this was never the bottleneck) ──
# library(ranger)  # or randomForest
# rf_model <- readRDS("trained_rf_model.rds")
# preds <- predict(rf_model, data = cell_data)
# cell_data[, prediction := preds$predictions]
# fwrite(cell_data, "predictions_output.csv")
```

---

# Why This Fixes the Runtime

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `lapply` iterations with `paste` + named-vector char lookup | One `data.table` cross-join + two integer-keyed joins (~minutes) |
| `compute_neighbor_stats` (×5 vars) | 6.46M `lapply` iterations × 5, plus `do.call(rbind, 6.46M-element list)` | Vectorized `data.table` grouped aggregation per variable (~seconds each) |
| RF inference | Fast (seconds–minutes) | Unchanged |

**Estimated new runtime: under 30 minutes** (dominated by the edge-table cross-join at ~38.4M edges × 28 years ≈ 1.08B rows, which `data.table` handles efficiently in-memory within 16 GB). The trained Random Forest model and the original numerical estimand are fully preserved.