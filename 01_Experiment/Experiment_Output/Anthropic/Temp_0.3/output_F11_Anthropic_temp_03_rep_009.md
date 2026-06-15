 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** stage — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` — as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, and for each row it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is hash-based but still carries significant per-call overhead when done 6.46 million times. With ~1.37 million directed neighbor relationships spread across 344,208 cells × 28 years, this function alone generates tens of millions of string operations.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46 million rows with `lapply`, subsetting vectors, removing NAs, and computing `max/min/mean`. That's ~32.3 million R-level function invocations just for the stats, each with allocation overhead from anonymous function closures and `c()` calls.

3. The final `do.call(rbind, result)` on a list of 6.46 million 3-element vectors is itself a well-known R performance anti-pattern.

4. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-trained model. Even with 6.46 million rows and 110 predictors, `predict.randomForest` (or `predict.ranger`) is implemented in C/C++ and typically completes in seconds to a few minutes. Loading a serialized model from disk (`readRDS`) is also fast. Writing predictions is a single vectorized write. None of these operations involve millions of R-level loop iterations.

**Conclusion:** The bottleneck is the O(n) R-level loop over 6.46 million rows done 6 times (1 build + 5 stats), totaling ~38.8 million interpreted R iterations with heavy string and list operations. This is what produces the 86+ hour runtime.

---

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup()`**: Replace the row-by-row `lapply` with a fully vectorized approach using `data.table` joins. Pre-expand the neighbor relationships into an edge table (cell→neighbor, by year) and join to get row indices, eliminating all per-row string operations.

2. **Vectorize `compute_neighbor_stats()`**: Use `data.table` grouped aggregation (`max`, `min`, `mean`) over the edge table instead of `lapply` over millions of rows.

3. **Preserve the trained Random Forest model**: No changes to the model or the predict step.

4. **Preserve the original numerical estimand**: The computed neighbor max, min, and mean values are numerically identical.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 0.  Assume these objects already exist in the environment:
#       cell_data              — data.frame with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, …
#       id_order               — integer/character vector of cell IDs (index = position in nb object)
#       rook_neighbors_unique  — spdep::nb list (length = length(id_order))
#       rf_model               — the pre-trained Random Forest model
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# 1.  Build a vectorized edge table of directed neighbor pairs
#     (focal_id -> neighbor_id) from the nb object — done ONCE.
# ---------------------------------------------------------------

build_edge_table <- function(id_order, nb_obj) {
  # For each cell position, get its neighbor positions
  from_pos <- rep(seq_along(nb_obj), lengths(nb_obj))
  to_pos   <- unlist(nb_obj)
  
  # Remove 0-entries that spdep uses for cells with no neighbors
  valid    <- to_pos != 0L
  from_pos <- from_pos[valid]
  to_pos   <- to_pos[valid]
  
  data.table(
    focal_id    = id_order[from_pos],
    neighbor_id = id_order[to_pos]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
cat("Edge table rows:", nrow(edge_dt), "\n")

# ---------------------------------------------------------------
# 2.  Convert cell_data to data.table and add a row-index column
# ---------------------------------------------------------------

dt <- as.data.table(cell_data)
dt[, row_idx := .I]

# ---------------------------------------------------------------
# 3.  Build the full focal–neighbor–year edge table by joining
#     on year.  This replaces build_neighbor_lookup entirely.
#
#     For every (focal_id, year) row, we find all neighbor rows
#     that share the same (neighbor_id, year).
# ---------------------------------------------------------------

# Keyed subsets for fast joins
focal_key   <- dt[, .(focal_row = row_idx, focal_id = id, year)]
neighbor_key <- dt[, .(neighbor_row = row_idx, neighbor_id = id, year)]

setkey(edge_dt, focal_id, neighbor_id)

# Expand edges across years:
#   focal_key  ⋈  edge_dt  on focal_id  →  (focal_row, year, neighbor_id)
#   then       ⋈  neighbor_key on (neighbor_id, year) → (focal_row, neighbor_row)

# Step A: attach year to each edge via focal cell
setkey(focal_key, focal_id)
setkey(edge_dt, focal_id)

edges_with_year <- edge_dt[focal_key,
  .(focal_row, neighbor_id, year),
  on = "focal_id",
  allow.cartesian = TRUE,
  nomatch = NULL
]

cat("Edges × years rows:", nrow(edges_with_year), "\n")

# Step B: resolve neighbor_id + year → neighbor_row
setkey(edges_with_year, neighbor_id, year)
setkey(neighbor_key, neighbor_id, year)

edges_resolved <- neighbor_key[edges_with_year,
  .(focal_row, neighbor_row),
  on = c("neighbor_id", "year"),
  nomatch = NULL
]

cat("Resolved edge-year rows:", nrow(edges_resolved), "\n")

# Clean up intermediates
rm(focal_key, neighbor_key, edges_with_year)
gc()

# ---------------------------------------------------------------
# 4.  Compute neighbor stats for all 5 variables in one pass
#     using data.table grouped aggregation.  This replaces
#     compute_neighbor_stats + the outer for-loop.
# ---------------------------------------------------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Attach neighbor values to the edge table (only the columns we need)
edges_resolved[, (neighbor_source_vars) :=
  dt[neighbor_row, ..neighbor_source_vars]
]

# Group by focal_row and compute max / min / mean for each variable
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(max(.(as.name(v)),   na.rm = TRUE)),
    bquote(min(.(as.name(v)),   na.rm = TRUE)),
    bquote(mean(.(as.name(v)),  na.rm = TRUE))
  )
}))

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", v, c("_max", "_min", "_mean"))
}))

# Build the aggregation call
stats_dt <- edges_resolved[,
  setNames(lapply(agg_exprs, eval, envir = .SD), agg_names),
  by = focal_row
]

# Handle Inf/-Inf from max/min on all-NA groups → convert to NA
for (col in agg_names) {
  set(stats_dt, which(is.infinite(stats_dt[[col]])), col, NA_real_)
}

# ---------------------------------------------------------------
# 5.  Join the aggregated neighbor features back to the main table
# ---------------------------------------------------------------

# Initialise new columns to NA (covers rows with no neighbors)
for (col in agg_names) {
  set(dt, j = col, value = NA_real_)
}

# Fast update join
dt[stats_dt, (agg_names) := mget(paste0("i.", agg_names)), on = "focal_row"]

# Drop helper column
dt[, row_idx := NULL]

# ---------------------------------------------------------------
# 6.  Random Forest prediction (UNCHANGED — not the bottleneck)
# ---------------------------------------------------------------

# Convert back if predict method expects a data.frame
cell_data <- as.data.frame(dt)

predictions <- predict(rf_model, newdata = cell_data)

cell_data$predicted <- predictions

# Write output
# e.g. fwrite(dt, "predictions_output.csv")

cat("Done. Predictions added.\n")
```

---

## Why This Is Fast

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup | 6.46M `lapply` iterations with `paste`/string matching | Single vectorized `data.table` join (~seconds) |
| Neighbor stats | 5 × 6.46M `lapply` iterations with per-row `max/min/mean` | One grouped `data.table` aggregation over the edge table |
| String operations | ~6.46M `paste()` + named-vector lookups per call | Zero — integer joins only |
| Memory pattern | 6.46M-element list of small vectors → `do.call(rbind, ...)` | Columnar `data.table` — cache-friendly, no list overhead |
| Expected runtime | 86+ hours | **Minutes** (edge expansion + grouped aggregation on ~38M rows is routine for `data.table`) |

The Random Forest `predict()` call remains untouched and was never the bottleneck. The numerical estimand (neighbor max, min, mean per variable per cell-year) is preserved exactly.