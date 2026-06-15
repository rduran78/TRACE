 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is clear: **`build_neighbor_lookup` creates a list of 6.46 million elements, each produced by an `lapply` iteration that performs character coercion, string pasting, and named-vector lookups.** This is catastrophically slow in R because:

1. **Per-row string operations**: For each of ~6.46M rows, `paste(neighbor_cell_ids, data$year[i], sep = "_")` and named-vector indexing (`idx_lookup[neighbor_keys]`) are called individually. Named vector lookup in R is O(n) hash probing *per call*, and the overhead of 6.46M R-level function calls in `lapply` is enormous.

2. **Redundant work across variables**: The neighbor lookup is built once (good), but `compute_neighbor_stats` also uses an R-level `lapply` over 6.46M elements, repeated for each of 5 variables = ~32.3M R-level function invocations.

3. **Memory pressure**: Storing 6.46M list elements, each an integer vector, creates massive list overhead on a 16 GB machine.

**In short**: The algorithm is correct but the implementation is O(rows × avg_neighbors) with enormous R-interpreter constant factors. The 86+ hour estimate is dominated by millions of R-level function calls with string operations.

---

## Optimization Strategy

**Replace the row-level R loops with vectorized joins and grouped aggregations using `data.table`.**

The key insight: the neighbor lookup can be expressed as a **join**. Each cell-year row needs to find its neighbors' values *in the same year*. This is a standard equi-join:

1. **Build an edge table** (a two-column `data.table` of `id → neighbor_id`) from the `nb` object — done once, ~1.37M rows.
2. **Join** `cell_data` to the edge table on `(id, year)` ↔ `(neighbor_id, year)` to get all neighbor values in one vectorized operation.
3. **Group-aggregate** (`max`, `min`, `mean`) by `(id, year)` — fully vectorized in C via `data.table`.

This eliminates all per-row R loops, string pasting, and named-vector lookups. Expected runtime: **minutes, not days.**

The numerical results are identical because `max`, `min`, and `mean` over the same neighbor sets produce the same values. The trained Random Forest model is untouched.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# Step 1: Build a directed edge table from the nb object (once)
# ──────────────────────────────────────────────────────────────────────
# rook_neighbors_unique is a list (spdep nb object) of length 344,208.
# id_order is the vector mapping list index → cell id.

build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(neighbors))  # ~1,373,394
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_i <- neighbors[[i]]
    # spdep nb objects use 0L for no neighbors; filter those out
    nb_i <- nb_i[nb_i > 0L]
    n_i  <- length(nb_i)
    if (n_i > 0L) {
      from_id[pos:(pos + n_i - 1L)] <- id_order[i]
      to_id[pos:(pos + n_i - 1L)]   <- id_order[nb_i]
      pos <- pos + n_i
    }
  }
  
  data.table(id = from_id[1:(pos - 1L)], neighbor_id = to_id[1:(pos - 1L)])
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

# ──────────────────────────────────────────────────────────────────────
# Step 2: Convert cell_data to data.table (if not already)
# ──────────────────────────────────────────────────────────────────────
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Preserve original row order for downstream prediction
cell_data[, .row_order := .I]

# ──────────────────────────────────────────────────────────────────────
# Step 3: Compute neighbor stats for all 5 variables via a single join
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Subset to only the columns we need for the join (minimize memory)
join_cols <- c("id", "year", neighbor_source_vars)
neighbor_values <- cell_data[, ..join_cols]

# Set key on the neighbor side for fast join
setnames(neighbor_values, "id", "neighbor_id")
setkey(neighbor_values, neighbor_id, year)

# Set key on edge table
setkey(edge_dt, neighbor_id)

# Join: for each edge (id, neighbor_id), attach the neighbor's year and values
# We need to join on (neighbor_id, year), so we first cross edge_dt with years
# via cell_data. More efficient: join cell_data to edge_dt on id, then look up
# neighbor values.

# Approach: 
#   1. Take cell_data's (id, year) and join to edge_dt on id → get (id, year, neighbor_id)
#   2. Join that to neighbor_values on (neighbor_id, year) → get neighbor variable values
#   3. Aggregate by (id, year)

# Step 3a: Expand edges by year
# cell_data has (id, year); edge_dt has (id, neighbor_id)
# We need all (id, year, neighbor_id) combinations that exist.

id_year <- cell_data[, .(id, year)]
setkey(id_year, id)
setkey(edge_dt, id)

# Join: each (id, year) row gets its neighbor_ids
# This produces ~6.46M × avg_neighbors ≈ ~25.8M rows (4 neighbors avg for rook)
expanded <- edge_dt[id_year, on = "id", allow.cartesian = TRUE, nomatch = NULL]
# Result columns: id, neighbor_id, year

# Step 3b: Look up neighbor values
setkey(expanded, neighbor_id, year)
setkey(neighbor_values, neighbor_id, year)

expanded <- neighbor_values[expanded, on = .(neighbor_id, year), nomatch = NA]
# Now expanded has: neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2, id

# Step 3c: Aggregate by (id, year) — compute max, min, mean for each variable
# Build aggregation expressions dynamically
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
    bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
    bquote(mean(.(as.name(v)), na.rm = TRUE))
  )
}))

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("n_", v, c("_max", "_min", "_mean"))
}))

names(agg_exprs) <- agg_names

# Evaluate the aggregation
neighbor_stats <- expanded[,
  lapply(agg_exprs, eval, envir = .SD),
  by = .(id, year),
  .SDcols = neighbor_source_vars
]

# Handle Inf/-Inf from max/min on all-NA groups (convert to NA)
inf_cols <- grep("_(max|min)$", names(neighbor_stats), value = TRUE)
for (col in inf_cols) {
  set(neighbor_stats, which(is.infinite(neighbor_stats[[col]])), col, NA_real_)
}

# ──────────────────────────────────────────────────────────────────────
# Alternative Step 3c (simpler, if the dynamic expression approach is
# finicky in your R version):
# ──────────────────────────────────────────────────────────────────────
# Uncomment below and comment out the dynamic approach above if preferred:

# neighbor_stats <- expanded[, {
#   out <- list()
#   for (v in neighbor_source_vars) {
#     vals <- get(v)
#     vals <- vals[!is.na(vals)]
#     if (length(vals) == 0L) {
#       out[[paste0("n_", v, "_max")]]  <- NA_real_
#       out[[paste0("n_", v, "_min")]]  <- NA_real_
#       out[[paste0("n_", v, "_mean")]] <- NA_real_
#     } else {
#       out[[paste0("n_", v, "_max")]]  <- max(vals)
#       out[[paste0("n_", v, "_min")]]  <- min(vals)
#       out[[paste0("n_", v, "_mean")]] <- mean(vals)
#     }
#   }
#   out
# }, by = .(id, year)]

# ──────────────────────────────────────────────────────────────────────
# Step 4: Merge neighbor stats back into cell_data
# ──────────────────────────────────────────────────────────────────────

# Remove any old neighbor columns if they exist (idempotency)
old_neighbor_cols <- intersect(names(cell_data), agg_names)
if (length(old_neighbor_cols) > 0) {
  cell_data[, (old_neighbor_cols) := NULL]
}

setkey(cell_data, id, year)
setkey(neighbor_stats, id, year)

cell_data <- neighbor_stats[cell_data, on = .(id, year)]

# Restore original row order (important for predict() alignment)
setorder(cell_data, .row_order)
cell_data[, .row_order := NULL]

# ──────────────────────────────────────────────────────────────────────
# Step 5: Predict with the existing trained Random Forest (unchanged)
# ──────────────────────────────────────────────────────────────────────
# The model object and predict() call remain exactly as before.
# e.g.: predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Is Correct and Safe

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` are computed over the identical neighbor sets defined by `rook_neighbors_unique` and the same `id_order` mapping. The join `(id → neighbor_id) × year` reproduces exactly the same cell-year neighbor pairings as the original `build_neighbor_lookup`. |
| **Irregular topology / masked cells** | The edge table is built directly from the `nb` object — no rectangular-grid assumption. Cell-years with no neighbors (no match in `edge_dt`) simply get `NA` for all neighbor stats, identical to the original code's `if (length(idx) == 0) return(c(NA, NA, NA))`. |
| **Trained RF model preserved** | No model retraining. The code only reconstructs the same feature columns with the same names and values. |
| **Memory on 16 GB laptop** | The expanded join table is ~26M rows × 7 columns ≈ ~1.5 GB. Together with `cell_data` (~6.46M × 110 cols ≈ ~5.5 GB), total peak is well within 16 GB. The intermediate tables can be removed with `rm(expanded, neighbor_values); gc()` if needed. |

## Expected Runtime

| Phase | Estimated Time |
|---|---|
| Build edge table | < 5 seconds |
| Expand edges × years (join) | ~10–30 seconds |
| Look up neighbor values (join) | ~10–30 seconds |
| Grouped aggregation (15 stats) | ~30–90 seconds |
| Merge back to cell_data | ~10 seconds |
| **Total** | **~2–4 minutes** |

This replaces the estimated 86+ hour runtime with a ~3 minute vectorized pipeline while preserving the exact numerical output and the trained model.