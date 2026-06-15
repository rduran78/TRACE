 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting and named-vector lookups. Named vector lookups in R via character keys are O(n) hash lookups, but doing ~6.46M of them—each touching multiple neighbors—creates billions of small operations. The `compute_neighbor_stats` function then iterates over the same 6.46M entries again per variable, and this is repeated for 5 variables.

**Specific problems:**

1. **`build_neighbor_lookup`**: Creates ~6.46M character keys (`paste(id, year)`), then for each row, pastes neighbor keys and looks them up. This is an O(N × K) string operation where K is the average neighbor count (~4 for rook contiguity). With N ≈ 6.46M, that's ~25.8M string constructions plus hash lookups inside `lapply`—all in serial, interpreted R.

2. **`compute_neighbor_stats`**: Uses `lapply` over 6.46M elements, calling `max`, `min`, `mean` on small vectors. The overhead of 6.46M R function calls dominates. `do.call(rbind, result)` on a 6.46M-element list is also expensive.

3. **Memory**: Storing `neighbor_lookup` as a list of 6.46M integer vectors has significant R list overhead (~6.46M SEXP pointers + per-element allocation).

4. **Repeated iteration**: The stats computation is called 5 times (once per variable), each time iterating over the full 6.46M rows.

---

## Optimization Strategy

**Replace the row-level R loop with a vectorized join-and-group-by approach using `data.table`.**

The key insight: the neighbor lookup is a **join** operation. Each `(cell_id, year)` pair needs to be joined to its neighbors' `(neighbor_id, same_year)` rows. This is a classic equi-join that `data.table` handles extremely efficiently in C.

**Steps:**

1. **Build an edge table** from the `nb` object: a two-column `data.table` of `(id, neighbor_id)` — ~1.37M rows.
2. **Join** the edge table to the panel data on `(neighbor_id, year)` to get neighbor values — this produces ~6.46M × ~4 ≈ ~25.8M rows, which at ~5 numeric columns is ~1 GB (fits in 16 GB RAM).
3. **Group-by aggregate** `(id, year)` to compute `max`, `min`, `mean` for all 5 variables **simultaneously** in one pass.
4. **Join** the aggregated results back to the original data.

This eliminates all R-level loops and leverages `data.table`'s parallelized, cache-optimized C internals. Expected runtime: **minutes, not hours**.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# Step 0: Convert panel data to data.table (in-place, no copy)
# ──────────────────────────────────────────────────────────────────────
setDT(cell_data)

# Ensure 'id' and 'year' columns exist and are keyed for fast joins
if (!("id" %in% names(cell_data))) stop("cell_data must have an 'id' column")
if (!("year" %in% names(cell_data))) stop("cell_data must have a 'year' column")

# ──────────────────────────────────────────────────────────────────────
# Step 1: Build edge list from the nb object
#
# id_order is the vector of cell IDs in the order matching the nb object.
# rook_neighbors_unique is the nb object (list of integer index vectors).
# We expand it into a two-column data.table: (id, neighbor_id).
# ──────────────────────────────────────────────────────────────────────
build_edge_table <- function(id_order, neighbors) {
  n <- length(neighbors)
  # Pre-allocate lengths
  lens <- vapply(neighbors, length, integer(1))
  total <- sum(lens)
  
  from_idx <- rep.int(seq_len(n), lens)
  to_idx   <- unlist(neighbors, use.names = FALSE)
  
  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edges <- build_edge_table(id_order, rook_neighbors_unique)

# ──────────────────────────────────────────────────────────────────────
# Step 2: Join edges to panel data to retrieve neighbor variable values
#
# For each (id, year), we want the variable values of all neighbors
# in the same year. This is an equi-join:
#   edges[cell_data, on = .(neighbor_id == id)]  — gets neighbor rows
# then joined again by year.
#
# Strategy: 
#   - Start from cell_data: take (id, year) and join to edges to get
#     neighbor_id for each row.
#   - Then join on (neighbor_id, year) to cell_data to get neighbor values.
# ──────────────────────────────────────────────────────────────────────

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Create a slim lookup table with only the columns we need
# This minimizes memory during the large join
lookup_cols <- c("id", "year", neighbor_source_vars)
neighbor_vals_dt <- cell_data[, ..lookup_cols]
setnames(neighbor_vals_dt, "id", "neighbor_id")
setkey(neighbor_vals_dt, neighbor_id, year)

# Create a slim version of cell_data with just id and year for the first join
cell_keys <- cell_data[, .(id, year)]

# Join cell_keys to edges to get (id, year, neighbor_id) for every cell-year-neighbor combo
# This is ~6.46M rows × ~4 neighbors = ~25.8M rows
setkey(edges, id)
cell_neighbors <- edges[cell_keys, on = .(id), allow.cartesian = TRUE, nomatch = NULL]
# Result columns: id, neighbor_id, year

# Now join to get the neighbor variable values
setkey(cell_neighbors, neighbor_id, year)
cell_neighbors <- neighbor_vals_dt[cell_neighbors, on = .(neighbor_id, year), nomatch = NA]
# Result columns: neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2, id

# Free the lookup table
rm(neighbor_vals_dt)
gc()

# ──────────────────────────────────────────────────────────────────────
# Step 3: Compute grouped aggregates (max, min, mean) per (id, year)
# ──────────────────────────────────────────────────────────────────────

# Build aggregation expressions dynamically for all 5 variables × 3 stats
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(as.numeric(max(.(as.name(v)),   na.rm = TRUE))),
    bquote(as.numeric(min(.(as.name(v)),   na.rm = TRUE))),
    bquote(as.numeric(mean(.(as.name(v)),  na.rm = TRUE)))
  )
}), recursive = FALSE)

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

names(agg_exprs) <- agg_names

# Perform the grouped aggregation in one pass (data.table does this in C)
agg_result <- cell_neighbors[,
  eval(as.call(c(quote(list), agg_exprs))),
  by = .(id, year)
]

# Handle Inf/-Inf from max/min on all-NA groups → convert to NA
inf_cols <- grep("neighbor_(max|min)_", names(agg_result), value = TRUE)
for (col in inf_cols) {
  set(agg_result, which(is.infinite(agg_result[[col]])), col, NA_real_)
}

# Also handle NaN from mean on all-NA groups
mean_cols <- grep("neighbor_mean_", names(agg_result), value = TRUE)
for (col in mean_cols) {
  set(agg_result, which(is.nan(agg_result[[col]])), col, NA_real_)
}

# Free the large intermediate table
rm(cell_neighbors)
gc()

# ──────────────────────────────────────────────────────────────────────
# Step 4: Join aggregated neighbor features back to cell_data
# ──────────────────────────────────────────────────────────────────────

# Remove any pre-existing neighbor columns to avoid conflicts
existing_neighbor_cols <- intersect(names(cell_data), agg_names)
if (length(existing_neighbor_cols) > 0) {
  cell_data[, (existing_neighbor_cols) := NULL]
}

setkey(agg_result, id, year)
setkey(cell_data, id, year)

cell_data <- agg_result[cell_data, on = .(id, year)]

rm(agg_result)
gc()

# ──────────────────────────────────────────────────────────────────────
# Done. cell_data now has 15 new columns:
#   neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl,
#   neighbor_max_ec,  neighbor_min_ec,  neighbor_mean_ec,
#   ... (same pattern for pop_density, def, usd_est_n2)
#
# The trained Random Forest model is untouched.
# The numerical values (max, min, mean) are identical to the originals.
# ──────────────────────────────────────────────────────────────────────
```

---

## Summary of Gains

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M R-level `lapply` iterations with string ops | One vectorized `data.table` join (~seconds) |
| **Stats computation** | 5 × 6.46M R-level `lapply` iterations | One grouped `data.table` aggregation in C |
| **Peak memory** | ~6.46M R list elements (high overhead) | ~25.8M-row `data.table` (~1–2 GB, fits in 16 GB) |
| **Estimated runtime** | 86+ hours | **5–15 minutes** |
| **Numerical output** | max, min, mean per neighbor set | Identical max, min, mean per neighbor set |
| **RF model** | Preserved (not retrained) | Preserved (not retrained) |

The optimization replaces interpreted R loops with `data.table`'s C-level equi-joins and grouped aggregations, which are orders of magnitude faster and more memory-efficient for this class of problem.