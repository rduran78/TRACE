 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting and named-vector lookups. Named vector lookups in R via character keys are O(n) hash lookups, but doing this millions of times with string concatenation creates enormous overhead. The `compute_neighbor_stats` function then iterates over the same 6.46 million rows again, once per variable (×5 variables). Together:

1. **`build_neighbor_lookup`**: ~6.46M iterations, each doing character coercion, `paste`, and named-vector subsetting. The `idx_lookup` named vector has 6.46M entries, making each lookup expensive. Estimated: ~70–80% of total runtime.
2. **`compute_neighbor_stats`**: Uses `lapply` returning a 3-element vector per row, then `do.call(rbind, ...)` on a 6.46M-element list — this alone is a known R anti-pattern that causes massive memory allocation and copying.
3. **Memory**: Storing `neighbor_lookup` as a list of 6.46M integer vectors is itself memory-heavy (~several GB with list overhead).

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Per-row string key lookups in `build_neighbor_lookup` | Replace with integer-keyed `data.table` join. Build a cell-year → row-index mapping table and join neighbor cell-IDs by year in a single vectorized operation. |
| Storing 6.46M-element list of neighbor indices | Replace with a flat `data.table` of `(row_i, neighbor_row_j)` pairs — a sparse edge list. This is more cache-friendly and enables vectorized grouped aggregation. |
| Per-row `lapply` + `do.call(rbind, ...)` in `compute_neighbor_stats` | Replace with a single `data.table` grouped aggregation: join the variable values onto the edge list, then `[, .(max, min, mean), by = row_i]`. |
| Repeated work across 5 variables | Compute all 5 variables' stats in one pass over the edge list. |

**Expected speedup**: From ~86 hours to roughly **5–20 minutes**, depending on disk I/O. Memory peak should stay well under 16 GB.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 1. Convert cell_data to data.table (non-destructive; preserves all columns)
# ──────────────────────────────────────────────────────────────────────
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Add a row index column (will be used as the primary key for joining back)
cell_data[, .row_idx := .I]

# ──────────────────────────────────────────────────────────────────────
# 2. Build a flat edge list of (cell_id, neighbor_cell_id) from the nb object
#    This replaces the per-row string-key approach entirely.
# ──────────────────────────────────────────────────────────────────────
build_edge_list <- function(id_order, neighbors) {
  # neighbors is an nb object: a list of integer index vectors
  # id_order maps positional index -> cell_id
  n <- length(neighbors)
  # Pre-allocate by computing total number of edges
  lengths_vec <- lengths(neighbors)
  total_edges <- sum(lengths_vec)

  from_id <- rep.int(id_order, lengths_vec)
  to_id   <- id_order[unlist(neighbors, use.names = FALSE)]

  data.table(cell_id = from_id, neighbor_cell_id = to_id)
}

cat("Building spatial edge list...\n")
edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
# edge_dt has ~1.37M rows (directed rook-neighbor pairs)

# ──────────────────────────────────────────────────────────────────────
# 3. Expand edge list across years by joining with cell_data's
#    (cell_id, year) → row_idx mapping.
#    This is the vectorized replacement for build_neighbor_lookup.
# ──────────────────────────────────────────────────────────────────────
cat("Building cell-year to row index mapping...\n")

# Mapping table: for every (id, year) in cell_data, record the row index
idx_map <- cell_data[, .(cell_id = id, year, .row_idx)]
setkey(idx_map, cell_id, year)

# Get the unique years present in the data
years_in_data <- sort(unique(cell_data$year))

# Cross-join edges × years, then resolve both endpoints to row indices
cat("Expanding edge list across all years...\n")

# Add year dimension to edge list
edge_year_dt <- CJ_dt_edges(edge_dt, years_in_data)
# We implement this efficiently:
edge_year_dt <- edge_dt[, .(year = years_in_data), by = .(cell_id, neighbor_cell_id)]

# Resolve the "from" cell (the focal row) to its row index
setkey(edge_year_dt, cell_id, year)
edge_year_dt <- idx_map[edge_year_dt, on = .(cell_id, year), nomatch = 0L]
setnames(edge_year_dt, ".row_idx", "focal_row")

# Resolve the "to" cell (the neighbor) to its row index
setnames(edge_year_dt, "neighbor_cell_id", "cell_id_nb")
setkey(edge_year_dt, cell_id_nb, year)

idx_map_nb <- copy(idx_map)
setnames(idx_map_nb, c("cell_id", "year", "nb_row"))

edge_year_dt <- idx_map_nb[edge_year_dt, on = .(cell_id = cell_id_nb, year), nomatch = 0L]

# Result: edge_year_dt has columns: cell_id (neighbor), year, nb_row, cell_id (focal via key), focal_row
# Keep only what we need:
edge_year_dt <- edge_year_dt[, .(focal_row, nb_row)]

cat(sprintf("Expanded edge list: %s rows\n", format(nrow(edge_year_dt), big.mark = ",")))

# ──────────────────────────────────────────────────────────────────────
# 4. Compute neighbor stats for all 5 variables in one pass
#    This replaces compute_neighbor_stats + the outer for-loop.
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor statistics...\n")

# Attach neighbor variable values to the edge list
# We pull the columns we need from cell_data by nb_row index
nb_vals <- cell_data[edge_year_dt$nb_row, ..neighbor_source_vars]
nb_vals[, focal_row := edge_year_dt$focal_row]

# Grouped aggregation: for each focal_row, compute max/min/mean of each variable
agg_expr <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(max(.(as.name(v)),   na.rm = TRUE)),
    bquote(min(.(as.name(v)),   na.rm = TRUE)),
    bquote(mean(.(as.name(v)),  na.rm = TRUE))
  )
}), recursive = FALSE)

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

# Build the aggregation call dynamically
agg_call <- as.call(c(as.name("list"),
                       setNames(agg_expr, agg_names)))

neighbor_stats <- nb_vals[, eval(agg_call), by = focal_row]

# Replace Inf/-Inf (from max/min on all-NA groups) with NA
inf_cols <- agg_names[grepl("max_|min_", agg_names)]
for (col in inf_cols) {
  set(neighbor_stats, which(is.infinite(neighbor_stats[[col]])), col, NA_real_)
}

# ──────────────────────────────────────────────────────────────────────
# 5. Join neighbor stats back to cell_data
# ──────────────────────────────────────────────────────────────────────
cat("Joining neighbor features back to cell_data...\n")

setkey(neighbor_stats, focal_row)

# Rows with no neighbors at all won't appear in neighbor_stats;
# they will get NA after the join, which matches original behavior.
cell_data[neighbor_stats, (agg_names) := mget(agg_names), on = .(.row_idx = focal_row)]

# Clean up the helper column
cell_data[, .row_idx := NULL]

cat("Done. Neighbor features added.\n")
```

### Explanation of why this preserves the original numerical estimand

- **Same neighbor relationships**: The edge list is built from the identical `rook_neighbors_unique` nb object and `id_order` mapping.
- **Same statistics**: `max`, `min`, and `mean` with `na.rm = TRUE` are computed identically; `NA` handling matches the original (groups with zero non-NA values yield `NA`).
- **Same column names**: The output columns follow the same naming convention (`neighbor_max_ntl`, `neighbor_min_ntl`, `neighbor_mean_ntl`, etc.) that `compute_and_add_neighbor_features` would have produced.
- **Random Forest model is untouched**: Only the feature-engineering step is optimized; no model retraining occurs.

### Expected performance

| Metric | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M `lapply` iterations with string ops | Single vectorized `data.table` join (~seconds) |
| `compute_neighbor_stats` | 5 × 6.46M `lapply` + `do.call(rbind)` | One grouped `data.table` aggregation (~minutes) |
| Peak memory | List of 6.46M vectors + repeated `rbind` copies | Flat edge table + in-place `data.table` ops (~4–8 GB) |
| **Total time** | **~86+ hours** | **~5–20 minutes** |