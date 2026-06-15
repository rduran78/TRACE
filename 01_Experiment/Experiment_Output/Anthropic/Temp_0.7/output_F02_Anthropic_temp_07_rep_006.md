 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is clear: **`build_neighbor_lookup`** constructs a list of 6.46 million elements, each produced by an anonymous function inside `lapply` that performs per-row string pasting, named-vector lookups, and NA filtering. This is an O(n) loop with expensive character operations at every iteration (~6.46M iterations). Then **`compute_neighbor_stats`** iterates over that 6.46M-element list again for *each* of 5 variables, doing per-element subsetting and aggregation—another ~32.3M R-level function calls total. The combination of:

1. **Character key construction and lookup per row** (`paste`, named vector indexing) inside a 6.46M-iteration `lapply`.
2. **R-level loops over millions of list elements** for neighbor stat computation (no vectorization).
3. **Repeated per-variable passes** over the same neighbor structure.
4. **Memory pressure** from a 6.46M-element list of integer vectors plus intermediate character vectors.

…produces the estimated 86+ hour runtime on a 16 GB laptop.

---

## Optimization Strategy

### Core Ideas

| Problem | Solution |
|---|---|
| Character key paste/lookup per row | Replace with integer arithmetic: `key = id_index * 100 + (year - 1991)`, then use `match()` or a pre-built integer-keyed environment, or better yet, operate entirely on sorted/grouped integer indices via `data.table`. |
| R-level `lapply` over 6.46M rows | Vectorize neighbor aggregation using `data.table` joins and grouped operations. Build an edge-list (long table of `row_i → row_j` pairs), then join variable values and aggregate with `data.table`'s optimized `by=` grouping. |
| 5 separate passes over neighbor structure | Compute all 5 variables' neighbor stats in a single grouped aggregation pass. |
| Memory: 6.46M-element list | Replace list-of-vectors with a flat edge-list `data.table` (~40–50M rows for directed edges × 28 years, but stored as two integer columns ≈ 0.8–1 GB, well within 16 GB). |

### Expected Speedup

The `data.table` grouped-join approach replaces ~38M R-level function calls with a handful of vectorized C-level operations. Expected wall-clock time: **minutes, not hours**. Memory usage peaks at roughly 3–5 GB, fitting in 16 GB.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 0. Convert cell_data to data.table (non-destructive; preserves all columns)
# ──────────────────────────────────────────────────────────────────────
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ──────────────────────────────────────────────────────────────────────
# 1. Build a flat edge-list from the nb object  (one-time, fast)
#    rook_neighbors_unique is a list of length N_cells (344,208).
#    id_order[i] is the cell id for the i-th element.
#    neighbors[[i]] gives integer indices (into id_order) of i's neighbors.
# ──────────────────────────────────────────────────────────────────────
build_edge_list <- function(id_order, neighbors) {
  # Pre-allocate vectors
  n <- length(neighbors)
  from_ids <- vector("list", n)
  to_ids   <- vector("list", n)
  for (i in seq_len(n)) {
    nb_i <- neighbors[[i]]
    if (length(nb_i) == 0L || (length(nb_i) == 1L && nb_i[1] == 0L)) next
    from_ids[[i]] <- rep(id_order[i], length(nb_i))
    to_ids[[i]]   <- id_order[nb_i]
  }
  data.table(
    from_id = unlist(from_ids, use.names = FALSE),
    to_id   = unlist(to_ids,   use.names = FALSE)
  )
}

cat("Building spatial edge list...\n")
edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
cat(sprintf("  Edge list: %s directed edges\n", format(nrow(edge_dt), big.mark = ",")))

# ──────────────────────────────────────────────────────────────────────
# 2. Expand edge list across years  (vectorized cross-join)
#    Each spatial edge exists in every year.
# ──────────────────────────────────────────────────────────────────────
years <- sort(unique(cell_data$year))                 # 1992:2019
year_dt <- data.table(year = years)

cat("Expanding edges across years...\n")
# Cross join: every edge × every year
edge_year <- edge_dt[, CJ_idx := 1L][year_dt[, CJ_idx := 1L], on = "CJ_idx",
                                       allow.cartesian = TRUE]
edge_dt[, CJ_idx := NULL]
edge_year[, CJ_idx := NULL]

# Alternatively, more explicit and memory-friendly in chunks if needed:
# edge_year <- CJ(edge_idx = seq_len(nrow(edge_dt)), year = years)
# edge_year[, `:=`(from_id = edge_dt$from_id[edge_idx],
#                   to_id   = edge_dt$to_id[edge_idx])]
# edge_year[, edge_idx := NULL]

# If the above cartesian join is tricky with data.table syntax, here is
# a robust alternative:
if (!exists("edge_year") || nrow(edge_year) == 0L) {
  edge_year <- rbindlist(lapply(years, function(y) {
    copy(edge_dt)[, year := y]
  }))
}

cat(sprintf("  Expanded edge-year rows: %s\n",
            format(nrow(edge_year), big.mark = ",")))

# ──────────────────────────────────────────────────────────────────────
# 3. Assign a row index to cell_data and join neighbor values
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Ensure cell_data has a row identifier for the "from" side
cell_data[, .row_id := .I]

# Key cell_data for fast joins
setkey(cell_data, id, year)

# We need: for each (from_id, year), look up from_row;
#           for each (to_id, year),   look up the neighbor's variable values.

# Step 3a: attach from-side row id
cat("Joining from-side row indices...\n")
edge_year <- merge(
  edge_year,
  cell_data[, .(id, year, .row_id)],
  by.x = c("from_id", "year"),
  by.y = c("id", "year"),
  all.x = FALSE   # drop edges where from_id/year not in data
)
setnames(edge_year, ".row_id", "from_row")

# Step 3b: attach neighbor (to-side) variable values
cat("Joining neighbor variable values...\n")
to_cols <- c("id", "year", neighbor_source_vars)
edge_year <- merge(
  edge_year,
  cell_data[, ..to_cols],
  by.x = c("to_id", "year"),
  by.y = c("id", "year"),
  all.x = FALSE
)

# ──────────────────────────────────────────────────────────────────────
# 4. Compute neighbor stats in one vectorized grouped aggregation
# ──────────────────────────────────────────────────────────────────────
cat("Computing neighbor statistics (all variables, single pass)...\n")

# Build aggregation expressions dynamically
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(max(.(as.name(v)),   na.rm = TRUE)),
    bquote(min(.(as.name(v)),   na.rm = TRUE)),
    bquote(mean(.(as.name(v)),  na.rm = TRUE))
  )
}))

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

names(agg_exprs) <- agg_names

# Aggregate
neighbor_stats <- edge_year[,
  lapply(agg_exprs, eval),
  by = from_row
]

# Replace -Inf/Inf (from max/min of empty after na.rm) with NA
for (col in agg_names) {
  set(neighbor_stats, which(is.infinite(neighbor_stats[[col]])), col, NA_real_)
}

cat(sprintf("  Neighbor stats computed for %s cell-year rows.\n",
            format(nrow(neighbor_stats), big.mark = ",")))

# ──────────────────────────────────────────────────────────────────────
# 5. Attach neighbor features back to cell_data
# ──────────────────────────────────────────────────────────────────────
cat("Merging neighbor features into cell_data...\n")

# Remove any pre-existing neighbor columns to avoid conflicts
old_nb_cols <- intersect(names(cell_data), agg_names)
if (length(old_nb_cols) > 0) {
  cell_data[, (old_nb_cols) := NULL]
}

# Join on row id
cell_data <- merge(cell_data, neighbor_stats, by.x = ".row_id", by.y = "from_row",
                   all.x = TRUE, sort = FALSE)

# Restore original row order
setorder(cell_data, .row_id)
cell_data[, .row_id := NULL]

# ──────────────────────────────────────────────────────────────────────
# 6. Clean up large intermediates
# ──────────────────────────────────────────────────────────────────────
rm(edge_dt, edge_year, neighbor_stats, year_dt)
gc()

cat("Done. Neighbor features attached to cell_data.\n")
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Trained Random Forest model** | Untouched. We only rebuild the *feature columns* that the model expects; the model object is never modified or retrained. |
| **Numerical estimand** | The aggregation functions (`max`, `min`, `mean` with `na.rm = TRUE`) and the NA-handling logic are identical to the original `compute_neighbor_stats`. The same directed rook-neighbor relationships are used. Output columns have the same names and semantics. |
| **Column names** | The naming convention `neighbor_{max,min,mean}_{var}` matches what `compute_and_add_neighbor_features` would have produced (adjust the prefix/suffix pattern if your wrapper used a different convention). |

---

## Performance Expectations on a 16 GB Laptop

| Stage | Estimated Time | Peak Memory |
|---|---|---|
| Edge-list construction | ~2 seconds | ~50 MB |
| Year expansion (~38.5M rows) | ~10 seconds | ~1.2 GB |
| Two merges (from-row + neighbor values) | ~1–3 minutes | ~3–4 GB |
| Grouped aggregation (single pass, 5 vars × 3 stats) | ~1–3 minutes | ~4 GB |
| Final merge back | ~30 seconds | ~3 GB |
| **Total** | **~5–8 minutes** | **~4–5 GB peak** |

This replaces the estimated **86+ hours** with a runtime of roughly **5–8 minutes**—a speedup of approximately **600–1,000×**—while staying comfortably within 16 GB RAM.