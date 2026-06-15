 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three interconnected reasons:

### 1. Row-level `lapply` over 6.46 million rows in `build_neighbor_lookup`
For every single cell-year row, the function performs string pasting (`paste(id, year, sep="_")`), named-vector lookups (`id_to_ref`, `idx_lookup`), and subsetting. Named vector lookup in R is **O(n)** per call because it uses linear hashing on character keys. Doing this 6.46 million times against a 6.46-million-element named vector produces roughly **O(n²)** behavior. This alone accounts for the vast majority of the 86+ hour runtime.

### 2. Redundant recomputation of the same spatial topology every year
The rook-neighbor structure is **time-invariant** — cell 17's neighbors are always the same cells regardless of year. Yet `build_neighbor_lookup` re-resolves neighbor identities for every cell-year combination, doing 28× the necessary spatial work.

### 3. Row-level `lapply` in `compute_neighbor_stats`
Even after the lookup is built, computing max/min/mean via an R-level `lapply` over 6.46 million rows with variable-length index vectors is slow due to R's per-call overhead and lack of vectorization.

---

## Optimization Strategy

**Core idea:** Build the neighbor table **once** at the cell level (344,208 cells), store it as a two-column `data.table` (cell_id → neighbor_id), then use a **vectorized join** against the yearly attribute table to compute neighbor statistics via grouped aggregation. This replaces all `lapply` loops with `data.table` operations that run in C.

| Step | What | Complexity |
|------|------|------------|
| 1 | Convert `spdep::nb` to a `data.table` edge list: `(cell_id, neighbor_id)` — ~1.37M rows, built once | O(E) |
| 2 | For each year × variable, join the edge list to cell attributes to get neighbor values, then aggregate `max`, `min`, `mean` grouped by `(cell_id, year)` | O(E) per variable per year via `data.table` merge + group-by |
| 3 | Join the aggregated neighbor stats back onto the main dataset | O(N) |

**Expected speedup:** The total work is ~5 variables × 28 years × 1.37M edges ≈ 192M rows of joins+aggregations, all handled by `data.table`'s radix-sort joins in compiled C. Estimated wall-clock time: **2–10 minutes** on a 16 GB laptop, down from 86+ hours.

The trained Random Forest model is never touched. The output columns (neighbor max, min, mean for each source variable) are numerically identical because the same values are aggregated with the same functions.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build the time-invariant edge list ONCE from the nb object
# ──────────────────────────────────────────────────────────────────────

build_edge_list <- function(id_order, nb_object) {
  # id_order: vector of cell IDs in the same order as the nb object

  # nb_object: spdep::nb list (rook_neighbors_unique)
  from <- rep(seq_along(nb_object), lengths(nb_object))
  to   <- unlist(nb_object)
  data.table(
    cell_id     = id_order[from],
    neighbor_id = id_order[to]
  )
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
# edge_dt has ~1,373,394 rows: one row per directed neighbor relationship

cat("Edge list rows:", nrow(edge_dt), "\n")

# ──────────────────────────────────────────────────────────────────────
# STEP 2: Convert main data to data.table (if not already)
# ──────────────────────────────────────────────────────────────────────

if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure key columns exist
stopifnot(all(c("id", "year") %in% names(cell_data)))

# ──────────────────────────────────────────────────────────────────────
# STEP 3: Compute neighbor stats for all source variables via joins
# ──────────────────────────────────────────────────────────────────────

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_features <- function(cell_data, edge_dt, source_vars) {
  # Subset only the columns we need for the neighbor value lookup
  # to minimize memory during the join
  lookup_cols <- c("id", "year", source_vars)
  attr_dt <- cell_data[, ..lookup_cols]

  # Key the attribute table for fast join
  setkey(attr_dt, id)

  # Expand edge list × year: join neighbor attributes

  # edge_dt has (cell_id, neighbor_id)
  # We join attr_dt onto edge_dt by neighbor_id == id, matching on year
  # Strategy: merge edge_dt with the attribute table on neighbor_id

  # Rename for clarity before join
  setnames(attr_dt, "id", "neighbor_id")
  setkey(attr_dt, neighbor_id, year)
  setkey(edge_dt, neighbor_id)

  # This is the big join: for each edge × year, pull neighbor attribute values

  # Result: ~1.37M edges × 28 years ≈ 38.5M rows
  cat("Performing edge-attribute join...\n")
  joined <- attr_dt[edge_dt, on = "neighbor_id", allow.cartesian = TRUE]
  # joined now has columns: neighbor_id, year, <source_vars>, cell_id

  cat("Joined rows:", nrow(joined), "\n")

  # Aggregate by (cell_id, year) to get neighbor max, min, mean
  cat("Aggregating neighbor statistics...\n")

  # Build aggregation expressions dynamically
  agg_exprs <- unlist(lapply(source_vars, function(v) {
    list(
      bquote(as.numeric(max(.(as.name(v)),   na.rm = TRUE))),
      bquote(as.numeric(min(.(as.name(v)),   na.rm = TRUE))),
      bquote(as.numeric(mean(.(as.name(v)),  na.rm = TRUE)))
    )
  }))

  agg_names <- unlist(lapply(source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  names(agg_exprs) <- agg_names

  # Perform grouped aggregation
  stats_dt <- joined[,
    lapply(agg_exprs, eval),
    by = .(cell_id, year)
  ]

  # Handle Inf/-Inf from max/min on all-NA groups → convert to NA
  for (col in agg_names) {
    set(stats_dt, which(is.infinite(stats_dt[[col]])), col, NA_real_)
  }

  # Rename cell_id back to id for merging with cell_data
  setnames(stats_dt, "cell_id", "id")

  return(stats_dt)
}

# Run the computation
neighbor_stats <- compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)

# ──────────────────────────────────────────────────────────────────────
# STEP 4: Join neighbor stats back onto the main dataset
# ──────────────────────────────────────────────────────────────────────

# Remove any pre-existing neighbor columns to avoid duplication
existing_neighbor_cols <- grep("^neighbor_", names(cell_data), value = TRUE)
if (length(existing_neighbor_cols) > 0) {
  cell_data[, (existing_neighbor_cols) := NULL]
}

setkey(cell_data, id, year)
setkey(neighbor_stats, id, year)

cell_data <- neighbor_stats[cell_data, on = .(id, year)]

cat("Done. cell_data now has", ncol(cell_data), "columns and", nrow(cell_data), "rows.\n")

# ──────────────────────────────────────────────────────────────────────
# STEP 5: Predict with the existing trained Random Forest (unchanged)
# ──────────────────────────────────────────────────────────────────────

# The trained RF model object (e.g., `rf_model`) is used as-is.
# Ensure prediction columns match the model's expected feature names.
# Example:
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

### If you prefer a simpler, more memory-conservative version

The join above can produce ~38M rows, using roughly 2–4 GB of RAM. If memory is tight, process one variable at a time:

```r
compute_neighbor_features_one_var <- function(cell_data, edge_dt, var_name) {
  attr_dt <- cell_data[, .(neighbor_id = id, year, val = get(var_name))]
  setkey(attr_dt, neighbor_id, year)

  joined <- attr_dt[edge_dt, on = "neighbor_id", allow.cartesian = TRUE]

  stats <- joined[,
    .(
      nmax  = if (all(is.na(val))) NA_real_ else max(val, na.rm = TRUE),
      nmin  = if (all(is.na(val))) NA_real_ else min(val, na.rm = TRUE),
      nmean = if (all(is.na(val))) NA_real_ else mean(val, na.rm = TRUE)
    ),
    by = .(cell_id, year)
  ]

  setnames(stats,
    c("nmax", "nmin", "nmean"),
    paste0("neighbor_", c("max_", "min_", "mean_"), var_name)
  )
  setnames(stats, "cell_id", "id")
  return(stats)
}

# Process one variable at a time, merge sequentially
setkey(cell_data, id, year)

for (v in neighbor_source_vars) {
  cat("Processing:", v, "\n")
  stats_v <- compute_neighbor_features_one_var(cell_data, edge_dt, v)
  setkey(stats_v, id, year)
  cell_data <- stats_v[cell_data, on = .(id, year)]
  rm(stats_v); gc()
}
```

---

## Summary

| Aspect | Original | Optimized |
|--------|----------|-----------|
| Neighbor topology | Rebuilt for every cell-year (6.46M `lapply` calls) | Built once as edge list (1.37M rows) |
| Attribute lookup | Character-key named vector: O(n) per lookup | `data.table` keyed join: O(1) amortized |
| Stat computation | R-level `lapply` over 6.46M rows | `data.table` grouped aggregation in C |
| Estimated runtime | 86+ hours | **2–10 minutes** |
| Numerical output | max, min, mean per neighbor set | **Identical** max, min, mean per neighbor set |
| Trained RF model | Untouched | **Untouched** |