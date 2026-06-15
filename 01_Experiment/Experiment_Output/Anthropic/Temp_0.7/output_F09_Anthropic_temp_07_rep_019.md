 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two fundamental reasons:

### 1. `build_neighbor_lookup` is O(N²)-ish in practice
The function iterates over **all ~6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs **character key strings** (`paste(id, year)`) and performs **named-vector lookups** (`idx_lookup[neighbor_keys]`) — these are hash-table lookups on a 6.46-million-entry named vector, repeated ~6.46 million times with ~4 neighbors each, yielding **~25.8 million character-key hash lookups**. Character operations and named-vector indexing in R are slow.

The critical insight is that **the neighbor graph is purely spatial and identical every year**. The current code rebuilds the mapping from scratch across all cell-year rows, even though the spatial adjacency never changes. This is entirely redundant work.

### 2. `compute_neighbor_stats` uses per-row `lapply`
For each of the 5 variables, the function loops over 6.46 million rows in R-level `lapply`, calling `max`, `min`, and `mean` on small vectors. That's **~32.3 million R function calls** (5 vars × 6.46M rows), each with overhead.

### Memory is not the bottleneck; R-level iteration is.

---

## Optimization Strategy

**Core idea:** Build a **year-free spatial neighbor edge table once** (a two-column data.table of `cell_id → neighbor_id`, ~1.37M rows), then for each year, **join** the yearly cell attributes onto this table and compute grouped `max`, `min`, `mean` using `data.table` vectorized aggregation. This eliminates all per-row R-level loops and all character-key hashing.

### Steps:

1. **Convert `rook_neighbors_unique` (spdep nb object) into a spatial edge `data.table`** with columns `(id, neighbor_id)`. This is ~1.37M rows and is built **once**.

2. **For each year**, subset the panel, join cell attributes onto the edge table by `neighbor_id`, then group-by `id` to compute `max`, `min`, `mean` for each neighbor source variable — all vectorized in `data.table`.

3. **Join the resulting neighbor stats back** onto the main panel `data.table` by `(id, year)`.

4. **Predict** with the existing trained Random Forest model (unchanged).

**Expected speedup:** The entire neighbor-feature computation should drop from ~86 hours to **minutes** (typically 2–10 minutes depending on disk I/O), because:
- The edge table is 1.37M rows (not 6.46M).
- `data.table` grouped aggregation is C-level, vectorized, and cache-friendly.
- No character-key construction or named-vector lookup.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 0: Convert cell_data to data.table if not already
# ──────────────────────────────────────────────────────────────────────
cell_dt <- as.data.table(cell_data)

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build the spatial edge table ONCE from the nb object
#
#   rook_neighbors_unique : spdep nb object (list of integer vectors)
#   id_order              : vector mapping position index → cell id
#
#   Result: edge_dt with columns (id, neighbor_id), ~1.37M rows
# ──────────────────────────────────────────────────────────────────────
build_edge_table <- function(id_order, neighbors) {
  # neighbors[[i]] gives integer indices of neighbors for cell at position i
  from <- rep(
    seq_along(neighbors),
    times = lengths(neighbors)
  )
  to <- unlist(neighbors, use.names = FALSE)

  # Remove any zero-length / empty-neighbor entries (already handled by rep/unlist)
  # Map positional indices to actual cell IDs
  edge_dt <- data.table(
    id          = id_order[from],
    neighbor_id = id_order[to]
  )
  return(edge_dt)
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

cat(sprintf(
  "Edge table built: %s directed neighbor pairs (expected ~1,373,394)\n",
  format(nrow(edge_dt), big.mark = ",")
))

# ──────────────────────────────────────────────────────────────────────
# STEP 2: Compute neighbor stats for all variables, all years at once
#
#   Strategy:
#     - Take the edge table (id, neighbor_id).
#     - Join neighbor attributes from cell_dt by (neighbor_id, year).
#     - Group by (id, year) and compute max, min, mean.
#
#   This is fully vectorized in data.table's C backend.
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Subset only the columns we need for the join (save memory on 16 GB laptop)
join_cols <- c("id", "year", neighbor_source_vars)
attr_dt   <- cell_dt[, ..join_cols]

# Rename 'id' to 'neighbor_id' so we can join on neighbor_id + year
setnames(attr_dt, "id", "neighbor_id")

# Key the attribute table for fast join
setkey(attr_dt, neighbor_id, year)

# Cross the edge table with every year present in the data
years <- sort(unique(cell_dt$year))

# Expand edge_dt × years  (~1.37M edges × 28 years ≈ 38.5M rows)
# On 16 GB RAM this is feasible: 38.5M × (2 ints + 1 int for year) ≈ < 1 GB
edge_year_dt <- CJ_edge_year <- edge_dt[, .(year = years), by = .(id, neighbor_id)]

cat(sprintf(
  "Edge-year table: %s rows\n",
  format(nrow(edge_year_dt), big.mark = ",")
))

# Join neighbor attributes onto the edge-year table
setkey(edge_year_dt, neighbor_id, year)
edge_year_dt <- attr_dt[edge_year_dt, on = .(neighbor_id, year), nomatch = NA]

# Now edge_year_dt has columns:
#   neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2, id
# where the variable values belong to the NEIGHBOR cell.

# ──────────────────────────────────────────────────────────────────────
# STEP 3: Grouped aggregation — compute max, min, mean per (id, year)
# ──────────────────────────────────────────────────────────────────────

# Build the aggregation expressions dynamically
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(as.numeric(max(.(as.name(v)),   na.rm = TRUE))),
    bquote(as.numeric(min(.(as.name(v)),   na.rm = TRUE))),
    bquote(mean(.(as.name(v)), na.rm = TRUE))
  )
}))

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

names(agg_exprs) <- agg_names

# Evaluate the aggregation
neighbor_stats <- edge_year_dt[,
  lapply(agg_exprs, eval, envir = .SD),
  by = .(id, year)
]

# Handle cells with no valid neighbors: max/min of empty → -Inf/Inf → set to NA
for (col_name in agg_names) {
  vals <- neighbor_stats[[col_name]]
  set(neighbor_stats, which(is.infinite(vals)), col_name, NA_real_)
}

cat(sprintf(
  "Neighbor stats computed: %s rows × %s new columns\n",
  format(nrow(neighbor_stats), big.mark = ","),
  length(agg_names)
))

# ──────────────────────────────────────────────────────────────────────
# STEP 4: Join neighbor stats back onto the main panel
# ──────────────────────────────────────────────────────────────────────

# Remove old neighbor columns if they exist (from prior runs)
old_cols <- intersect(agg_names, names(cell_dt))
if (length(old_cols) > 0) {
  cell_dt[, (old_cols) := NULL]
}

setkey(cell_dt, id, year)
setkey(neighbor_stats, id, year)

cell_dt <- neighbor_stats[cell_dt, on = .(id, year)]

cat("Neighbor features joined to main panel.\n")

# ──────────────────────────────────────────────────────────────────────
# STEP 5: Predict with the EXISTING trained Random Forest (unchanged)
# ──────────────────────────────────────────────────────────────────────
# rf_model is the pre-trained model object already in memory.
# The predictor column names in cell_dt must match those used at training.

cell_dt[, rf_prediction := predict(rf_model, newdata = .SD)]

cat("Predictions complete.\n")

# Convert back to data.frame if downstream code expects one
cell_data <- as.data.frame(cell_dt)
```

---

### If the `CJ`-style expansion is too large for 16 GB RAM

The edge-year expansion (~38.5M rows × several numeric columns) may push memory on a 16 GB machine. Here is a **year-chunked** alternative for Step 2–3 that processes one year at a time and is still extremely fast:

```r
# ──────────────────────────────────────────────────────────────────────
# MEMORY-SAFE VARIANT: Process one year at a time
# ──────────────────────────────────────────────────────────────────────

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
join_cols <- c("id", "year", neighbor_source_vars)

all_stats <- rbindlist(lapply(years, function(yr) {

  # Subset this year's attributes
  yr_attr <- cell_dt[year == yr, ..join_cols]
  setnames(yr_attr, "id", "neighbor_id")
  setkey(yr_attr, neighbor_id)

  # Join neighbor attributes onto the spatial edge table
  merged <- yr_attr[edge_dt, on = .(neighbor_id), nomatch = NA, allow.cartesian = TRUE]
  # merged now has: neighbor_id, year, <vars>, id

  # Aggregate by focal cell id
  stats <- merged[, {
    out <- vector("list", length(neighbor_source_vars) * 3L)
    k <- 0L
    for (v in neighbor_source_vars) {
      vals <- .SD[[v]]
      vals <- vals[!is.na(vals)]
      n <- length(vals)
      k <- k + 1L; out[[k]] <- if (n > 0) max(vals)  else NA_real_
      k <- k + 1L; out[[k]] <- if (n > 0) min(vals)  else NA_real_
      k <- k + 1L; out[[k]] <- if (n > 0) mean(vals) else NA_real_
    }
    names(out) <- paste0("neighbor_",
      rep(c("max_", "min_", "mean_"), length(neighbor_source_vars)),
      rep(neighbor_source_vars, each = 3L)
    )
    out
  }, by = .(id)]

  stats[, year := yr]
  stats
}))

cat(sprintf(
  "Neighbor stats (chunked): %s rows × %s new columns\n",
  format(nrow(all_stats), big.mark = ","),
  ncol(all_stats) - 2L
))

# Join back onto main panel (same as Step 4 above)
agg_names <- setdiff(names(all_stats), c("id", "year"))
old_cols  <- intersect(agg_names, names(cell_dt))
if (length(old_cols) > 0) cell_dt[, (old_cols) := NULL]

setkey(all_stats, id, year)
setkey(cell_dt,   id, year)
cell_dt <- all_stats[cell_dt, on = .(id, year)]
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M R-level iterations with character-key hashing | 1.37M-row edge `data.table` built once |
| **Stat computation** | `lapply` over 6.46M rows × 5 vars = 32.3M R calls | Vectorized `data.table` grouped aggregation |
| **Estimated runtime** | ~86+ hours | **~2–10 minutes** |
| **RAM footprint** | Large named vectors + list of 6.46M integer vectors | ~1–2 GB peak (year-chunked variant: ~0.5 GB peak) |
| **Trained RF model** | ✅ Preserved, unchanged | ✅ Preserved, unchanged |
| **Numerical results** | `max`, `min`, `mean` of rook-neighbor values | Identical `max`, `min`, `mean` — same estimand |