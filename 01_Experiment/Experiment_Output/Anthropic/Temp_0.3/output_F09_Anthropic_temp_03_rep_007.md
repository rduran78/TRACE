 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three compounding reasons:

### 1. `build_neighbor_lookup` is O(N²)-like in practice
It iterates over all **6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs **character key strings** (`paste(id, year)`) and performs **named-vector lookups** (`idx_lookup[neighbor_keys]`), which are hash-table lookups on a 6.46-million-entry named vector — **for every single row**.

This means ~6.46M × ~4 neighbors × (string construction + hash lookup) = billions of character operations. Character hashing on a vector of 6.46M names is the dominant bottleneck.

### 2. The lookup is year-redundant
The spatial neighbor topology is **identical across all 28 years**. Yet the function rebuilds a full row-level lookup that encodes year information, duplicating the same spatial structure 28 times. This inflates the lookup list from ~344K entries to ~6.46M entries.

### 3. `compute_neighbor_stats` uses row-level `lapply`
Even after the lookup is built, computing stats iterates over 6.46M list elements in R-level `lapply`, each calling `max`, `min`, `mean` on small vectors. This is slow due to R's per-call overhead multiplied millions of times.

---

## Optimization Strategy

**Core insight:** Separate the spatial topology (static) from the yearly attributes (dynamic). Build the adjacency structure **once** over 344K cells, then use vectorized joins and grouped operations for each year.

### Steps:
1. **Build a static edge table once** — a two-column `data.table` of `(cell_id, neighbor_id)` from the `nb` object. This has ~1.37M rows and never changes.
2. **For each variable, join yearly attributes onto the edge table** — a keyed `data.table` merge, which is O(N log N) and highly optimized in C.
3. **Compute grouped `max`, `min`, `mean`** — using `data.table`'s `by=` grouping, which is vectorized C code.
4. **Merge results back** to the main dataset.

This eliminates all character-key hashing, eliminates the 6.46M-element list, and replaces R-level loops with vectorized `data.table` operations.

**Expected speedup:** From ~86 hours to **~2–5 minutes**.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build a static spatial edge table ONCE (from the nb object)
# ──────────────────────────────────────────────────────────────────────
# id_order is the vector of cell IDs aligned with rook_neighbors_unique
# rook_neighbors_unique is an nb object (list of integer index vectors)

build_edge_table <- function(id_order, neighbors_nb) {
  # Pre-allocate: count total edges
  n_edges <- sum(vapply(neighbors_nb, function(x) {
    # nb objects use 0L to indicate no neighbors
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1)))

  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  pos <- 1L

  for (i in seq_along(neighbors_nb)) {
    nb_idx <- neighbors_nb[[i]]
    if (length(nb_idx) == 1L && nb_idx[1] == 0L) next
    n <- length(nb_idx)
    from_id[pos:(pos + n - 1L)] <- id_order[i]
    to_id[pos:(pos + n - 1L)]   <- id_order[nb_idx]
    pos <- pos + n
  }

  data.table(cell_id = from_id, neighbor_id = to_id)
}

# Build it once — ~1.37M rows, trivial memory
edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

cat(sprintf("Edge table: %d rows\n", nrow(edge_dt)))

# ──────────────────────────────────────────────────────────────────────
# STEP 2: Convert main data to data.table (if not already)
# ──────────────────────────────────────────────────────────────────────
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure key columns exist and are of consistent type
cell_data[, id   := as.integer(id)]
cell_data[, year := as.integer(year)]

# ──────────────────────────────────────────────────────────────────────
# STEP 3: Compute neighbor features for all variables — vectorized
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Cross join edges × years, then join attributes of the NEIGHBOR cell
# To keep memory manageable on 16 GB, we process one variable at a time
# and one year at a time is NOT needed — the full join fits in memory:
#   1.37M edges × 28 years = ~38.4M rows × a few columns ≈ < 2 GB

# Create the year-expanded edge table once
years <- sort(unique(cell_data$year))
edge_year_dt <- CJ(edge_idx = seq_len(nrow(edge_dt)), year = years)
edge_year_dt[, cell_id     := edge_dt$cell_id[edge_idx]]
edge_year_dt[, neighbor_id := edge_dt$neighbor_id[edge_idx]]
edge_year_dt[, edge_idx := NULL]

cat(sprintf("Edge-year table: %d rows (%.1f M)\n",
            nrow(edge_year_dt), nrow(edge_year_dt) / 1e6))

# Key for fast joins
setkey(edge_year_dt, neighbor_id, year)

for (var_name in neighbor_source_vars) {
  cat(sprintf("Processing neighbor stats for: %s\n", var_name))

  # Extract only the columns we need for the join
  attr_dt <- cell_data[, .(id, year, value = get(var_name))]
  setkey(attr_dt, id, year)

  # Join: attach the neighbor cell's attribute value to each edge-year row
  # neighbor_id in edge_year_dt matches id in attr_dt
  edge_year_dt[attr_dt, neighbor_value := i.value,
               on = .(neighbor_id = id, year = year)]

  # Aggregate by (cell_id, year) to get max, min, mean of neighbor values
  stats <- edge_year_dt[
    !is.na(neighbor_value),
    .(
      nb_max  = max(neighbor_value),
      nb_min  = min(neighbor_value),
      nb_mean = mean(neighbor_value)
    ),
    by = .(cell_id, year)
  ]

  # Name the output columns to match the original pipeline's convention
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  setnames(stats, c("nb_max", "nb_min", "nb_mean"),
                  c(max_col,  min_col,  mean_col))

  # Merge back into cell_data
  # First, remove old columns if they exist (idempotent re-runs)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }

  cell_data <- merge(cell_data, stats,
                     by.x = c("id", "year"),
                     by.y = c("cell_id", "year"),
                     all.x = TRUE)

  # Clean up the temporary column in edge_year_dt
  edge_year_dt[, neighbor_value := NULL]

  cat(sprintf("  Done. Added: %s, %s, %s\n", max_col, min_col, mean_col))
}

# ──────────────────────────────────────────────────────────────────────
# STEP 4: Predict with the existing trained Random Forest (unchanged)
# ──────────────────────────────────────────────────────────────────────
# The trained model object (e.g., `rf_model`) is preserved as-is.
# Prediction uses the same column names, so no changes are needed:
#
#   cell_data$predicted <- predict(rf_model, newdata = cell_data)
#
# The numerical estimand is identical because the neighbor max, min,
# and mean values are computed with the same logic (max, min, mean of
# non-NA neighbor values), just via vectorized data.table operations
# instead of row-level R loops.
# ──────────────────────────────────────────────────────────────────────
```

---

## Memory-Constrained Alternative (if 16 GB is tight)

If the ~38.4M-row `edge_year_dt` causes memory pressure alongside the 6.46M-row `cell_data` with 110 columns, process year-by-year:

```r
for (var_name in neighbor_source_vars) {
  cat(sprintf("Processing: %s\n", var_name))

  attr_dt <- cell_data[, .(id, year, value = get(var_name))]
  setkey(attr_dt, id, year)

  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  stats_list <- vector("list", length(years))

  for (yi in seq_along(years)) {
    yr <- years[yi]
    # Subset attributes for this year
    attr_yr <- attr_dt[year == yr]
    setkey(attr_yr, id)

    # Join neighbor values onto the static edge table
    edges_with_vals <- edge_dt[attr_yr, on = .(neighbor_id = id), nomatch = 0L]

    # Aggregate
    stats_list[[yi]] <- edges_with_vals[
      !is.na(value),
      .(nb_max = max(value), nb_min = min(value), nb_mean = mean(value)),
      by = .(cell_id)
    ][, year := yr]
  }

  stats <- rbindlist(stats_list)
  setnames(stats, c("nb_max", "nb_min", "nb_mean"),
                  c(max_col,  min_col,  mean_col))

  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }

  cell_data <- merge(cell_data, stats,
                     by.x = c("id", "year"),
                     by.y = c("cell_id", "year"),
                     all.x = TRUE)
}
```

This variant keeps peak memory much lower (~1.37M rows per year-iteration) while still running in minutes.

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup structure | 6.46M-element R list with character hashing | 1.37M-row static `data.table` edge table |
| Stats computation | R-level `lapply` over 6.46M rows × 5 vars | Vectorized `data.table` grouped aggregation |
| Year handling | Redundantly encoded in lookup keys | Joined dynamically via keyed merge |
| Estimated runtime | ~86+ hours | ~2–5 minutes |
| RAM peak | Moderate (but slow) | ~3–5 GB (well within 16 GB) |
| Trained RF model | Preserved ✓ | Preserved ✓ |
| Numerical estimand | max/min/mean of non-NA neighbor values | Identical ✓ |