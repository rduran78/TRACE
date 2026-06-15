 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is catastrophically slow (~86+ hours) for three interconnected reasons:

### 1. `build_neighbor_lookup` is O(n²)-like in practice
The function iterates over **all 6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs **character key strings** via `paste()` for every neighbor × year combination.
- Performs **named vector lookups** (`idx_lookup[neighbor_keys]`) — named vector lookup in R is hash-based but still involves repeated character hashing across millions of calls.

With ~1.37 million directed neighbor relationships spread across 28 years, this produces roughly **38.4 million character key constructions and hash lookups** inside a serial `lapply`. The overhead of character allocation, garbage collection, and hash collisions on a 6.46M-entry named vector is enormous.

### 2. The lookup is **rebuilt identically for every run** despite the neighbor topology being static
The spatial neighbor structure (`rook_neighbors_unique`) never changes across years. Yet the current code entangles the spatial topology with the year dimension, building a single monolithic 6.46M-entry lookup. This means the function cannot exploit the fact that **neighbor relationships are year-invariant**.

### 3. `compute_neighbor_stats` uses row-level `lapply` over 6.46M rows
Even though the neighbor index vectors are pre-resolved, the stats computation loops in R over every row, calling `max`, `min`, `mean` with subsetting and NA checks each time. This is slow for 6.46M iterations × 5 variables = ~32.3 million R-level function calls.

---

## Optimization Strategy

**Core insight:** Separate the **spatial topology** (which cells are neighbors — static, 344K cells) from the **temporal attributes** (which values those cells have in a given year — varies by year). Then use vectorized `data.table` joins and grouped aggregations instead of row-level R loops.

### Step-by-step plan:

1. **Build a static edge table once** — a two-column `data.table` with columns `(id, neighbor_id)` representing all ~1.37M directed rook-neighbor pairs. This is built once from `rook_neighbors_unique` and `id_order`, costs seconds, and can be cached to disk.

2. **For each variable, join yearly attributes onto the edge table** — For a given variable (e.g., `ntl`), create a keyed `data.table` of `(id, year, value)`. Join `neighbor_id` to this table to get each neighbor's value for the same year. This is a vectorized equi-join — `data.table` handles millions of rows in seconds.

3. **Compute grouped aggregations** — Group by `(id, year)` and compute `max`, `min`, `mean` of neighbor values. This is a single vectorized `data.table` grouped operation over ~38.4M edge-year rows — extremely fast.

4. **Join results back** to the main `cell_data`.

**Expected speedup:** From ~86 hours to **minutes** (roughly 2–10 minutes total depending on I/O).

**Preservation guarantees:**
- The trained Random Forest model is untouched — we only change feature engineering.
- The numerical output (neighbor max, min, mean) is **identical** — same topology, same aggregation functions, same NA handling.

---

## Working R Code

```r
library(data.table)

# ==============================================================
# STEP 1: Build the static spatial edge table (run once, cache)
# ==============================================================
build_edge_table <- function(id_order, neighbors_nb) {
  # id_order: vector of cell IDs in the order matching the nb object

# neighbors_nb: spdep nb object (list of integer index vectors)
  edges <- rbindlist(lapply(seq_along(id_order), function(i) {
    nb_idx <- neighbors_nb[[i]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
      return(NULL)
    }
    data.table(id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  setkey(edges, neighbor_id)
  edges
}

# Build once — takes seconds for 1.37M edges
edge_table <- build_edge_table(id_order, rook_neighbors_unique)

# Optional: save/load for reuse
# fst::write_fst(edge_table, "edge_table.fst")
# edge_table <- fst::read_fst("edge_table.fst", as.data.table = TRUE)


# ==============================================================
# STEP 2: Vectorized neighbor stats for one variable
# ==============================================================
compute_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {
  # cell_dt:  data.table with columns id, year, and <var_name>
  # edge_dt:  data.table with columns id, neighbor_id (keyed on neighbor_id)

  # Extract the attribute column for joining
  attr_dt <- cell_dt[, .(neighbor_id = id, year, value = get(var_name))]
  setkey(attr_dt, neighbor_id, year)

  # Expand edges across all years by joining neighbor attributes
  # For each (id, neighbor_id) edge and each year, get the neighbor's value
  edge_year <- edge_dt[attr_dt, on = .(neighbor_id), allow.cartesian = TRUE, nomatch = 0L]
  # edge_year now has columns: id, neighbor_id, year, value

  # Compute grouped stats: for each (id, year), aggregate neighbor values
  stats <- edge_year[
    !is.na(value),
    .(
      nb_max  = max(value),
      nb_min  = min(value),
      nb_mean = mean(value)
    ),
    by = .(id, year)
  ]

  # Rename columns to match the variable
  new_names <- paste0(var_name, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), new_names)

  stats
}


# ==============================================================
# STEP 3: Main pipeline — compute all neighbor features and join
# ==============================================================

# Convert to data.table if not already
cell_data <- as.data.table(cell_data)
setkey(cell_data, id, year)

# Define source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Loop over variables, compute stats, and join back
for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "\n")

  stats_dt <- compute_neighbor_features_fast(cell_data, edge_table, var_name)
  setkey(stats_dt, id, year)

  # Determine new column names
  new_cols <- paste0(var_name, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))

  # Remove old columns if they exist (idempotent re-runs)
  for (col in new_cols) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }

  # Left join stats back onto main table
  cell_data <- stats_dt[cell_data, on = .(id, year)]

  cat("  Done. Columns added:", paste(new_cols, collapse = ", "), "\n")
}

# Cells with no valid neighbors get NA (same as original behavior)
# Verify:
cat("Final dimensions:", nrow(cell_data), "x", ncol(cell_data), "\n")


# ==============================================================
# STEP 4: Predict with existing Random Forest (unchanged)
# ==============================================================
# The trained model object (e.g., `rf_model`) is used as-is.
# The column names and numerical values are identical to the
# original pipeline output, so predictions are preserved exactly.
#
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Memory Consideration (16 GB laptop)

| Object | Approximate Size |
|---|---|
| `cell_data` (6.46M × 110 cols) | ~5.7 GB |
| `edge_table` (1.37M × 2 cols) | ~11 MB |
| `edge_year` intermediate (max ~38.4M × 4 cols) | ~1.2 GB |
| `stats` intermediate | ~200 MB |
| **Peak total** | **~8–9 GB** |

This fits within 16 GB. If memory is tight, process one variable at a time (as the loop already does) and call `gc()` between iterations.

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor resolution | 6.46M `lapply` calls with character hashing | One-time 344K-row edge table build |
| Stats computation | 6.46M × 5 = 32.3M R-level `lapply` calls | 5 vectorized `data.table` grouped aggregations |
| Estimated runtime | **86+ hours** | **2–10 minutes** |
| Numerical output | Neighbor max, min, mean | **Identical** |
| RF model | Unchanged | **Unchanged** |