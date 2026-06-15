 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` — O(n) list of per-row lookups via `lapply`

- With ~6.46 million rows, this creates a list of 6.46M elements. Each iteration does string pasting, hash-table lookups (`idx_lookup[neighbor_keys]`), and NA filtering. The string-key approach (`paste(id, year, sep="_")`) is extremely expensive at this scale: it allocates ~6.46M strings just for the keys, then does millions of named-vector lookups (which are O(n) in the worst case for named vectors in R).

### Bottleneck B: `compute_neighbor_stats` — `lapply` over 6.46M elements

- For each of the 5 variables, another `lapply` iterates over 6.46M rows, subsetting a numeric vector, removing NAs, and computing `max`, `min`, `mean`. This is repeated 5 times. The `do.call(rbind, result)` on a 6.46M-element list of 3-element vectors is also very slow (millions of small allocations).

### Why it takes 86+ hours

- ~6.46M iterations × 5 variables × expensive per-element R-level operations = billions of interpreted R operations.
- Named vector lookups, string operations, and `lapply` with closures are among the slowest patterns in R.

### Why raster focal/kernel operations are not directly applicable

The data is an **irregular spatial panel** (spdep `nb` object with variable numbers of neighbors per cell), not a regular raster grid. Focal operations assume a fixed rectangular kernel on a regular grid. Using them would either require rasterizing (losing the exact neighbor topology) or padding/masking, which risks altering the numerical results. **We must preserve the original numerical estimand**, so we use the actual neighbor graph but vectorize the computation.

---

## 2. Optimization Strategy

### Strategy: Fully vectorized sparse-matrix approach

1. **Replace string-key lookups with integer indexing.** Build a fast integer mapping from `(id, year)` → row index using `data.table` or `match` on a compound integer key.

2. **Expand the neighbor graph into a cell-year edge list once.** Instead of building a per-row list, create a two-column integer matrix `(from_row, to_row)` representing all directed neighbor-year edges. With ~1.37M directed neighbor pairs × 28 years ≈ ~38.5M edges, this fits easily in RAM.

3. **Compute stats via vectorized group-by operations.** Use `data.table` to group `to_row` values by `from_row`, then compute `max`, `min`, `mean` per group — all in C-level `data.table` internals. This replaces 6.46M × 5 = 32.3M `lapply` iterations with 5 fast `data.table` aggregations.

4. **Expected speedup:** From 86+ hours to **minutes** (typically 2–10 minutes on a 16 GB laptop).

---

## 3. Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build a fast (id, year) → row_index mapping
# ──────────────────────────────────────────────────────────────────────

# Convert cell_data to data.table (in-place if possible to save RAM)
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Preserve original row order for later
cell_data[, .row_idx := .I]

# Build integer lookup: for every (id, year) → row index
# Using a keyed data.table for O(log n) binary-search joins
id_year_map <- cell_data[, .(id, year, .row_idx)]
setkey(id_year_map, id, year)

# ──────────────────────────────────────────────────────────────────────
# STEP 2: Expand neighbor graph into a cell-year edge list
# ──────────────────────────────────────────────────────────────────────

# id_order is the vector of cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique is an nb object: a list of integer index vectors

# Build directed edge list at the cell level (not yet year-expanded)
# Each element rook_neighbors_unique[[i]] contains indices into id_order
message("Building cell-level edge list...")
n_cells <- length(id_order)

# Pre-compute lengths for pre-allocation
n_lengths <- vapply(rook_neighbors_unique, length, integer(1))
total_edges <- sum(n_lengths)  # ~1.37M

from_cell_idx <- rep(seq_len(n_cells), times = n_lengths)
to_cell_idx   <- unlist(rook_neighbors_unique, use.names = FALSE)

# Map cell indices to actual cell IDs
from_cell_id <- id_order[from_cell_idx]
to_cell_id   <- id_order[to_cell_idx]

# Create cell-level edge data.table
cell_edges <- data.table(from_id = from_cell_id, to_id = to_cell_id)

# Expand across all years via cross join
years <- sort(unique(cell_data$year))
message(sprintf("Expanding %d cell edges across %d years...", nrow(cell_edges), length(years)))

# Memory-efficient expansion: cross join edges × years
cell_edges_expanded <- cell_edges[, .(year = years), by = .(from_id, to_id)]

# Now join to get row indices for both from and to
# Join for 'from' rows
setkey(cell_edges_expanded, from_id, year)
cell_edges_expanded[id_year_map, from_row := i..row_idx, on = .(from_id = id, year = year)]

# Join for 'to' rows (neighbors whose values we read)
setkey(cell_edges_expanded, to_id, year)
cell_edges_expanded[id_year_map, to_row := i..row_idx, on = .(to_id = id, year = year)]

# Drop edges where either side has no matching row
edge_dt <- cell_edges_expanded[!is.na(from_row) & !is.na(to_row), .(from_row, to_row)]

# Clean up intermediates
rm(cell_edges, cell_edges_expanded, from_cell_idx, to_cell_idx,
   from_cell_id, to_cell_id, n_lengths)
gc()

message(sprintf("Edge list ready: %d directed cell-year edges.", nrow(edge_dt)))

# ──────────────────────────────────────────────────────────────────────
# STEP 3: Vectorized neighbor stats via data.table grouping
# ──────────────────────────────────────────────────────────────────────

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor stats for: %s", var_name))
  
  # Extract neighbor values via vectorized indexing
  edge_dt[, val := cell_data[[var_name]][to_row]]
  
  # Aggregate: max, min, mean of non-NA neighbor values, grouped by from_row
  stats <- edge_dt[!is.na(val),
                   .(nb_max  = max(val),
                     nb_min  = min(val),
                     nb_mean = mean(val)),
                   keyby = from_row]
  
  # Prepare output columns (NA for rows with no valid neighbors)
  max_col  <- paste0(var_name, "_nb_max")
  min_col  <- paste0(var_name, "_nb_min")
  mean_col <- paste0(var_name, "_nb_mean")
  
  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]
  
  # Assign results back by row index
  cell_data[stats$from_row, (max_col)  := stats$nb_max]
  cell_data[stats$from_row, (min_col)  := stats$nb_min]
  cell_data[stats$from_row, (mean_col) := stats$nb_mean]
  
  message(sprintf("  Done: %s — %d rows received neighbor stats.", var_name, nrow(stats)))
}

# Clean up edge value column
edge_dt[, val := NULL]

# Remove helper column
cell_data[, .row_idx := NULL]

message("All neighbor features computed.")

# ──────────────────────────────────────────────────────────────────────
# STEP 4: Predict with the pre-trained Random Forest (unchanged)
# ──────────────────────────────────────────────────────────────────────
# The trained RF model object (e.g., `rf_model`) is used as-is:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## 4. Summary of Guarantees

| Requirement | How it is preserved |
|---|---|
| **Numerical estimand** | Identical `max`, `min`, `mean` computed over exactly the same rook-neighbor sets and the same variable values. No approximation, no rasterization, no kernel substitution. |
| **Trained RF model** | Untouched. Only the feature-engineering step is optimized. Column names match the original (`{var}_nb_max`, `{var}_nb_min`, `{var}_nb_mean`). |
| **RAM ≤ 16 GB** | Edge list: ~38.5M rows × 2 int cols ≈ 308 MB. `cell_data`: ~6.46M × 110 cols ≈ 5.7 GB. Total peak ≈ 8–10 GB. Fits comfortably. |
| **Runtime** | Expected **2–10 minutes** vs. 86+ hours. The entire computation is pushed into C-level `data.table` group-by operations with no per-row R interpretation. |