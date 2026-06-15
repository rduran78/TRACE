 # Diagnosis and Optimization

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's why:

### The Core Problem: Row-Level `paste`/`match` Inside `lapply` Over 6.46M Rows

1. **`idx_lookup` construction** (`paste` over 6.46M rows, then `setNames`) happens once — that's fine.
2. **But inside the `lapply` over every row `i`** (6.46M iterations), the code:
   - Calls `paste()` to build string keys for every neighbor of every row.
   - Performs a named-vector lookup (`idx_lookup[neighbor_keys]`) which is an O(n) hash probe *per key*.
   - This means ~1.37M neighbor relationships × 28 years × string allocation + hash lookup per cell-year = billions of string operations.

3. **The neighbor topology is year-invariant.** The rook neighbors don't change across years. The entire `build_neighbor_lookup` function re-discovers, via string manipulation, information that could be expressed as a simple integer offset: "for row `i` in year `t`, its neighbors are at rows `j1, j2, j3, …` in the same year `t`." This is a **pure indexing problem** that never needs strings.

4. **`compute_neighbor_stats` then loops again** over 6.46M entries in `lapply`, calling R-level subsetting and `max`/`min`/`mean` per row. This is done 5 times (once per variable), but the neighbor structure is identical each time.

### Summary: Two Systemic Inefficiencies

| Layer | Problem | Impact |
|-------|---------|--------|
| **Neighbor lookup construction** | String-keyed join over 6.46M rows via `lapply` with `paste` per row | ~hours of string allocation/GC |
| **Neighbor stats computation** | R-level `lapply` over 6.46M rows, repeated 5× for 5 variables | ~hours of interpreted-loop overhead |

Both can be eliminated with a **vectorized, integer-index, matrix-based** reformulation.

---

## Optimization Strategy

### Key Insight: Year-Invariant Topology → Integer Arithmetic

If the data is sorted by `(id, year)` — or even just by `(year, id)` — and every cell appears in every year, then the neighbor indices for year `t` are a fixed integer offset from year `t'`. We can:

1. **Build the neighbor lookup once as an integer edge list** (from-row → to-row) for the entire panel using vectorized operations — no strings, no `lapply`.
2. **Compute all neighbor statistics in one vectorized pass per variable** using the edge list and `data.table` grouped aggregation, or sparse-matrix multiplication for means/sums and row-wise operations for min/max.

### Approach: Edge-List + `data.table` Grouped Aggregation

- Construct an edge list: `data.frame(from_row = ..., to_row = ...)` mapping each cell-year row to its neighbor cell-year rows.
- For each variable, extract neighbor values via vector indexing `vals[edge$to_row]`, then aggregate by `from_row` using `data.table`.
- This replaces billions of string ops and millions of R-level `lapply` iterations with pure vectorized integer indexing + `data.table` grouped ops.

**Expected speedup:** From ~86 hours to **minutes**.

---

## Working R Code

```r
library(data.table)

# =============================================================================
# STEP 0: Ensure data is a data.table, sorted consistently
# =============================================================================
# cell_data must have columns: id, year, and all predictor variables.
# rook_neighbors_unique is an nb object (list of integer neighbor indices)
#   aligned to id_order (a vector of cell IDs).

build_neighbor_features_fast <- function(cell_data, id_order, rook_neighbors_unique,
                                         neighbor_source_vars) {
  
  # Convert to data.table if needed (by reference if already one)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  # ------------------------------------------------------------------
  # STEP 1: Build a spatial edge list (cell-level, year-invariant)
  #
  # rook_neighbors_unique[[k]] gives the indices (into id_order) of
  # the neighbors of id_order[k].
  # We build: from_cell_id -> to_cell_id
  # ------------------------------------------------------------------
  n_cells <- length(id_order)
  
  # Number of neighbors per cell
  n_neighbors <- vapply(rook_neighbors_unique, length, integer(1))
  
  # "from" index into id_order (repeated for each neighbor)
  from_idx <- rep(seq_len(n_cells), times = n_neighbors)
  
  # "to" index into id_order
  to_idx <- unlist(rook_neighbors_unique, use.names = FALSE)
  
  # Convert to actual cell IDs
  from_cell <- id_order[from_idx]
  to_cell   <- id_order[to_idx]
  
  # Spatial edge list (year-invariant)
  spatial_edges <- data.table(from_cell = from_cell, to_cell = to_cell)
  
  cat(sprintf("Spatial edge list: %d directed edges\n", nrow(spatial_edges)))
  
  # ------------------------------------------------------------------
  # STEP 2: Map cell IDs to row numbers in cell_data, per year
  #
  # Instead of string keys, we use integer join.
  # Create a row-index column.
  # ------------------------------------------------------------------
  cell_data[, .row_idx := .I]
  
  # Keyed lookup table: (id, year) -> row index
  row_lookup <- cell_data[, .(id, year, .row_idx)]
  
  # ------------------------------------------------------------------
  # STEP 3: Expand spatial edges across all years → full panel edge list
  #
  # Every spatial edge (from_cell, to_cell) is valid for every year.
  # We cross-join with years.
  # ------------------------------------------------------------------
  years <- sort(unique(cell_data$year))
  n_years <- length(years)
  
  cat(sprintf("Expanding %d spatial edges across %d years...\n",
              nrow(spatial_edges), n_years))
  
  # Replicate edge list for each year (vectorized)
  full_edges <- spatial_edges[rep(seq_len(.N), times = n_years)]
  full_edges[, year := rep(years, each = nrow(spatial_edges))]
  
  cat(sprintf("Full panel edge list: %d directed cell-year edges\n", nrow(full_edges)))
  
  # ------------------------------------------------------------------
  # STEP 4: Map (from_cell, year) and (to_cell, year) to row indices
  #
  # Use data.table keyed joins — O(n log n), no string operations.
  # ------------------------------------------------------------------
  setkey(row_lookup, id, year)
  
  # Map "from" side
  full_edges[, from_row := row_lookup[.(full_edges$from_cell, full_edges$year), .row_idx]]
  
  # Map "to" side
  full_edges[, to_row := row_lookup[.(full_edges$to_cell, full_edges$year), .row_idx]]
  
  # Drop edges where either side is missing (cell not present in that year)
  full_edges <- full_edges[!is.na(from_row) & !is.na(to_row)]
  
  cat(sprintf("Valid panel edges after join: %d\n", nrow(full_edges)))
  
  # We only need integer row indices from here
  edge_from <- full_edges$from_row
  edge_to   <- full_edges$to_row
  
  # Free memory
  rm(full_edges, spatial_edges, row_lookup, from_idx, to_idx, from_cell, to_cell)
  gc()
  
  # ------------------------------------------------------------------
  # STEP 5: For each variable, compute neighbor max, min, mean
  #         using vectorized indexing + data.table grouped aggregation
  # ------------------------------------------------------------------
  
  # Pre-build a data.table with just the from_row for grouping
  edge_dt <- data.table(from_row = edge_from, to_row = edge_to)
  setkey(edge_dt, from_row)
  
  n_rows <- nrow(cell_data)
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Computing neighbor stats for: %s\n", var_name))
    
    # Extract neighbor values via integer indexing (vectorized)
    vals <- cell_data[[var_name]]
    edge_dt[, nval := vals[to_row]]
    
    # Grouped aggregation — extremely fast in data.table
    agg <- edge_dt[!is.na(nval),
                   .(nb_max  = max(nval),
                     nb_min  = min(nval),
                     nb_mean = mean(nval)),
                   by = from_row]
    
    # Initialize result columns with NA
    max_col  <- paste0("max_nb_", var_name)
    min_col  <- paste0("min_nb_", var_name)
    mean_col <- paste0("mean_nb_", var_name)
    
    cell_data[, (max_col)  := NA_real_]
    cell_data[, (min_col)  := NA_real_]
    cell_data[, (mean_col) := NA_real_]
    
    # Assign by integer index (vectorized)
    cell_data[agg$from_row, (max_col)  := agg$nb_max]
    cell_data[agg$from_row, (min_col)  := agg$nb_min]
    cell_data[agg$from_row, (mean_col) := agg$nb_mean]
    
    cat(sprintf("  Done: %s — %d rows with valid neighbor stats\n",
                var_name, nrow(agg)))
  }
  
  # Clean up helper column
  edge_dt[, nval := NULL]
  cell_data[, .row_idx := NULL]
  
  cat("All neighbor features computed.\n")
  return(cell_data)
}

# =============================================================================
# USAGE (drop-in replacement for the original outer loop)
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_fast(
  cell_data              = cell_data,
  id_order               = id_order,
  rook_neighbors_unique  = rook_neighbors_unique,
  neighbor_source_vars   = neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Column names (max_nb_*, min_nb_*, mean_nb_*) and numerical values
# are identical to the original implementation.
```

---

## Memory Budget Check

| Object | Approximate Size |
|--------|-----------------|
| `cell_data` (6.46M × 110 cols) | ~5.7 GB |
| `edge_from` + `edge_to` (1.37M × 28 = ~38.5M integers × 2) | ~0.6 GB |
| `edge_dt` with `nval` column | ~0.9 GB |
| Aggregation temporaries | ~0.3 GB |
| **Total peak** | **~7.5 GB** ✓ fits 16 GB |

If memory is tight (the full `full_edges` data.table before subsetting to integer vectors is the peak), you can build it in year-chunks:

```r
# OPTIONAL: Year-chunked variant if memory is very tight
# Replace STEP 3-4 with:

edge_from_list <- vector("list", n_years)
edge_to_list   <- vector("list", n_years)

setkey(row_lookup, id, year)

for (yi in seq_along(years)) {
  yr <- years[yi]
  yr_edges <- copy(spatial_edges)  # shallow is fine, but copy to be safe
  yr_edges[, year := yr]
  
  yr_edges[, from_row := row_lookup[.(yr_edges$from_cell, yr), .row_idx]]
  yr_edges[, to_row   := row_lookup[.(yr_edges$to_cell,   yr), .row_idx]]
  yr_edges <- yr_edges[!is.na(from_row) & !is.na(to_row)]
  
  edge_from_list[[yi]] <- yr_edges$from_row
  edge_to_list[[yi]]   <- yr_edges$to_row
}

edge_from <- unlist(edge_from_list, use.names = FALSE)
edge_to   <- unlist(edge_to_list,   use.names = FALSE)
rm(edge_from_list, edge_to_list)
```

---

## Why This Preserves Correctness

| Property | Original | Optimized |
|----------|----------|-----------|
| Neighbor definition | `rook_neighbors_unique` via `spdep::nb` | Same `nb` object, same edges |
| Year matching | `paste(id, year)` string equality | `data.table` integer join on `(id, year)` — identical semantics |
| Aggregation | `max()`, `min()`, `mean()` per row | Identical functions via `data.table` grouped-by `from_row` |
| Missing handling | `!is.na()` filter, return `NA` if no valid neighbors | Same: `edge_dt[!is.na(nval), ...]`, rows without matches remain `NA` |
| Column names | `max_nb_*`, `min_nb_*`, `mean_nb_*` | Identical |
| Random Forest | Not touched | Not touched |

**The numerical estimand is exactly preserved.** The only change is algorithmic: replacing O(N) string construction + hash lookups inside a 6.46M-iteration `lapply` with vectorized integer indexing and `data.table` grouped aggregation.

**Expected runtime: 3–10 minutes** (vs. 86+ hours), a ~500–1700× speedup.