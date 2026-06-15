 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It creates a list of 6.46 million elements, where each element is built by:

1. **Character key construction and lookup** (`paste`, `setNames`, named-vector indexing) — for every single row. Named-vector lookup in R is O(n) per query in the worst case because R's named vectors use linear hashing with potential collisions, and here the lookup table has ~6.46M entries. Doing this 6.46M times yields an effective O(n²) operation.
2. **`lapply` over 6.46M rows** — each iteration does string pasting, named-vector subsetting, and NA filtering. The per-iteration overhead of R's interpreted loop is enormous at this scale.

Secondary cost: `compute_neighbor_stats` also uses `lapply` over 6.46M elements, but each iteration is cheap (a few numeric operations). Still, 6.46M R-level function calls add up.

**Estimated cost breakdown:**
- `build_neighbor_lookup`: ~80+ hours (dominant)
- `compute_neighbor_stats` × 5 variables: ~5–6 hours
- Total: ~86+ hours

## Optimization Strategy

### 1. Replace named-vector lookup with `data.table` hash joins (O(1) amortized)

Instead of building a 6.46M-entry named character vector and indexing into it row-by-row, we:
- Create a `data.table` keyed on `(id, year)` with a row-index column.
- Expand the neighbor list into an edge table: `(source_row, neighbor_cell_id)`.
- Join the edge table against the keyed `data.table` to resolve `(neighbor_cell_id, year)` → `neighbor_row` in one vectorized hash join.

This replaces 6.46M interpreted R iterations with a single vectorized join.

### 2. Replace `lapply`-based stats with `data.table` grouped aggregation

Once we have an edge table `(source_row, neighbor_row)`, we pull the variable values for all neighbor rows, then `group by source_row` to compute `max`, `min`, `mean` — all vectorized in C via `data.table`.

### 3. Memory management

- The edge table will have ~6.46M × 4 neighbors ≈ 26M rows (but actually ~1.37M directed edges × 28 years ≈ 38.5M rows). At ~3 integer columns, this is ~900 MB — fits in 16 GB.
- We process one variable at a time and discard intermediate objects.

**Expected runtime: ~2–5 minutes total** (down from 86+ hours).

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# Step 0: Convert cell_data to data.table if not already.
#         Assumes cell_data has columns: id, year, and the source vars.
#         Assumes id_order is a vector of cell IDs in the same order as
#         rook_neighbors_unique (the spdep nb object).
# ──────────────────────────────────────────────────────────────────────

if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Preserve original row order so downstream predictions are aligned
cell_data[, .row_idx := .I]

# ──────────────────────────────────────────────────────────────────────
# Step 1: Build the spatial edge list (cell-level, time-invariant)
#
#   rook_neighbors_unique[[i]] gives the indices (into id_order) of
#   the rook neighbors of cell id_order[i].
#
#   We expand this into a two-column data.table:
#     (focal_id, neighbor_id)
# ──────────────────────────────────────────────────────────────────────

build_edge_list <- function(id_order, nb_obj) {
  # nb_obj is a list of integer vectors (spdep nb object)
  n <- length(nb_obj)
  # Pre-compute lengths for pre-allocation
  lens <- vapply(nb_obj, length, integer(1))
  total_edges <- sum(lens)

  focal_idx    <- rep.int(seq_len(n), lens)
  neighbor_idx <- unlist(nb_obj, use.names = FALSE)

  # Remove the spdep convention where 0 means "no neighbors"
  valid <- neighbor_idx != 0L
  focal_idx    <- focal_idx[valid]
  neighbor_idx <- neighbor_idx[valid]

  data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)

cat(sprintf("Edge list: %d directed cell-level edges\n", nrow(edge_dt)))

# ──────────────────────────────────────────────────────────────────────
# Step 2: Expand edges across years by joining to the panel
#
#   For every (focal_id, year) row in cell_data, we need the row
#   indices of (neighbor_id, year).  We do this with two keyed joins.
# ──────────────────────────────────────────────────────────────────────

# Create a lookup: (id, year) -> row index in cell_data
row_lookup <- cell_data[, .(id, year, .row_idx)]
setkey(row_lookup, id, year)

# Cross the edge list with all years present in the data
all_years <- sort(unique(cell_data$year))

# Expand: each spatial edge × each year
# Memory: ~38.5M rows × 4 cols (ints) ≈ manageable
edge_year <- CJ(edge_idx = seq_len(nrow(edge_dt)), year = all_years)
edge_year[, focal_id    := edge_dt$focal_id[edge_idx]]
edge_year[, neighbor_id := edge_dt$neighbor_id[edge_idx]]
edge_year[, edge_idx := NULL]

# Join to get focal row index
setkey(edge_year, focal_id, year)
edge_year <- row_lookup[edge_year, on = .(id = focal_id, year = year), nomatch = 0L]
setnames(edge_year, ".row_idx", "focal_row")

# Join to get neighbor row index
setkey(edge_year, neighbor_id, year)
edge_year <- row_lookup[edge_year, on = .(id = neighbor_id, year = year), nomatch = 0L]
setnames(edge_year, ".row_idx", "neighbor_row")

# Keep only what we need
edge_year <- edge_year[, .(focal_row, neighbor_row)]
setkey(edge_year, focal_row)

cat(sprintf("Panel edge table: %d directed cell-year edges\n", nrow(edge_year)))

# Free memory
rm(row_lookup)
gc()

# ──────────────────────────────────────────────────────────────────────
# Step 3: Compute neighbor stats for each variable — fully vectorized
# ──────────────────────────────────────────────────────────────────────

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {

  cat(sprintf("Computing neighbor stats for: %s\n", var_name))

  # Pull neighbor values into the edge table
  edge_year[, val := cell_data[[var_name]][neighbor_row]]

  # Aggregate by focal_row, dropping NAs
  stats <- edge_year[!is.na(val),
    .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ),
    by = focal_row
  ]

  # Initialize new columns with NA
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]

  # Assign results back by row index
  cell_data[stats$focal_row, (max_col)  := stats$nb_max]
  cell_data[stats$focal_row, (min_col)  := stats$nb_min]
  cell_data[stats$focal_row, (mean_col) := stats$nb_mean]

  # Clean up the temporary column in edge_year
  edge_year[, val := NULL]

  cat(sprintf("  Done: %s — %d rows with neighbor stats\n", var_name, nrow(stats)))
  rm(stats)
  gc()
}

# ──────────────────────────────────────────────────────────────────────
# Step 4: Restore original row order and clean up
# ──────────────────────────────────────────────────────────────────────

setorder(cell_data, .row_idx)
cell_data[, .row_idx := NULL]

# If downstream code expects a data.frame:
# cell_data <- as.data.frame(cell_data)

cat("Neighbor feature engineering complete.\n")
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same neighbor relationships** | `build_edge_list` reads directly from the same `rook_neighbors_unique` nb object and `id_order`. No relationships are added or dropped. |
| **Same numerical estimand** | `max`, `min`, `mean` are computed on exactly the same sets of non-NA neighbor values per `(focal_id, year)` as the original code. The `data.table` aggregation is numerically identical (IEEE 754 double arithmetic, same operations). |
| **Trained Random Forest untouched** | No model code is modified. The output columns have the same names and values, so `predict(rf_model, cell_data)` produces identical results. |
| **Handles irregular topology** | Cells with zero neighbors in the nb object produce zero edges, so they get `NA` for all neighbor stats — identical to the original `if (length(idx) == 0) return(c(NA, NA, NA))` logic. Boundary cells with fewer than 4 neighbors are handled naturally. |

## Performance Estimate

| Stage | Original | Optimized |
|---|---|---|
| Build lookup / edge table | ~80 hours | ~10–30 seconds |
| Compute stats (5 vars) | ~6 hours | ~1–3 minutes |
| **Total** | **~86 hours** | **~2–5 minutes** |

The speedup factor is roughly **1,000–2,500×**, achieved entirely by replacing interpreted row-by-row string operations with vectorized integer hash joins in `data.table`.