 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It creates a list of length ~6.46 million, where each element is built by:

1. **Character key construction and lookup** (`paste` + named-vector indexing) — for every single row. Named-vector lookup in R is O(n) in the worst case per query because it uses linear hashing with potential collisions, and doing this ~6.46M times with a lookup vector of ~6.46M entries is catastrophic.
2. **`lapply` over 6.46M rows** — each iteration does string pasting, named-vector subsetting, and NA filtering. The per-element overhead of `lapply` at this scale is significant.
3. **`compute_neighbor_stats`** then does *another* `lapply` over 6.46M elements, 5 times (once per variable). That's 32.3M R-level function calls with vector subsetting inside each.

**Root cause summary:**
- Named character vector indexing is used as a hash map but is extremely slow at scale in R.
- The entire approach is row-wise in R (no vectorization).
- The neighbor lookup is ~6.46M list elements, each constructed via string operations — this alone likely accounts for 80+ hours.

## Optimization Strategy

1. **Replace character-key lookups with integer-arithmetic joins.** Since years are contiguous (1992–2019, 28 years), we can compute the row index of any (cell, year) pair arithmetically if the data is sorted by (id, year). Row index = `(cell_position - 1) * 28 + (year - 1992) + 1`. This eliminates all `paste`/string operations and named-vector lookups.

2. **Vectorize neighbor stat computation using `data.table`.** Expand the neighbor list into an edge list (from_row, to_row), join the variable values, and compute grouped `max`, `min`, `mean` in one vectorized pass per variable.

3. **Avoid creating the 6.46M-element list entirely.** The edge-list approach replaces it with a two-column integer matrix (~1.37M edges × 28 years ≈ 38.5M rows), which `data.table` handles in seconds.

**Expected speedup:** From 86+ hours to **~2–5 minutes**.

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 0.  Ensure cell_data is a data.table sorted by (id, year)
# ──────────────────────────────────────────────────────────────────────
cell_dt <- as.data.table(cell_data)
setorder(cell_dt, id, year)              # sort in place
cell_dt[, row_idx := .I]                 # row index 1..N

# ──────────────────────────────────────────────────────────────────────
# 1.  Build an integer mapping:  cell id  →  position (1-based)
#     and store year range info
# ──────────────────────────────────────────────────────────────────────
id_order_vec  <- id_order                          # length = 344,208
n_cells       <- length(id_order_vec)
year_min      <- 1992L
n_years       <- 28L                               # 1992-2019

# Map from cell id to its 1-based position in id_order
id_to_pos <- integer(max(id_order_vec))            # direct-address table
id_to_pos[id_order_vec] <- seq_along(id_order_vec)
# If ids are not contiguous / too large, use data.table or environment:
# But for typical grid-cell integer ids this is fine.
# Fallback for very large / sparse ids:
if (max(id_order_vec) > 5e7) {
  id_to_pos_env <- new.env(hash = TRUE, size = n_cells)
  for (k in seq_along(id_order_vec)) {
    id_to_pos_env[[as.character(id_order_vec[k])]] <- k
  }
  get_pos <- function(ids) {
    vapply(as.character(ids), function(x) id_to_pos_env[[x]], integer(1),
           USE.NAMES = FALSE)
  }
} else {
  get_pos <- function(ids) id_to_pos[ids]
}

# ──────────────────────────────────────────────────────────────────────
# 2.  Build directed edge list  (from_cell_pos, to_cell_pos)
#     from the spdep::nb object  rook_neighbors_unique
# ──────────────────────────────────────────────────────────────────────
from_pos <- rep(
  seq_along(rook_neighbors_unique),
  lengths(rook_neighbors_unique)
)
to_pos <- unlist(rook_neighbors_unique, use.names = FALSE)

# Remove the spdep convention where 0 means "no neighbors"
valid <- to_pos != 0L
from_pos <- from_pos[valid]
to_pos   <- to_pos[valid]

edges <- data.table(from_pos = from_pos, to_pos = to_pos)
cat("Directed edges (cell-level):", nrow(edges), "\n")

# ──────────────────────────────────────────────────────────────────────
# 3.  Expand edges across all 28 years to get (from_row, to_row)
#     Row index formula (data sorted by id, year):
#       row_of(cell_pos, year) = (cell_pos - 1) * n_years + (year - year_min) + 1
#
#     We verify the sort assumption:
# ──────────────────────────────────────────────────────────────────────
# Verify mapping is correct for a sample
stopifnot(all(cell_dt$id == id_order_vec[
  rep(seq_len(n_cells), each = n_years)
]))

years_vec <- seq.int(year_min, year_min + n_years - 1L)

# Cross join edges × years  (38.5M rows, 3 integer cols ≈ 460 MB)
edge_years <- CJ(edge_idx = seq_len(nrow(edges)), year = years_vec)
edge_years[, `:=`(
  from_row = (edges$from_pos[edge_idx] - 1L) * n_years + (year - year_min) + 1L,
  to_row   = (edges$to_pos[edge_idx]   - 1L) * n_years + (year - year_min) + 1L
)]
edge_years[, edge_idx := NULL]

cat("Edge-year rows:", nrow(edge_years), "\n")

# ──────────────────────────────────────────────────────────────────────
# 4.  Compute neighbor stats per variable — fully vectorized
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Processing neighbor stats for:", var_name, "\n")

  # Attach the neighbor's value to each edge-year row
  edge_years[, nbr_val := cell_dt[[var_name]][to_row]]

  # Compute grouped stats by from_row (= the focal cell-year)
  stats <- edge_years[
    !is.na(nbr_val),
    .(
      nb_max  = max(nbr_val),
      nb_min  = min(nbr_val),
      nb_mean = mean(nbr_val)
    ),
    keyby = from_row
  ]

  # Initialize columns with NA
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")

  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]

  # Assign results
  cell_dt[stats$from_row, (max_col)  := stats$nb_max]
  cell_dt[stats$from_row, (min_col)  := stats$nb_min]
  cell_dt[stats$from_row, (mean_col) := stats$nb_mean]

  # Clean up for next iteration
  edge_years[, nbr_val := NULL]
  rm(stats)
}

# ──────────────────────────────────────────────────────────────────────
# 5.  Convert back to data.frame if needed downstream
# ──────────────────────────────────────────────────────────────────────
cell_dt[, row_idx := NULL]
cell_data <- as.data.frame(cell_dt)

# The trained Random Forest model is untouched — only input features
# were computed. The numerical estimand is preserved exactly:
#   neighbor_max  = max of non-NA rook-neighbor values
#   neighbor_min  = min of non-NA rook-neighbor values
#   neighbor_mean = mean of non-NA rook-neighbor values
# These are identical to the original compute_neighbor_stats definitions.
```

## Memory Note

The `edge_years` table is the largest object (~38.5M rows × 3 integer columns + 1 double column ≈ 460–770 MB). On a 16 GB laptop this is feasible. If memory is tight, process in year-batches:

```r
for (yr in years_vec) {
  ey <- edges[, .(
    from_row = (from_pos - 1L) * n_years + (yr - year_min) + 1L,
    to_row   = (to_pos   - 1L) * n_years + (yr - year_min) + 1L
  )]
  # ... compute stats for this year-slice and rbind
}
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Lookup construction | 6.46M string pastes + named-vector lookups | Integer arithmetic, O(1) per edge |
| Stat computation | 6.46M × 5 `lapply` calls | 5 vectorized `data.table` group-bys |
| Data structure | 6.46M-element R list | Single edge-list `data.table` |
| Estimated runtime | 86+ hours | **2–5 minutes** |
| Numerical result | max/min/mean of non-NA rook neighbors | **Identical** |
| RF model | Untouched | **Untouched** |