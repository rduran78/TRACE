 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Inside the Loop

`build_neighbor_lookup` iterates over all ~6.46 million rows. For **each row**, it:

1. Converts `data$id[i]` to character and looks it up in `id_to_ref` — **6.46M character coercions + named-vector lookups**.
2. Subsets `id_order[neighbors[[ref_idx]]]` — fine, but done row-by-row.
3. Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` — **allocates a new character vector per row** (~4 neighbor keys × 6.46M rows ≈ 25.8M string constructions).
4. Looks up `idx_lookup[neighbor_keys]` — **named-vector lookup via string matching on a 6.46M-element named vector**, repeated 6.46M times.

The string-key lookup on a named vector of length 6.46M is approximately O(N) per probe (R's named vector lookup uses hashing, but the construction and GC pressure of millions of temporary character vectors is enormous). Total cost: **~6.46M × 4 × hash-lookup = ~25.8M hash probes**, plus the string allocation/GC overhead.

### The Deeper Structural Insight

The neighbor relationship is **spatial only** — it doesn't change across years. Every cell has the same neighbors in every year. The current code re-discovers this for every cell-year row by encoding `(cell_id, year)` as a string key. This is unnecessary. The neighbor lookup can be built **once at the cell level** (344K cells), and then applied across all 28 years via vectorized indexing.

Similarly, `compute_neighbor_stats` runs an `lapply` over 6.46M rows, extracting neighbor values one row at a time. This can be replaced with a **fully vectorized matrix operation**.

### Summary of Inefficiencies

| Layer | Problem | Scale |
|-------|---------|-------|
| String key construction | `paste()` inside 6.46M-iteration loop | ~25.8M string allocs |
| Named vector lookup | Hash probe on 6.46M-element named vector, per row | ~25.8M probes |
| Row-level `lapply` in `build_neighbor_lookup` | Inherently serial R loop over 6.46M rows | 6.46M iterations |
| Row-level `lapply` in `compute_neighbor_stats` | Another serial R loop over 6.46M rows, repeated 5× | 32.3M iterations |
| Redundant structure | Neighbor topology is year-invariant but re-resolved per cell-year | 28× redundant work |

---

## Optimization Strategy

1. **Exploit year-invariance**: Build a neighbor index at the **cell level** (344K entries), not the cell-year level (6.46M entries). The `spdep::nb` object already provides this.

2. **Convert the ragged neighbor list to a CSR (Compressed Sparse Row) representation**: Two integer vectors (`adj_ptr` and `adj_ids`) replace millions of list elements. This enables fully vectorized operations.

3. **Arrange data so that all years for a given cell are contiguous** (or use a cell-to-rows mapping). Then neighbor stats for all years can be computed via vectorized matrix/column operations.

4. **Replace all `lapply` loops with vectorized `rowSums`/`rowMeans`-style operations** using sparse-matrix or direct indexed arithmetic.

5. **Compute all 5 variables' neighbor stats in one pass** over the adjacency structure.

Expected speedup: from ~86+ hours to **minutes** (roughly 3–10 minutes depending on RAM pressure).

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE CONSTRUCTION
# =============================================================================
# Preserves the exact numerical estimand: for each cell-year row and each
# neighbor source variable, compute max, min, and mean of that variable
# across the cell's rook neighbors in the SAME year.
#
# Requirements:
#   - cell_data: data.frame/data.table with columns 'id', 'year', and the
#     neighbor_source_vars. Rows are cell-year observations.
#   - id_order: integer/numeric vector of cell IDs in the order matching
#     rook_neighbors_unique (i.e., id_order[k] is the cell ID for the
#     k-th element of the nb object).
#   - rook_neighbors_unique: an spdep::nb object (list of integer vectors
#     of neighbor indices, with 0L for no-neighbor entries).
#   - neighbor_source_vars: character vector of variable names.
# =============================================================================

library(data.table)

optimized_neighbor_features <- function(cell_data,
                                        id_order,
                                        rook_neighbors_unique,
                                        neighbor_source_vars) {

  # --- Step 0: Convert to data.table for speed (non-destructive) -----------
  dt <- as.data.table(cell_data)

  # --- Step 1: Build cell-level adjacency in CSR format --------------------
  # Map cell id -> reference index (position in id_order / nb object)
  n_cells <- length(id_order)
  id_to_ref <- integer(max(id_order))
  id_to_ref[id_order] <- seq_len(n_cells)
  # If id_order values are very large/sparse, use a hash instead:
  # id_to_ref_env <- new.env(hash = TRUE, size = n_cells)
  # for (k in seq_len(n_cells)) id_to_ref_env[[as.character(id_order[k])]] <- k

  # Clean the nb object: replace 0L (spdep's "no neighbor" code) with empty
  nb_clean <- lapply(rook_neighbors_unique, function(x) {
    x <- x[x != 0L]
    x
  })

  # Build CSR: adj_ids contains neighbor *cell IDs* (not ref indices),
  # adj_ptr[k]:(adj_ptr[k+1]-1) indexes into adj_ids for cell k (by ref index)
  adj_lengths <- vapply(nb_clean, length, integer(1))
  adj_ptr     <- c(1L, cumsum(adj_lengths) + 1L)  # length n_cells + 1
  adj_ref_ids <- unlist(nb_clean, use.names = FALSE)  # neighbor ref indices
  adj_cell_ids <- id_order[adj_ref_ids]               # neighbor cell IDs

  cat(sprintf("Adjacency CSR built: %d cells, %d directed edges\n",
              n_cells, length(adj_cell_ids)))

  # --- Step 2: Ensure data is keyed by (id, year) for fast join ------------
  setkey(dt, id, year)

  # Create a row-index column so we can map back
  dt[, .row_idx := .I]

  # --- Step 3: Build cell-ref to data-row mapping -------------------------
  # For each cell (by ref index) and each year, we need the row in dt.
  # We'll work year-by-year to keep memory bounded.

  years <- sort(unique(dt$year))
  n_years <- length(years)

  # Pre-allocate output columns
  for (var_name in neighbor_source_vars) {
    max_col <- paste0("neighbor_max_", var_name)
    min_col <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]
  }

  cat(sprintf("Processing %d years x %d variables...\n",
              n_years, length(neighbor_source_vars)))

  # --- Step 4: Year-by-year vectorized computation -------------------------
  for (yr in years) {

    # Subset rows for this year
    yr_rows <- dt[year == yr]
    n_yr <- nrow(yr_rows)
    if (n_yr == 0) next

    # Map cell_id -> position within this year's subset
    # (We need to look up neighbor values by cell_id within the same year)
    yr_cell_ids <- yr_rows$id
    yr_row_indices <- yr_rows$.row_idx  # original row positions in dt

    # Create a fast lookup: cell_id -> index in yr_rows
    # Using a pre-allocated vector indexed by cell_id (fast if IDs are dense)
    max_id <- max(yr_cell_ids)
    cellid_to_yrpos <- rep(NA_integer_, max_id)
    cellid_to_yrpos[yr_cell_ids] <- seq_len(n_yr)

    # For each cell present this year, find its ref index
    yr_ref_indices <- id_to_ref[yr_cell_ids]  # ref index for each row this year

    # For each row this year, gather neighbor values using CSR adjacency
    # We vectorize this by expanding the adjacency for all cells present this year

    # adj_start and adj_end for each row's cell
    a_start <- adj_ptr[yr_ref_indices]
    a_end   <- adj_ptr[yr_ref_indices + 1L] - 1L
    a_len   <- a_end - a_start + 1L
    a_len[a_len < 0L] <- 0L  # cells with no neighbors

    # Total number of (row, neighbor) pairs this year
    total_pairs <- sum(a_len)

    # Expand: for each row i (1..n_yr), repeat i a_len[i] times
    row_rep <- rep(seq_len(n_yr), times = a_len)

    # Gather the neighbor cell IDs for all rows
    # We need to index into adj_cell_ids using the CSR ranges
    # Build the flat index into adj_cell_ids
    seq_within <- sequence(a_len, from = a_start)
    neighbor_cids <- adj_cell_ids[seq_within]

    # Map neighbor cell IDs to their position in this year's data
    neighbor_yrpos <- cellid_to_yrpos[neighbor_cids]

    # Some neighbors may not be present this year (NA) — mark for removal
    valid <- !is.na(neighbor_yrpos)
    row_rep_v <- row_rep[valid]
    neighbor_yrpos_v <- neighbor_yrpos[valid]

    # Now compute stats for each variable
    for (var_name in neighbor_source_vars) {
      vals_yr <- yr_rows[[var_name]]

      # Gather neighbor values
      nvals <- vals_yr[neighbor_yrpos_v]

      # Remove NAs in the variable itself
      valid_val <- !is.na(nvals)
      rr <- row_rep_v[valid_val]
      nv <- nvals[valid_val]

      if (length(nv) == 0) next

      # Compute grouped max, min, sum, count using data.table
      tmp <- data.table(row = rr, val = nv)
      stats <- tmp[, .(
        nb_max  = max(val),
        nb_min  = min(val),
        nb_mean = mean(val)
      ), by = row]

      # Write results back into dt at the correct original row positions
      target_rows <- yr_row_indices[stats$row]

      max_col  <- paste0("neighbor_max_", var_name)
      min_col  <- paste0("neighbor_min_", var_name)
      mean_col <- paste0("neighbor_mean_", var_name)

      set(dt, i = target_rows, j = max_col,  value = stats$nb_max)
      set(dt, i = target_rows, j = min_col,  value = stats$nb_min)
      set(dt, i = target_rows, j = mean_col, value = stats$nb_mean)
    }

    if (yr %% 5 == 0 || yr == years[1] || yr == years[n_years]) {
      cat(sprintf("  Year %d done (%d rows, %d valid neighbor pairs)\n",
                  yr, n_yr, sum(valid)))
    }
  }

  # --- Step 5: Clean up and return -----------------------------------------
  dt[, .row_idx := NULL]

  cat("Neighbor feature construction complete.\n")
  return(dt)
}

# =============================================================================
# USAGE (drop-in replacement for the original outer loop)
# =============================================================================
# cell_data <- optimized_neighbor_features(
#   cell_data,
#   id_order,
#   rook_neighbors_unique,
#   neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# )
#
# # The trained Random Forest model is untouched — just use cell_data for
# # prediction as before. The output columns have identical names and values.
```

---

## Why This Is Correct and Equivalent

| Property | Original | Optimized |
|----------|----------|-----------|
| Neighbor definition | Rook neighbors from `spdep::nb` | Same `spdep::nb` object, same adjacency |
| Year matching | `paste(neighbor_id, year)` key ensures same-year lookup | Explicit year-by-year loop ensures same-year lookup |
| Missing handling | `!is.na(result)` drops neighbors absent in that year; `!is.na(neighbor_vals)` drops NA values | `!is.na(neighbor_yrpos)` drops absent neighbors; `!is.na(nvals)` drops NA values |
| Aggregation | `max`, `min`, `mean` of non-NA neighbor values | Identical `max`, `min`, `mean` via `data.table` grouped aggregation |
| Output columns | `neighbor_max_<var>`, `neighbor_min_<var>`, `neighbor_mean_<var>` | Identical column names and semantics |
| RF model | Untouched | Untouched — only the feature table is rebuilt |

## Performance Estimate

| Component | Original | Optimized |
|-----------|----------|-----------|
| String allocations | ~25.8M per variable | **Zero** |
| Named-vector lookups on 6.46M keys | ~25.8M | **Zero** (integer indexing) |
| R-level loop iterations | 6.46M × 6 (build + 5 vars) | **28** (one per year) |
| Memory pattern | 6.46M-element list of integer vectors | Flat integer vectors, year-chunked |
| **Estimated wall time** | **86+ hours** | **3–10 minutes** |

The speedup comes from three reinforcing changes: (1) eliminating all string operations, (2) replacing 6.46M R-level loop iterations with 28 vectorized year-passes, and (3) using `data.table` grouped aggregation instead of per-row `lapply`.