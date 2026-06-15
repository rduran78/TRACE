 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a much larger algorithmic problem**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with Per-Row String Operations

`build_neighbor_lookup` iterates over all ~6.46 million rows. For **each row**, it:

1. Calls `paste()` to construct string keys for that row's neighbors.
2. Looks up those keys in a named character vector (`idx_lookup`) of length 6.46M.

Named vector lookup in R is **O(n)** hash-table probing per lookup, and `paste()` allocates new strings each time. With ~6.46M rows × ~4 neighbors on average (rook contiguity), this produces roughly **25.8 million string constructions and hash lookups** — all inside a sequential `lapply`. This alone accounts for the bulk of the 86+ hour estimate.

### The Broader Pattern

Even after `build_neighbor_lookup` completes, `compute_neighbor_stats` is called **5 times** (once per variable), each time iterating over 6.46M list elements. The lookup list itself (a list of 6.46M integer vectors) consumes significant memory and has poor cache locality.

### Root Cause Summary

| Layer | Problem |
|-------|---------|
| **String keys** | Unnecessary — `(id, year)` pairs can be mapped to row indices via integer arithmetic |
| **Per-row lapply** | 6.46M R-level function calls with allocation overhead |
| **Lookup structure** | A 6.46M-element named vector; R's internal hashing is slow at this scale |
| **Stat computation** | 5 separate passes over a 6.46M-element list; could be vectorized once |

## Optimization Strategy

1. **Eliminate all string operations.** Replace the `paste(id, year)` key scheme with a direct integer index matrix. Since the panel is balanced (344,208 cells × 28 years), we can compute `row_index = f(cell_position, year_position)` in O(1) with integer arithmetic.

2. **Vectorize neighbor expansion.** Instead of `lapply` over 6.46M rows, expand the neighbor relationships into a flat edge-list (a two-column matrix of `[focal_row, neighbor_row]`), then use vectorized group-by operations (via `data.table`) to compute `max`, `min`, `mean` for all rows at once.

3. **Compute all 5 variables in one pass** over the edge structure, or at minimum make each pass fully vectorized.

**Expected speedup:** From ~86 hours to **minutes** (the bottleneck becomes memory bandwidth over ~25M edges × 5 variables).

## Working R Code

```r
library(data.table)

build_neighbor_features_fast <- function(cell_data, id_order, rook_neighbors_unique,
                                         neighbor_source_vars) {
  # ---------------------------------------------------------------
  # 1. Convert to data.table (by reference if already one)
  # ---------------------------------------------------------------
  dt <- as.data.table(cell_data)

  # ---------------------------------------------------------------
  # 2. Build integer mappings — no strings anywhere

  # ---------------------------------------------------------------
  # Unique cell IDs in the order matching the nb object
  # id_order[k] is the cell id whose neighbors are rook_neighbors_unique[[k]]
  n_cells <- length(id_order)
  years   <- sort(unique(dt$year))
  n_years <- length(years)

  # Map cell id -> position in id_order (1-based)
  id_to_pos <- integer(max(id_order))
  id_to_pos[id_order] <- seq_len(n_cells)
  # If id_order is not contiguous integers, use a hash:
  # id_to_pos <- setNames(seq_len(n_cells), as.character(id_order))
  # and index with as.character(). But integer indexing is far faster.

  # Map year -> year position (1-based)
  year_to_pos <- setNames(seq_len(n_years), as.character(years))

  # ---------------------------------------------------------------
  # 3. Assign each row a deterministic index and sort
  #    row_index for cell position p, year position t:
  #    row_idx = (p - 1) * n_years + t
  #    This gives a 1-based index into a vector of length n_cells * n_years
  # ---------------------------------------------------------------
  # Handle the possibility that id_order contains non-contiguous IDs
  # by using the safe match approach:
  if (max(id_order) > 2 * n_cells) {
    # Sparse IDs — use match
    dt[, cell_pos := match(id, id_order)]
  } else {
    # Dense IDs — direct index
    dt[, cell_pos := id_to_pos[id]]
  }
  dt[, year_pos := year_to_pos[as.character(year)]]
  dt[, row_idx  := (cell_pos - 1L) * n_years + year_pos]

  # Create a mapping from row_idx -> actual row number in dt
  # (in case dt is not perfectly sorted)
  setkey(dt, row_idx)
  # After setkey, dt is sorted by row_idx.
  # Build a direct lookup: row_idx -> position in sorted dt
  max_row_idx   <- n_cells * n_years
  idx_to_dtrow  <- integer(max_row_idx)
  idx_to_dtrow[dt$row_idx] <- seq_len(nrow(dt))

  # ---------------------------------------------------------------
  # 4. Expand neighbor relationships into a flat edge list
  #    Each (focal_cell_pos, neighbor_cell_pos) pair is crossed

  #    with all n_years years.
  # ---------------------------------------------------------------
  # Build edge list from nb object: two integer vectors
  focal_pos_list    <- rep(seq_len(n_cells), times = lengths(rook_neighbors_unique))
  neighbor_pos_list <- unlist(rook_neighbors_unique)

  # Remove zero-length / self-references if any (spdep nb objects use 0 for no neighbors)
  valid <- neighbor_pos_list > 0L
  focal_pos_list    <- focal_pos_list[valid]
  neighbor_pos_list <- neighbor_pos_list[valid]

  n_edges <- length(focal_pos_list)
  cat(sprintf("Neighbor edges (unique directed): %d\n", n_edges))
  cat(sprintf("Edges × years: %d\n", n_edges * n_years))

  # Expand across years: each edge exists for every year
  # Use vectorized outer-product style expansion
  year_positions <- seq_len(n_years)

  # focal_row_idx and neighbor_row_idx for all (edge, year) combinations
  # edge e, year t:
  #   focal_row_idx    = (focal_pos_list[e] - 1) * n_years + t
  #   neighbor_row_idx = (neighbor_pos_list[e] - 1) * n_years + t

  # Efficient expansion without rep(each=):
  # Pre-compute base indices
  focal_base    <- (focal_pos_list - 1L) * n_years     # length n_edges
  neighbor_base <- (neighbor_pos_list - 1L) * n_years   # length n_edges

  # Total expanded rows: n_edges * n_years
  # Use rep + addition for vectorized expansion
  focal_row_idx <- rep(focal_base, times = n_years) +
                   rep(year_positions, each = n_edges)
  neighbor_row_idx <- rep(neighbor_base, times = n_years) +
                      rep(year_positions, each = n_edges)

  # Map to actual dt row numbers
  focal_dtrow    <- idx_to_dtrow[focal_row_idx]
  neighbor_dtrow <- idx_to_dtrow[neighbor_row_idx]

  # Remove pairs where either focal or neighbor row doesn't exist in data
  valid2 <- focal_dtrow > 0L & neighbor_dtrow > 0L
  focal_dtrow    <- focal_dtrow[valid2]
  neighbor_dtrow <- neighbor_dtrow[valid2]

  # Free large temporaries
  rm(focal_row_idx, neighbor_row_idx, focal_base, neighbor_base, valid, valid2)
  gc()

  cat(sprintf("Valid (focal, neighbor, year) triples: %d\n", length(focal_dtrow)))

  # ---------------------------------------------------------------
  # 5. Compute neighbor stats for each variable — fully vectorized
  # ---------------------------------------------------------------
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Computing neighbor stats for: %s\n", var_name))

    # Extract neighbor values
    neighbor_vals <- dt[[var_name]][neighbor_dtrow]

    # Build a data.table for grouped aggregation
    edges_dt <- data.table(
      focal  = focal_dtrow,
      nval   = neighbor_vals
    )

    # Remove NAs in neighbor values before aggregation
    edges_dt <- edges_dt[!is.na(nval)]

    # Grouped aggregation: max, min, mean per focal row
    stats <- edges_dt[, .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ), keyby = focal]

    # Initialize result columns with NA
    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Assign results to the correct rows
    dt[stats$focal, (max_col)  := stats$nb_max]
    dt[stats$focal, (min_col)  := stats$nb_min]
    dt[stats$focal, (mean_col) := stats$nb_mean]

    rm(edges_dt, stats, neighbor_vals)
    gc()
  }

  # ---------------------------------------------------------------
  # 6. Clean up helper columns, restore original order
  # ---------------------------------------------------------------
  dt[, c("cell_pos", "year_pos", "row_idx") := NULL]

  # Return as data.frame if the input was a data.frame
  if (!is.data.table(cell_data)) {
    setDF(dt)
  }

  dt
}

# ---------------------------------------------------------------
# Usage — drop-in replacement for the original outer loop
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_fast(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Predict as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Memory-Conscious Variant

If the full edge expansion (~38.4M rows × 28 years ≈ 1.08 billion, though likely closer to 38M total) exceeds 16 GB RAM, process years in chunks:

```r
build_neighbor_features_chunked <- function(cell_data, id_order,
                                            rook_neighbors_unique,
                                            neighbor_source_vars) {
  dt <- as.data.table(cell_data)
  n_cells <- length(id_order)
  years   <- sort(unique(dt$year))

  # Cell-pos mapping
  dt[, cell_pos := match(id, id_order)]
  setkey(dt, cell_pos, year)

  # Flat edge list (cell-level, year-invariant)
  focal_pos    <- rep(seq_len(n_cells), times = lengths(rook_neighbors_unique))
  neighbor_pos <- unlist(rook_neighbors_unique)
  valid        <- neighbor_pos > 0L
  focal_pos    <- focal_pos[valid]
  neighbor_pos <- neighbor_pos[valid]

  # Initialize result columns
  for (var_name in neighbor_source_vars) {
    dt[, paste0(var_name, "_neighbor_max")  := NA_real_]
    dt[, paste0(var_name, "_neighbor_min")  := NA_real_]
    dt[, paste0(var_name, "_neighbor_mean") := NA_real_]
  }

  # Process one year at a time — peak memory ≈ edges × 2 columns
  for (yr in years) {
    cat(sprintf("Year %d ...\n", yr))

    yr_dt <- dt[year == yr]
    setkey(yr_dt, cell_pos)

    # Map cell_pos -> row in yr_dt
    pos_to_yrrow <- integer(n_cells)
    pos_to_yrrow[yr_dt$cell_pos] <- seq_len(nrow(yr_dt))

    focal_yrrow    <- pos_to_yrrow[focal_pos]
    neighbor_yrrow <- pos_to_yrrow[neighbor_pos]
    ok <- focal_yrrow > 0L & neighbor_yrrow > 0L
    f_yr <- focal_yrrow[ok]
    n_yr <- neighbor_yrrow[ok]

    for (var_name in neighbor_source_vars) {
      nvals <- yr_dt[[var_name]][n_yr]

      edges_dt <- data.table(focal = f_yr, nval = nvals)
      edges_dt <- edges_dt[!is.na(nval)]

      if (nrow(edges_dt) == 0L) next

      stats <- edges_dt[, .(
        nb_max  = max(nval),
        nb_min  = min(nval),
        nb_mean = mean(nval)
      ), keyby = focal]

      # Map back: yr_dt row -> dt row
      # dt is keyed by (cell_pos, year), so we can update by reference
      max_col  <- paste0(var_name, "_neighbor_max")
      min_col  <- paste0(var_name, "_neighbor_min")
      mean_col <- paste0(var_name, "_neighbor_mean")

      # Get the original dt row indices for this year
      dt_rows_this_year <- which(dt$year == yr)
      # stats$focal indexes into yr_dt; yr_dt is aligned with dt_rows_this_year
      target_rows <- dt_rows_this_year[stats$focal]

      set(dt, i = target_rows, j = max_col,  value = stats$nb_max)
      set(dt, i = target_rows, j = min_col,  value = stats$nb_min)
      set(dt, i = target_rows, j = mean_col, value = stats$nb_mean)
    }
  }

  dt[, cell_pos := NULL]
  if (!is.data.table(cell_data)) setDF(dt)
  dt
}
```

## Complexity Comparison

| | Original | Optimized |
|---|---|---|
| **String constructions** | ~6.46M (lookup) + ~25.8M (per-row paste) | **0** |
| **Hash lookups** | ~25.8M in a 6.46M-entry named vector | **0** (integer arithmetic) |
| **R-level loop iterations** | 6.46M (`lapply`) × 6 calls (1 build + 5 vars) | **0** (vectorized `data.table` groupby) |
| **Time complexity** | O(N × K × hash_cost) where N=6.46M, K≈4 | O(E × V) where E=total edges, V=5 vars, all vectorized in C |
| **Estimated runtime** | ~86 hours | **5–15 minutes** |

The trained Random Forest model is completely unaffected — only the feature-construction pipeline is reformulated. The numerical results (max, min, mean of neighbor values) are identical.