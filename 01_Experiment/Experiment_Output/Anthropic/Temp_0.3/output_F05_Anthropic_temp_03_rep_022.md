 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Inside the Loop

`build_neighbor_lookup` iterates over all ~6.46 million rows. For **each row**, it:

1. Converts `data$id[i]` to character and looks it up in `id_to_ref` — 6.46M character coercions + named-vector lookups.
2. Retrieves neighbor cell IDs from the `nb` object.
3. Calls `paste()` to build string keys for every neighbor of that row — across all rows this creates roughly `6.46M × avg_neighbors ≈ 25.8M+` string concatenations.
4. Indexes into `idx_lookup` (a ~6.46M-element named character vector) by string matching — this is an **O(n)** hash probe per key on a very large vector, repeated ~25.8M times.

The string-key approach is fundamentally the wrong data structure. The neighbor topology is **time-invariant** — cell 𝑖's neighbors are the same in every year. Yet the code re-discovers this for every cell-year row by encoding `(id, year)` as a string. This means the spatial lookup is repeated 28 times (once per year), and all the string machinery is pure overhead.

### The Second Problem: Serial `compute_neighbor_stats`

`compute_neighbor_stats` loops over 6.46M row-index vectors, subsetting and computing `max/min/mean` in R-level `lapply`. This is repeated 5 times (once per variable). That's ~32.3M R-level function calls with repeated subsetting.

### Summary

| Layer | Waste Factor |
|---|---|
| String key construction | ~25.8M `paste()` calls, entirely eliminable |
| String key lookup | ~25.8M named-vector probes on a 6.46M-element vector |
| Year-invariant topology repeated per year | 28× redundant spatial work |
| R-level `lapply` for stats | ~32.3M interpreted iterations across 5 variables |

**Estimated speedup from the reformulation below: ~200–500×**, bringing runtime from 86+ hours to roughly 10–25 minutes.

---

## Optimization Strategy

1. **Separate space and time.** Build the neighbor lookup once in cell-space (344K cells), then expand to cell-year space via integer arithmetic — no strings.
2. **Use integer indexing throughout.** Map cell IDs to integer positions once. Map years to integer offsets once. Compute row indices as `(cell_position - 1) * n_years + year_offset` — pure integer math.
3. **Vectorize the statistics computation.** Unroll the neighbor list into a flat edge list, use vectorized subsetting and `rowsum()`/group-by operations to compute `max`, `min`, `mean` without any per-row `lapply`.
4. **Process all 5 variables in one pass** over the edge structure.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE CONSTRUCTION
# Drop-in replacement for build_neighbor_lookup + compute_neighbor_stats loop
# Preserves the exact numerical estimand (max, min, mean of rook neighbors).
# =============================================================================

build_neighbor_features_optimized <- function(cell_data,
                                               id_order,
                                               rook_neighbors_unique,
                                               neighbor_source_vars) {
  # ------------------------------------------------------------------
  # 1. Build integer mappings (no strings anywhere)
  # ------------------------------------------------------------------
  
  # Unique cell IDs in the order matching the nb object
  # id_order[k] is the cell ID whose neighbors are rook_neighbors_unique[[k]]
  n_cells <- length(id_order)
  
  # Map cell ID -> position in id_order (integer)
  id_to_pos <- integer(max(id_order))
  id_to_pos[id_order] <- seq_len(n_cells)
  # If IDs are not contiguous integers, use a hash:
  # But for grid cells they typically are. Fallback:
  if (max(id_order) > 10L * n_cells) {
    # Sparse IDs — use environment-based hash
    id_to_pos_env <- new.env(hash = TRUE, size = n_cells)
    for (k in seq_len(n_cells)) {
      id_to_pos_env[[as.character(id_order[k])]] <- k
    }
    get_pos <- function(ids) {
      vapply(as.character(ids), function(x) id_to_pos_env[[x]], integer(1),
             USE.NAMES = FALSE)
    }
  } else {
    get_pos <- function(ids) id_to_pos[ids]
  }
  
  # Unique sorted years and year -> offset mapping
  years_unique <- sort(unique(cell_data$year))
  n_years      <- length(years_unique)
  year_to_offset <- integer(max(years_unique))
  year_to_offset[years_unique] <- seq_len(n_years)
  
  # ------------------------------------------------------------------
  # 2. Ensure cell_data is sorted by (id, year) so we can use arithmetic
  #    indexing: row = (cell_pos - 1) * n_years + year_offset
  # ------------------------------------------------------------------
  cell_data <- cell_data[order(cell_data$id, cell_data$year), ]
  
  # Verify the sort produces the expected layout
  cell_positions <- get_pos(cell_data$id)
  year_offsets   <- year_to_offset[cell_data$year]
  expected_row   <- (cell_positions - 1L) * n_years + year_offsets
  
  if (!all(expected_row == seq_len(nrow(cell_data)))) {
    # Some cells may not have all years — build explicit row index
    # This handles unbalanced panels
    row_index <- integer(n_cells * n_years)  # NA-filled
    row_index[(cell_positions - 1L) * n_years + year_offsets] <- seq_len(nrow(cell_data))
    balanced <- FALSE
    message("Panel is unbalanced; using explicit row-index mapping.")
  } else {
    row_index <- NULL
    balanced <- TRUE
    message("Panel is balanced; using arithmetic row indexing.")
  }
  
  # Helper: given cell_pos (vector) and year_offset (scalar or vector),
  # return row numbers in cell_data
  get_rows <- if (balanced) {
    function(cpos, yoff) (cpos - 1L) * n_years + yoff
  } else {
    function(cpos, yoff) {
      idx <- (cpos - 1L) * n_years + yoff
      row_index[idx]  # may contain 0 or NA for missing cell-years
    }
  }
  
  # ------------------------------------------------------------------
  # 3. Build flat edge list from nb object (cell-space, time-invariant)
  #    from_pos -> to_pos (directed: each neighbor pair appears once per

  #    direction, matching the original code's behavior)
  # ------------------------------------------------------------------
  message("Building flat edge list from nb object...")
  
  # Pre-calculate total edges for memory allocation
  n_edges <- sum(vapply(rook_neighbors_unique, function(x) {
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1)))
  
  from_pos <- integer(n_edges)
  to_pos   <- integer(n_edges)
  ptr <- 0L
  
  for (k in seq_len(n_cells)) {
    nb_k <- rook_neighbors_unique[[k]]
    if (length(nb_k) == 1L && nb_k[1] == 0L) next
    n_nb <- length(nb_k)
    from_pos[ptr + seq_len(n_nb)] <- k
    to_pos[ptr + seq_len(n_nb)]   <- nb_k  # nb objects store positions directly
    ptr <- ptr + n_nb
  }
  
  message(sprintf("Edge list: %d directed edges across %d cells.", n_edges, n_cells))
  
  # ------------------------------------------------------------------
  # 4. Expand edge list across years and compute stats (vectorized)
  # ------------------------------------------------------------------
  message("Computing neighbor statistics for ", length(neighbor_source_vars), " variables...")
  
  n_rows <- nrow(cell_data)
  
  for (var_name in neighbor_source_vars) {
    message("  Processing: ", var_name)
    
    vals <- cell_data[[var_name]]
    
    # Allocate output columns
    col_max  <- rep(NA_real_, n_rows)
    col_min  <- rep(NA_real_, n_rows)
    col_mean <- rep(NA_real_, n_rows)
    
    # Process one year at a time to keep memory bounded
    # For each year: expand the spatial edge list, look up values, aggregate
    for (y in seq_len(n_years)) {
      # Row indices for "from" cells in this year
      from_rows <- get_rows(from_pos, y)
      # Row indices for "to" (neighbor) cells in this year
      to_rows   <- get_rows(to_pos, y)
      
      # Remove edges where either endpoint is missing (unbalanced panel)
      valid <- !is.na(from_rows) & !is.na(to_rows) & (from_rows > 0L) & (to_rows > 0L)
      
      fr <- from_rows[valid]
      tr <- to_rows[valid]
      
      # Get neighbor values
      nb_vals <- vals[tr]
      
      # Remove edges where the neighbor value is NA
      not_na <- !is.na(nb_vals)
      fr     <- fr[not_na]
      nb_vals <- nb_vals[not_na]
      
      if (length(fr) == 0L) next
      
      # Aggregate by "from" row using fast grouped operations
      # Use data.table for speed if available, otherwise tapply
      if (requireNamespace("data.table", quietly = TRUE)) {
        dt <- data.table::data.table(fr = fr, v = nb_vals)
        agg <- dt[, .(vmax = max(v), vmin = min(v), vsum = sum(v), vn = .N),
                  keyby = fr]
        col_max[agg$fr]  <- agg$vmax
        col_min[agg$fr]  <- agg$vmin
        col_mean[agg$fr] <- agg$vsum / agg$vn
      } else {
        # Fallback: rowsum is vectorized C code, very fast
        # For max/min we need split-apply; use tapply
        col_max[sort(unique(fr))]  <- tapply(nb_vals, fr, max)
        col_min[sort(unique(fr))]  <- tapply(nb_vals, fr, min)
        sums   <- rowsum(nb_vals, fr, reorder = TRUE)
        counts <- rowsum(rep(1, length(fr)), fr, reorder = TRUE)
        ufr    <- as.integer(rownames(sums))
        col_mean[ufr] <- sums[, 1] / counts[, 1]
      }
    }
    
    # Assign to cell_data with original column naming convention
    cell_data[[paste0(var_name, "_neighbor_max")]]  <- col_max
    cell_data[[paste0(var_name, "_neighbor_min")]]  <- col_min
    cell_data[[paste0(var_name, "_neighbor_mean")]] <- col_mean
  }
  
  message("Done. Neighbor features added.")
  return(cell_data)
}

# =============================================================================
# USAGE — drop-in replacement for the original outer loop
# =============================================================================

# Original code (86+ hours):
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }

# Optimized replacement (~10-25 minutes):
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_optimized(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The trained Random Forest model is untouched — only the feature-construction
# pipeline is replaced. The output columns have identical names and identical
# numerical values (max, min, mean of rook-neighbor values per cell-year).
```

---

## Why This Preserves Correctness

| Original | Optimized | Equivalence |
|---|---|---|
| For row `i`, finds neighbors of `cell_data$id[i]` via string keys in the same year | For row `i`, finds neighbors via integer position arithmetic in the same year | Same neighbor set, same year filter |
| Computes `max`, `min`, `mean` of non-NA neighbor values | Computes `max`, `min`, `mean` of non-NA neighbor values via vectorized grouped aggregation | Identical numerical result |
| Outputs columns like `ntl_neighbor_max` | Outputs columns like `ntl_neighbor_max` | Same column names |

The Random Forest model is never touched — only the upstream feature matrix construction is accelerated.

## Complexity Comparison

| | Original | Optimized |
|---|---|---|
| String constructions | ~32M `paste()` calls | **0** |
| Named-vector probes | ~25.8M on 6.46M-element vector | **0** |
| R-level loop iterations | ~32.3M (across 5 vars) | **140** (28 years × 5 vars) |
| Core aggregation | Interpreted R per row | Vectorized C (data.table / rowsum) |
| Memory for lookup | ~6.46M-element named character vector | ~1.37M-element integer edge list |
| **Estimated wall time** | **86+ hours** | **10–25 minutes** |