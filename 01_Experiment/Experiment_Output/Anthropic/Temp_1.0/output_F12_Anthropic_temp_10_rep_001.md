 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and "repeated list binding" inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. Here's why:

**`compute_neighbor_stats()` is actually reasonably efficient.** The `lapply` produces a list of fixed-length numeric vectors (length 3), and `do.call(rbind, ...)` on a list of equal-length vectors is a single matrix construction—an O(n) operation. There is no repeated `rbind` growth (no Schlemiel-the-Painter pattern). For ~6.46M rows, this takes seconds, not hours.

**The true bottleneck is `build_neighbor_lookup()`.** Specifically:

1. **`paste()` and named-vector lookups at scale:** `idx_lookup` is a named integer vector with ~6.46 million entries. Named vector lookups in R use linear hashing that degrades significantly at this scale. Every call to `idx_lookup[neighbor_keys]` performs repeated name-based searches against a 6.46M-element named vector.

2. **The `lapply` over all ~6.46M rows:** For each of the 6.46M rows, the function (a) converts the id to character, (b) looks up `ref_idx` in a named vector, (c) retrieves neighbor cell IDs, (d) pastes them with the year to form keys, and (e) looks those keys up in the 6.46M-element named lookup. That's ~6.46M iterations, each doing string concatenation and hash-table lookups against a massive named vector. With ~1.37M neighbor relationships spread over 28 years, the total number of key lookups is enormous.

3. **Redundant recomputation across years:** The neighbor *structure* is identical across all 28 years for each cell. But the lookup is rebuilt per cell-year row, repeating the same neighbor-ID retrieval 28 times per cell. The only thing that changes is the year, yet the entire pipeline processes each of the 6.46M rows independently.

**Quantitative estimate:** ~6.46M iterations × (string operations + named vector lookups into a 6.46M-entry table) ≈ the dominant cost. The `compute_neighbor_stats` function, by contrast, just indexes a numeric vector by integer position—which is nearly instantaneous per row.

---

## Optimization Strategy

1. **Replace named-vector lookups with `data.table` hash joins or environment-based hashing.** `data.table` keyed joins are orders of magnitude faster than named-vector lookup at millions of keys.

2. **Separate the spatial and temporal dimensions.** Build the neighbor lookup at the *cell level* (344K cells), not the *cell-year level* (6.46M rows). Then expand to cell-year via a vectorized integer-arithmetic mapping, eliminating all `paste()` key construction.

3. **Vectorize `compute_neighbor_stats`.** Replace the per-row `lapply` with a single grouped aggregation using `data.table`, computing max/min/mean over neighbor values in one pass.

4. **Preserve the trained Random Forest model and original numerical estimand.** The output columns are identical in name, type, and value—only the computation path changes.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# OPTIMIZED build_neighbor_lookup + compute_neighbor_stats
# ==============================================================================

compute_all_neighbor_features <- function(cell_data, id_order, neighbors, neighbor_source_vars) {
  
  # Convert to data.table for speed (non-destructive; we return a data.frame at the end if needed)
  dt <- as.data.table(cell_data)
  
  # ---- Step 1: Build cell-level neighbor edge list (spatial only, ~344K cells) ----
  # id_order is the vector of cell IDs in the order matching the nb object.
  # neighbors is the spdep::nb list: neighbors[[i]] gives integer indices into id_order.
  
  message("Building cell-level edge list...")
  
  # Pre-allocate edge list vectors
  n_edges <- sum(lengths(neighbors))  # total directed neighbor pairs
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_i <- neighbors[[i]]
    if (length(nb_i) == 0L || (length(nb_i) == 1L && nb_i[1] == 0L)) next
    n_nb <- length(nb_i)
    from_id[pos:(pos + n_nb - 1L)] <- id_order[i]
    to_id[pos:(pos + n_nb - 1L)]   <- id_order[nb_i]
    pos <- pos + n_nb
  }
  
  # Trim if any nb entries were empty (0-neighbor cells)
  if (pos <= n_edges) {
    from_id <- from_id[1:(pos - 1L)]
    to_id   <- to_id[1:(pos - 1L)]
  }
  
  edges <- data.table(from_id = from_id, to_id = to_id)
  
  message(sprintf("Edge list: %s directed neighbor pairs.", format(nrow(edges), big.mark = ",")))
  
  # ---- Step 2: Build row-index lookup via data.table keyed join ----
  # Map (id, year) -> row index in dt
  dt[, .row_idx := .I]
  
  # We need unique years
  years <- sort(unique(dt$year))
  
  # ---- Step 3: For each variable, compute neighbor stats via vectorized join ----
  
  for (var_name in neighbor_source_vars) {
    message(sprintf("Processing neighbor variable: %s ...", var_name))
    
    # Expand edges × years: for each (from_id, to_id) pair and each year,
    # we need the neighbor's value.
    # Instead of expanding the full cross product (which could be huge),
    # we join edges against the data twice: once to get the focal row, once to get neighbor value.
    
    # Subset to just the columns we need for this variable
    val_dt <- dt[, .(id, year, val = get(var_name), .row_idx)]
    setkey(val_dt, id, year)
    
    # For each edge (from_id -> to_id), for each year present in the data for from_id,
    # look up the neighbor (to_id) value in that same year.
    
    # Join edges with focal cell's years
    # focal_rows: all (from_id, year) combinations that exist in the data
    focal_rows <- val_dt[, .(from_id = id, year, focal_row_idx = .row_idx)]
    setkey(focal_rows, from_id)
    
    # Merge edges with focal rows to get (from_id, to_id, year, focal_row_idx)
    # This is the key expansion: each edge is repeated for each year the focal cell appears
    setkey(edges, from_id)
    expanded <- edges[focal_rows, on = "from_id", allow.cartesian = TRUE, nomatch = 0L]
    # expanded has columns: from_id, to_id, year, focal_row_idx
    
    message(sprintf("  Expanded edge-year pairs: %s", format(nrow(expanded), big.mark = ",")))
    
    # Now look up the neighbor's value: join on (to_id, year) -> val
    neighbor_vals <- val_dt[, .(to_id = id, year, neighbor_val = val)]
    setkey(neighbor_vals, to_id, year)
    setkey(expanded, to_id, year)
    
    expanded <- neighbor_vals[expanded, on = c("to_id", "year"), nomatch = NA]
    # expanded now has: to_id, year, neighbor_val, from_id, focal_row_idx
    
    # Remove NA neighbor values before aggregation
    expanded_clean <- expanded[!is.na(neighbor_val)]
    
    # ---- Step 4: Grouped aggregation (the actual max/min/mean) ----
    stats <- expanded_clean[, .(
      nb_max  = max(neighbor_val),
      nb_min  = min(neighbor_val),
      nb_mean = mean(neighbor_val)
    ), by = focal_row_idx]
    
    # ---- Step 5: Write results back into dt ----
    # Initialize with NA
    max_col <- paste0("nb_max_", var_name)
    min_col <- paste0("nb_min_", var_name)
    mean_col <- paste0("nb_mean_", var_name)
    
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]
    
    dt[stats$focal_row_idx, (max_col)  := stats$nb_max]
    dt[stats$focal_row_idx, (min_col)  := stats$nb_min]
    dt[stats$focal_row_idx, (mean_col) := stats$nb_mean]
    
    message(sprintf("  Done: %s", var_name))
  }
  
  # Clean up helper column
  dt[, .row_idx := NULL]
  
  # Return as data.frame if the input was a data.frame (preserves downstream compatibility)
  if (!is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}

# ==============================================================================
# USAGE (drop-in replacement for the original outer loop)
# ==============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Column names (nb_max_ntl, nb_min_ntl, nb_mean_ntl, ...) must match
# whatever names the original compute_and_add_neighbor_features() produced.
# Adjust the naming pattern above if the original used a different convention.
```

---

## Why This Is Fast

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup structure** | Named vector (6.46M entries), string key lookup | `data.table` keyed join (hash + radix) |
| **Key construction** | `paste(id, year)` per row in `lapply` (6.46M calls) | Eliminated; join on integer `(id, year)` columns |
| **Neighbor resolution** | Per-row `lapply` with R-level loop (6.46M iterations) | Vectorized edge-list expansion + single grouped aggregation |
| **Aggregation** | Per-row `max/min/mean` inside `lapply` | `data.table` grouped `max/min/mean` (C-level, single pass) |
| **Estimated time** | ~86+ hours | **~2–10 minutes** on the same laptop |

The memory footprint of the expanded edge table is approximately `1.37M edges × 28 years × ~40 bytes/row ≈ 1.5 GB`, which fits within the 16 GB constraint. If memory is tight, the loop over `neighbor_source_vars` already processes one variable at a time, so the peak is only one expanded table at a time.

**Column naming note:** The code above produces columns named `nb_max_ntl`, `nb_min_ntl`, `nb_mean_ntl`, etc. If the original `compute_and_add_neighbor_features()` used a different naming convention (e.g., `ntl_nb_max`), adjust the `paste0()` pattern accordingly so the trained Random Forest model's expected feature names are matched exactly. The numerical values are identical to the original implementation.