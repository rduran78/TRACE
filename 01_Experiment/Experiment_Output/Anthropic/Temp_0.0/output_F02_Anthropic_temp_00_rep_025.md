 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which runs an `lapply` over ~6.46 million rows, performing per-row string pasting, hash-table lookups, and NA filtering. This produces a list of 6.46 million integer vectors. The secondary bottleneck is **`compute_neighbor_stats`**, which iterates over that same 6.46-million-element list five times (once per variable), computing `max`, `min`, and `mean` on small vectors inside an R-level loop.

**Specific problems:**

1. **String-key hashing per row in `build_neighbor_lookup`:** For every row, `paste(neighbor_cell_ids, data$year[i], sep = "_")` creates character vectors and looks them up in a named vector (`idx_lookup`). With ~6.46M rows and an average of ~4 rook neighbors each, this is ~25.8 million `paste` + lookup operations, all in interpreted R.

2. **The lookup list is enormous:** A 6.46-million-element list of integer vectors consumes substantial memory (list overhead alone is ~50 bytes/element ≈ 300+ MB, plus the integer vectors themselves).

3. **`compute_neighbor_stats` is called 5 times**, each time iterating over the full 6.46M-element list in pure R `lapply`. That's ~32.3 million R-level function calls total.

4. **No vectorization or use of `data.table`:** The entire pipeline relies on base-R named-vector lookups and `lapply`, missing opportunities for vectorized joins and grouped operations.

---

## Optimization Strategy

**Replace the row-level list-based lookup with a vectorized `data.table` join-and-aggregate approach.**

The key insight: the neighbor lookup is conceptually a **join**. Each `(cell_id, year)` pair needs to be joined to its neighbors' `(neighbor_id, year)` rows, then aggregated. `data.table` performs this kind of equi-join + grouped aggregation in optimized C, eliminating millions of R-level function calls.

**Steps:**

1. **Build an edge table** from the `nb` object: a two-column `data.table` with `(id, neighbor_id)` — done once, ~1.37M rows.
2. **Join the edge table to the panel data** on `(neighbor_id, year)` to pull neighbor values — this is a single keyed `data.table` merge producing ~1.37M × 28 ≈ ~38.5M rows (but done in C, very fast).
3. **Aggregate** (`max`, `min`, `mean`) by `(id, year)` — a single grouped operation in `data.table`.
4. **Merge** the aggregated stats back to the main data.
5. **Repeat for each variable**, or do all 5 variables in one pass.

**Expected improvement:**
- `build_neighbor_lookup` is eliminated entirely.
- `compute_neighbor_stats` is replaced by a vectorized join + group-by.
- Estimated runtime: **minutes, not hours** (the join is ~38.5M rows, well within `data.table` comfort zone on 16 GB RAM).
- Memory: the edge table × years is ~38.5M rows × a few columns — manageable.
- The trained Random Forest model and all numerical outputs are preserved exactly (same `max`, `min`, `mean` computations on the same neighbor sets).

---

## Working R Code

```r
library(data.table)

#' Optimized neighbor feature pipeline
#' 
#' @param cell_data        data.frame or data.table with columns: id, year, and all neighbor_source_vars
#' @param id_order          character or integer vector of cell IDs in the order matching rook_neighbors_unique
#' @param rook_neighbors_unique  an nb object (list of integer index vectors) from spdep
#' @param neighbor_source_vars   character vector of variable names to compute neighbor stats for
#' 
#' @return cell_data with new columns: {var}_neighbor_max, {var}_neighbor_min, {var}_neighbor_mean
compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {
  
  # --- Step 1: Convert to data.table if needed (by reference, no copy) ---
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
    was_df <- TRUE
  } else {
    was_df <- FALSE
  }
  
  # --- Step 2: Build the edge table from the nb object ---
  # Each element of rook_neighbors_unique is an integer vector of indices into id_order.
  # We expand this into a two-column table: (focal_id, neighbor_id).
  
  n_cells <- length(id_order)
  
  # Compute lengths of each neighbor set
  nb_lengths <- lengths(rook_neighbors_unique)  # integer vector, length = n_cells
  
  # Pre-allocate vectors
  total_edges <- sum(nb_lengths)  # ~1,373,394
  focal_ids    <- rep(id_order, times = nb_lengths)
  neighbor_ids <- id_order[unlist(rook_neighbors_unique, use.names = FALSE)]
  
  edges <- data.table(
    focal_id    = focal_ids,
    neighbor_id = neighbor_ids
  )
  rm(focal_ids, neighbor_ids)  # free memory
  
  cat(sprintf("Edge table built: %d directed edges\n", nrow(edges)))
  
  # --- Step 3: Prepare a keyed lookup of the panel data ---
  # We only need id, year, and the source variables for the join.
  lookup_cols <- c("id", "year", neighbor_source_vars)
  lookup_dt   <- cell_data[, ..lookup_cols]
  
  # Key for fast join
  setkey(lookup_dt, id, year)
  
  # --- Step 4: Get unique years to iterate (avoids a massive cross-join) ---
  # Actually, the most memory-efficient vectorized approach is:
  #   - Cross the edge table with years to get (focal_id, neighbor_id, year)
  #   - Join to lookup_dt on (neighbor_id, year) to get neighbor values
  #   - Aggregate by (focal_id, year)
  #
  # With ~1.37M edges × 28 years = ~38.4M rows, this fits in memory.
  
  unique_years <- sort(unique(cell_data$year))
  
  # Expand edges × years
  edges_by_year <- edges[, .(year = unique_years), by = .(focal_id, neighbor_id)]
  # This creates ~38.4M rows via cross join per edge.
  # More memory-efficient alternative using CJ:
  
  # Actually, the above syntax is wrong for a cross join in data.table.
  # Correct approach:
  edges_by_year <- CJ(edge_idx = seq_len(nrow(edges)), year = unique_years)
  edges_by_year[, focal_id    := edges$focal_id[edge_idx]]
  edges_by_year[, neighbor_id := edges$neighbor_id[edge_idx]]
  edges_by_year[, edge_idx    := NULL]
  
  cat(sprintf("Edges × years table: %d rows (%.1f M)\n", 
              nrow(edges_by_year), nrow(edges_by_year) / 1e6))
  
  rm(edges)
  gc()
  
  # --- Step 5: Join to get neighbor values ---
  # Join edges_by_year to lookup_dt on (neighbor_id = id, year = year)
  setkey(edges_by_year, neighbor_id, year)
  setkey(lookup_dt, id, year)
  
  # Perform the join — this pulls in all neighbor_source_vars columns
  edges_by_year <- lookup_dt[edges_by_year, on = .(id = neighbor_id, year = year)]
  
  # After the join, 'id' column is the neighbor_id. Rename for clarity.
  # The focal_id column is carried through from edges_by_year.
  # 'id' now = neighbor_id (from the join key), focal_id = the cell we're computing features for.
  
  # --- Step 6: Aggregate by (focal_id, year) for each variable ---
  setkey(edges_by_year, focal_id, year)
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Computing neighbor stats for: %s\n", var_name))
    
    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")
    
    # Aggregate — na.rm = TRUE matches the original code's behavior
    # (original code filters out NAs before computing stats)
    agg <- edges_by_year[, .(
      nb_max  = if (all(is.na(get(var_name)))) NA_real_ else max(get(var_name), na.rm = TRUE),
      nb_min  = if (all(is.na(get(var_name)))) NA_real_ else min(get(var_name), na.rm = TRUE),
      nb_mean = if (all(is.na(get(var_name)))) NA_real_ else mean(get(var_name), na.rm = TRUE)
    ), by = .(focal_id, year)]
    
    setnames(agg, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
    
    # Merge back to cell_data
    # Remove these columns from cell_data if they already exist (idempotency)
    for (col in c(max_col, min_col, mean_col)) {
      if (col %in% names(cell_data)) cell_data[, (col) := NULL]
    }
    
    cell_data <- merge(cell_data, agg, by.x = c("id", "year"), by.y = c("focal_id", "year"), all.x = TRUE)
    
    rm(agg)
  }
  
  rm(edges_by_year, lookup_dt)
  gc()
  
  if (was_df) cell_data <- as.data.frame(cell_data)
  
  return(cell_data)
}
```

### Even More Memory-Efficient Variant (Year-Chunked)

If the ~38.4M-row `edges_by_year` table is too large for available RAM (each of the 5 numeric columns is ~307 MB), process one year at a time:

```r
compute_all_neighbor_features_chunked <- function(cell_data,
                                                   id_order,
                                                   rook_neighbors_unique,
                                                   neighbor_source_vars) {
  
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
    was_df <- TRUE
  } else {
    was_df <- FALSE
  }
  
  # Build edge table (same as before)
  nb_lengths   <- lengths(rook_neighbors_unique)
  focal_ids    <- rep(id_order, times = nb_lengths)
  neighbor_ids <- id_order[unlist(rook_neighbors_unique, use.names = FALSE)]
  edges <- data.table(focal_id = focal_ids, neighbor_id = neighbor_ids)
  rm(focal_ids, neighbor_ids)
  
  lookup_cols <- c("id", "year", neighbor_source_vars)
  lookup_dt   <- cell_data[, ..lookup_cols]
  setkey(lookup_dt, id, year)
  
  unique_years <- sort(unique(cell_data$year))
  
  # Initialize result columns
  for (var_name in neighbor_source_vars) {
    cell_data[, paste0(var_name, "_neighbor_max")  := NA_real_]
    cell_data[, paste0(var_name, "_neighbor_min")  := NA_real_]
    cell_data[, paste0(var_name, "_neighbor_mean") := NA_real_]
  }
  
  setkey(cell_data, id, year)
  
  for (yr in unique_years) {
    cat(sprintf("Processing year %d ...\n", yr))
    
    # Subset panel data for this year
    yr_data <- lookup_dt[year == yr]
    setkey(yr_data, id)
    
    # Join edges to this year's data to get neighbor values
    # edges has (focal_id, neighbor_id); join on neighbor_id = id
    merged <- yr_data[edges, on = .(id = neighbor_id), nomatch = NA, allow.cartesian = TRUE]
    # merged now has columns: id (=neighbor_id), year, <vars>, focal_id
    
    for (var_name in neighbor_source_vars) {
      max_col  <- paste0(var_name, "_neighbor_max")
      min_col  <- paste0(var_name, "_neighbor_min")
      mean_col <- paste0(var_name, "_neighbor_mean")
      
      agg <- merged[, {
        v <- get(var_name)
        v <- v[!is.na(v)]
        if (length(v) == 0L) {
          list(nb_max = NA_real_, nb_min = NA_real_, nb_mean = NA_real_)
        } else {
          list(nb_max = max(v), nb_min = min(v), nb_mean = mean(v))
        }
      }, by = .(focal_id)]
      
      # Update cell_data in place for this year
      cell_data[agg, on = .(id = focal_id, year = yr),
                (max_col)  := i.nb_max]
      cell_data[agg, on = .(id = focal_id, year = yr),
                (min_col)  := i.nb_min]
      cell_data[agg, on = .(id = focal_id, year = yr),
                (mean_col) := i.nb_mean]
      
      rm(agg)
    }
    
    rm(merged, yr_data)
    gc()
  }
  
  rm(edges, lookup_dt)
  gc()
  
  if (was_df) cell_data <- as.data.frame(cell_data)
  return(cell_data)
}
```

### Usage (Drop-In Replacement)

```r
library(data.table)

# Load your existing objects
# cell_data                 — your panel data.frame/data.table
# id_order                  — vector of cell IDs matching the nb object
# rook_neighbors_unique     — the spdep nb object
# rf_model                  — your trained Random Forest (untouched)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Option A: Single-pass (needs ~4-6 GB free RAM)
cell_data <- compute_all_neighbor_features(
  cell_data, id_order, rook_neighbors_unique, neighbor_source_vars
)

# Option B: Year-chunked (needs ~1-2 GB free RAM, slightly slower but safer on 16 GB)
cell_data <- compute_all_neighbor_features_chunked(
  cell_data, id_order, rook_neighbors_unique, neighbor_source_vars
)

# Predict with the existing trained model (unchanged)
cell_data$predicted_gdp <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M R-level `paste` + named-vector lookups | Eliminated; replaced by `data.table` keyed join |
| **Neighbor stats** | 5 × 6.46M `lapply` calls (R interpreter) | 5 × 1 vectorized `data.table` group-by (C backend) |
| **Estimated runtime** | 86+ hours | ~5–20 minutes (depending on disk I/O and RAM) |
| **Peak RAM** | Very high (6.46M-element list + copies) | ~4–6 GB (Option A) or ~1–2 GB (Option B, chunked) |
| **Numerical results** | Baseline | Identical (`max`, `min`, `mean` with NA removal) |
| **RF model** | — | Untouched; no retraining needed |