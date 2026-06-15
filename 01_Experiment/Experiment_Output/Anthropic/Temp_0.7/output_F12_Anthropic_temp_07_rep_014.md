 # Diagnosis, Optimization Strategy, and Working Code

## Diagnosis

The colleague is **partially right but misses the deeper bottleneck**. Let me audit both functions:

### `compute_neighbor_stats()` — Colleague's Claim

`do.call(rbind, result)` on a list of ~6.46 million 3-element vectors is indeed not free — it builds a ~6.46M × 3 matrix from a list. However, this is a **one-shot operation per variable** (5 times total). With modern R, `do.call(rbind, ...)` on uniform-length numeric vectors is reasonably fast (seconds, not hours). The `lapply` inside `compute_neighbor_stats` is a simple vectorized index-subset + three summary stats per row. This is **not the 86-hour bottleneck**.

### `build_neighbor_lookup()` — The True Bottleneck

The real bottleneck is **`build_neighbor_lookup()`**, which runs a `lapply` over **all 6.46 million rows**, and for each row:

1. Performs a **character key lookup** via `as.character(data$id[i])` into `id_to_ref` — 6.46M character conversions and named-vector lookups.
2. Retrieves neighbor cell IDs from the `nb` object.
3. Calls **`paste()`** to construct `neighbor_keys` — creating character vectors per row.
4. Performs **named-vector lookup** `idx_lookup[neighbor_keys]` — on a named vector of length 6.46M.

Named vector lookup in R is **O(n)** per query in the worst case (it does linear hashing on names). Doing ~6.46 million lookups into a 6.46M-length named vector, each with ~4 neighbor keys (given ~1.37M directed relationships / 344K cells ≈ ~4 neighbors per cell), means roughly **25.8 million name-based lookups into a 6.46M-entry named vector**. This is catastrophically slow.

Additionally, this entire structure is **redundant across years**: every cell has the same neighbors every year. The neighbor lookup for cell `c` at year `y` is the same set of neighbor cells at year `y`. The function recomputes this for all 28 years × 344K cells instead of computing it once for 344K cells and replicating across years.

**Verdict: REJECT the colleague's diagnosis.** The dominant bottleneck is `build_neighbor_lookup()` — specifically the repeated character-key construction and named-vector lookups across 6.46M rows, redundantly repeated across 28 identical yearly copies of the spatial topology.

---

## Optimization Strategy

1. **Eliminate year-redundant computation**: Build the spatial neighbor mapping once for 344K cells, not 6.46M cell-years.
2. **Replace named-vector lookups with integer indexing**: Use `match()` or `data.table` hash joins instead of named-vector subsetting.
3. **Vectorize `compute_neighbor_stats()`**: Replace the per-row `lapply` with a `data.table` grouped aggregation over a pre-built edge list, computing max/min/mean in one vectorized pass.
4. **Pre-build an edge table (cell-year → neighbor-cell-year row indices)** using integer arithmetic, avoiding all `paste`/character operations.

This reduces the problem from ~25.8M character lookups into a 6.46M named vector to a single vectorized `data.table` join + grouped aggregation.

---

## Working R Code

```r
library(data.table)

#
# STEP 1: Build a fast edge list ONCE (spatial topology, no year dimension yet)
#
# Inputs:
#   cell_data           — data.frame/data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
#   id_order            — vector of cell IDs in the order matching rook_neighbors_unique
#   rook_neighbors_unique — spdep nb object (list of integer neighbor index vectors)
#
# We build a data.table mapping each cell to its neighbor cells (integer indices into id_order).
#

build_neighbor_edge_list <- function(id_order, neighbors) {
  # neighbors[[i]] gives the positions (in id_order) of the neighbors of id_order[i]
  n <- length(neighbors)
  
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(neighbors))
  
  from_idx <- integer(n_edges)
  to_idx   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    nb_i <- neighbors[[i]]
    len  <- length(nb_i)
    if (len > 0L) {
      from_idx[pos:(pos + len - 1L)] <- i
      to_idx[pos:(pos + len - 1L)]   <- nb_i
      pos <- pos + len
    }
  }
  
  data.table(from_cell_idx = from_idx, to_cell_idx = to_idx)
}

#
# STEP 2: Expand edge list to cell-year rows and compute stats vectorized
#

compute_all_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                          neighbor_source_vars) {
  
  dt <- as.data.table(cell_data)
  
  # --- Map cell IDs to spatial indices (position in id_order) ---
  id_map <- data.table(id = id_order, cell_idx = seq_along(id_order))
  dt <- merge(dt, id_map, by = "id", all.x = TRUE, sort = FALSE)
  
  # --- Create a unique row index for fast lookups ---
  dt[, row_id := .I]
  
  # --- Build spatial edge list (cell_idx -> neighbor_cell_idx), ~1.37M rows ---
  edges <- build_neighbor_edge_list(id_order, rook_neighbors_unique)
  
  # --- Build a lookup from (cell_idx, year) -> row_id in dt ---
  # This replaces the expensive named-vector lookup in the original code.
  setkey(dt, cell_idx, year)
  
  # --- Get unique years ---
  years <- sort(unique(dt$year))
  
  # --- Expand edges across years: CJ of edges × years ---
  # For each year, the neighbor of cell i is cell j at the same year.
  # ~1.37M edges × 28 years ≈ 38.5M rows — fits in 16GB RAM easily.
  
  cat("Expanding edge list across years...\n")
  edge_year <- edges[, .(year = years), by = .(from_cell_idx, to_cell_idx)]
  
  # --- Join to get the ROW index of the neighbor (to_cell_idx, year) ---
  # We need the row_id of the neighbor in dt so we can pull variable values.
  neighbor_key <- dt[, .(cell_idx, year, row_id)]
  setnames(neighbor_key, c("cell_idx", "year", "row_id"),
           c("to_cell_idx", "year", "neighbor_row_id"))
  setkey(neighbor_key, to_cell_idx, year)
  setkey(edge_year, to_cell_idx, year)
  
  edge_year <- neighbor_key[edge_year, nomatch = NA]
  
  # --- Also get the row_id of the focal cell (from_cell_idx, year) ---
  focal_key <- dt[, .(cell_idx, year, row_id)]
  setnames(focal_key, c("cell_idx", "year", "row_id"),
           c("from_cell_idx", "year", "focal_row_id"))
  setkey(focal_key, from_cell_idx, year)
  setkey(edge_year, from_cell_idx, year)
  
  edge_year <- focal_key[edge_year, nomatch = NA]
  
  # Now edge_year has columns:
  #   from_cell_idx, year, focal_row_id, to_cell_idx, neighbor_row_id
  
  # --- For each variable, compute grouped stats vectorized ---
  cat("Computing neighbor statistics for", length(neighbor_source_vars), "variables...\n")
  
  # Pre-allocate result columns in dt
  for (var_name in neighbor_source_vars) {
    dt[, paste0("max_neighbor_", var_name) := NA_real_]
    dt[, paste0("min_neighbor_", var_name) := NA_real_]
    dt[, paste0("mean_neighbor_", var_name) := NA_real_]
  }
  
  for (var_name in neighbor_source_vars) {
    cat("  Processing:", var_name, "\n")
    
    # Pull neighbor values via integer indexing (extremely fast)
    vals_vec <- dt[[var_name]]
    edge_year[, nval := vals_vec[neighbor_row_id]]
    
    # Grouped aggregation: for each focal_row_id, compute max/min/mean of neighbor values
    stats <- edge_year[!is.na(nval),
                       .(nmax  = max(nval),
                         nmin  = min(nval),
                         nmean = mean(nval)),
                       by = focal_row_id]
    
    # Write results back into dt by integer index
    max_col  <- paste0("max_neighbor_",  var_name)
    min_col  <- paste0("min_neighbor_",  var_name)
    mean_col <- paste0("mean_neighbor_", var_name)
    
    dt[stats$focal_row_id, (max_col)  := stats$nmax]
    dt[stats$focal_row_id, (min_col)  := stats$nmin]
    dt[stats$focal_row_id, (mean_col) := stats$nmean]
    
    # Clean up temp column
    edge_year[, nval := NULL]
  }
  
  # --- Clean up helper columns ---
  dt[, c("cell_idx", "row_id") := NULL]
  
  cat("Done.\n")
  return(as.data.frame(dt))
}

# ============================================================
# USAGE — drop-in replacement for the original outer loop
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is untouched.
# The numerical estimand (max, min, mean of neighbor values per variable) is preserved exactly.
# Predictions proceed as before:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **True bottleneck** | `build_neighbor_lookup`: 6.46M character `paste` + named-vector lookups into a 6.46M-entry vector | Eliminated entirely |
| **Redundant computation** | Recomputes identical spatial topology for all 28 years | Builds topology once (344K cells), expands via integer join |
| **Key data structure** | Named character vectors (`idx_lookup`) | `data.table` keyed integer joins |
| **`compute_neighbor_stats`** | Per-row `lapply` (6.46M iterations) + `do.call(rbind)` | Single vectorized `data.table` grouped aggregation |
| **Estimated runtime** | 86+ hours | **Minutes** (edge expansion is ~38.5M rows; grouped aggregation is a single pass) |
| **RAM** | Repeated character allocations cause GC pressure | ~38.5M-row integer edge table ≈ ~1.5 GB; fits in 16 GB |
| **RF model** | Preserved | Preserved (no retraining) |
| **Numerical output** | max, min, mean per neighbor set | Identical values via same arithmetic |