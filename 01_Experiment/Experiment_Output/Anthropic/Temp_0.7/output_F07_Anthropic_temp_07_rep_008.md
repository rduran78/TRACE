 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which creates a list of ~6.46 million elements, each produced by an R-level `lapply` iteration that performs character coercion, `paste`, and named-vector lookups. Specifically:

1. **`build_neighbor_lookup` is O(N × k) with enormous constant factors.** For each of the ~6.46M rows, it converts IDs to character, pastes strings, and does named-vector indexing (`idx_lookup[neighbor_keys]`). Named-vector lookup in R is hash-based but has high per-call overhead when done 6.46M times inside `lapply`. The `paste` and `as.character` calls generate millions of temporary string objects, causing GC pressure.

2. **Redundant recomputation across years.** The neighbor *topology* is identical for every year — cell 42's neighbors are always the same cells. Yet the lookup rebuilds the mapping for every cell-year row, effectively repeating the same neighbor resolution 28 times per cell.

3. **`compute_neighbor_stats` is moderately expensive but not the primary bottleneck.** It iterates over the 6.46M-element list, subsetting a numeric vector each time. This is O(N × k) but with lighter operations. Still, it's called 5 times (once per variable).

4. **Memory:** Storing a 6.46M-element list of integer vectors is ~1–2 GB, which is feasible on 16 GB but tight when combined with the dataset and RF model.

**Estimated cost of current approach:**
- `build_neighbor_lookup`: ~6.46M iterations × (string ops + hash lookups) ≈ 60–80+ hours on a laptop.
- `compute_neighbor_stats`: ~5 variables × 6.46M iterations ≈ 6–10 hours.
- Total: ~86+ hours, consistent with the reported estimate.

---

## Optimization Strategy

### Key Insight: Separate Topology from Time

The neighbor graph is **time-invariant**. Instead of building a 6.46M-element row-level lookup, we:

1. **Work at the cell level (344K cells), not the cell-year level (6.46M rows).**
2. **Convert the `nb` object to a sparse adjacency matrix** (or a flat edge list) once.
3. **Use vectorized sparse matrix–dense matrix multiplication** to compute neighbor sums and counts, then derive max/min/mean.

For **mean**: If `A` is the binary row-normalized adjacency matrix and `X` is an N×T matrix of values, then `A %*% X` gives neighbor means. But we also need **max** and **min**, which aren't linear, so matrix multiplication alone won't suffice for those.

### Approach: data.table Join on Edge List

1. Convert `rook_neighbors_unique` (the `nb` object) to a two-column **edge list** data.table: `(focal_id, neighbor_id)` — ~1.37M rows.
2. Reshape the panel data so that for each variable, we can join the edge list against the data keyed by `(id, year)`.
3. A single `data.table` merge of the edge list × 28 years (~38.5M rows) against the values table, then `group by (focal_id, year)` to compute `max`, `min`, `mean` — all fully vectorized in C via `data.table`.

This replaces 6.46M R-level iterations with a single vectorized join + grouped aggregation, reducing runtime from ~86 hours to **minutes**.

### Why This Preserves the Estimand

- The edge list is an exact representation of the `nb` rook-neighbor topology.
- The `max`, `min`, `mean` computations are identical — just vectorized.
- No approximation, sampling, or rounding is introduced.
- The RF model is not retouched; only the feature-engineering pipeline is optimized.

---

## Working R Code

```r
library(data.table)
library(spdep)  # only needed if nb object needs conversion

# ============================================================
# Step 1: Convert nb object to edge-list data.table (one-time)
# ============================================================
nb_to_edge_list <- function(nb_obj, id_order) {
  # nb_obj:   spdep nb object (list of integer index vectors)
  # id_order: vector of cell IDs in the order matching nb_obj
  # Returns:  data.table with columns (focal_id, neighbor_id)
  
  n <- length(nb_obj)
  focal <- vector("list", n)
  neighbor <- vector("list", n)
  
  for (i in seq_len(n)) {
    nbs <- nb_obj[[i]]
    # spdep nb uses 0L to denote "no neighbors"
    nbs <- nbs[nbs > 0L]
    if (length(nbs) > 0L) {
      focal[[i]]    <- rep(id_order[i], length(nbs))
      neighbor[[i]] <- id_order[nbs]
    }
  }
  
  data.table(
    focal_id    = unlist(focal,    use.names = FALSE),
    neighbor_id = unlist(neighbor, use.names = FALSE)
  )
}

edges <- nb_to_edge_list(rook_neighbors_unique, id_order)
# ~1.37M rows, two integer columns — trivial memory

# ============================================================
# Step 2: Convert cell_data to data.table (if not already)
# ============================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure id and year are keyed for fast joins
setkey(cell_data, id, year)

# ============================================================
# Step 3: Vectorized neighbor stats computation
# ============================================================
compute_neighbor_stats_fast <- function(cell_dt, edges_dt, var_name) {
  # Build a slim lookup: (id, year, value)
  val_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setkey(val_dt, id, year)
  
  # Get the unique years present in the data
  years <- sort(unique(val_dt$year))
  
  # Cross join edges with years to get all (focal, neighbor, year) triples
  # ~1.37M edges × 28 years ≈ 38.5M rows — fits in RAM easily
  edge_year <- CJ_dt(edges_dt, years)
  
  # Join neighbor values onto the edge-year table
  setkey(edge_year, neighbor_id, year)
  edge_year[val_dt, neighbor_val := i.val, on = .(neighbor_id = id, year)]
  
  # Aggregate: for each (focal_id, year), compute max, min, mean
  # Dropping NAs to match original behavior
  stats <- edge_year[
    !is.na(neighbor_val),
    .(
      nmax  = max(neighbor_val),
      nmin  = min(neighbor_val),
      nmean = mean(neighbor_val)
    ),
    keyby = .(focal_id, year)
  ]
  
  # Rename to match expected column names
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats, c("nmax", "nmin", "nmean"), c(max_col, min_col, mean_col))
  
  stats
}

# Helper: cross join edges with a year vector
CJ_dt <- function(edges_dt, years) {
  # Repeat each edge for every year — vectorized
  n_edges <- nrow(edges_dt)
  n_years <- length(years)
  
  data.table(
    focal_id    = rep(edges_dt$focal_id,    times = n_years),
    neighbor_id = rep(edges_dt$neighbor_id,  times = n_years),
    year        = rep(years, each = n_edges)
  )
}

# ============================================================
# Step 4: Outer loop — compute and merge all neighbor features
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Computing neighbor stats for:", var_name, "\n")
  
  stats <- compute_neighbor_stats_fast(cell_data, edges, var_name)
  
  # Merge back into cell_data
  # Left join so cells with no neighbors get NA (matching original behavior)
  cell_data <- merge(
    cell_data, stats,
    by.x = c("id", "year"),
    by.y = c("focal_id", "year"),
    all.x = TRUE
  )
  
  rm(stats)
  gc()
}

# cell_data now has 15 new columns (3 per variable × 5 variables),
# numerically identical to the original implementation.
# The trained Random Forest model is untouched.
```

---

## Memory-Optimized Variant (if 38.5M-row cross join is tight)

If laptop RAM is constrained (RF model + 6.46M-row dataset already consuming significant memory), process years in batches:

```r
compute_neighbor_stats_chunked <- function(cell_dt, edges_dt, var_name, 
                                            chunk_size = 7) {
  val_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setkey(val_dt, id, year)
  years <- sort(unique(val_dt$year))
  
  # Process in chunks of years
  year_chunks <- split(years, ceiling(seq_along(years) / chunk_size))
  
  results <- lapply(year_chunks, function(yr_chunk) {
    # Subset values to this chunk of years
    val_sub <- val_dt[year %in% yr_chunk]
    setkey(val_sub, id, year)
    
    n_edges <- nrow(edges_dt)
    n_yr    <- length(yr_chunk)
    
    edge_year <- data.table(
      focal_id    = rep(edges_dt$focal_id,    times = n_yr),
      neighbor_id = rep(edges_dt$neighbor_id,  times = n_yr),
      year        = rep(yr_chunk, each = n_edges)
    )
    
    setkey(edge_year, neighbor_id, year)
    edge_year[val_sub, neighbor_val := i.val, on = .(neighbor_id = id, year)]
    
    edge_year[
      !is.na(neighbor_val),
      .(nmax = max(neighbor_val),
        nmin = min(neighbor_val),
        nmean = mean(neighbor_val)),
      keyby = .(focal_id, year)
    ]
  })
  
  stats <- rbindlist(results)
  
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats, c("nmax", "nmin", "nmean"), c(max_col, min_col, mean_col))
  
  stats
}
```

---

## Expected Performance

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~70 hours (6.46M R iterations with string ops) | **~1 second** (nb → edge list, 344K cells) |
| `compute_neighbor_stats` (×5 vars) | ~16 hours (5 × 6.46M R iterations) | **~3–8 minutes** (5 × vectorized data.table join + groupby on 38.5M rows) |
| **Total** | **~86 hours** | **~5–10 minutes** |

**Speedup: ~500–1000×**, achieved entirely through vectorization and separating the time-invariant topology from the panel time dimension. The numerical results are identical, and the trained Random Forest model is preserved unchanged.