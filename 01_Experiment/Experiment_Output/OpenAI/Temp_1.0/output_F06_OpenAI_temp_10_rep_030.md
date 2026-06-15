 **Diagnosis**  
The main bottleneck is the iterative approach in `compute_neighbor_stats()` across 6.46M rows, combined with repeated list processing and lookups. Each row performs a separate small neighbor computation using `lapply`, which is extremely inefficient at this scale.  
Key issues:  
- **High-frequency R loops**: Billions of interpreted operations.  
- **Redundant computation** for each variable separately.  
- **Inefficient memory layout** for join-like operations.  

---

**Optimization Strategy**  
1. **Vectorize & collapse id/neighbor operations into a *long join***:  
   - Build a long-form table linking each cell-year `src` to its neighbors `nbr` through `data.table`.  
   - Compute grouped stats (`max`, `min`, `mean`) per `(src_id, year)` in one pass.  
   - Repeat for 5 variables without redundant lookups.  

2. **Leverage `data.table` for speed and memory efficiency**:  
   - Single neighbor lookup creation → long table: `(src_key, nbr_idx)`.  
   - Use `on` joins instead of looping through lists.  

3. **Preserve estimands**: Stats must be computed exactly over neighbors, skipping NAs (as before).  

Expected improvement: From **86+ hours** to about **minutes** on 16 GB RAM by removing per-row `lapply` calls.

---

**Optimized R Code**  

```r
library(data.table)

# Convert input to data.table
setDT(cell_data)

# Precompute keys
cell_data[, key := paste(id, year, sep = "_")]

# Step 1: Build neighbor table in long form once
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
idx_lookup <- setNames(seq_len(nrow(cell_data)), cell_data$key)

# Create base mapping: each id -> neighbors
neighbor_pairs <- rbindlist(lapply(seq_along(id_order), function(ref_idx) {
  src <- id_order[ref_idx]
  nbrs <- id_order[rook_neighbors_unique[[ref_idx]]]
  if (length(nbrs) == 0) return(NULL)
  data.table(src_id = src, nbr_id = nbrs)
}))

# Expand across all years
years <- unique(cell_data$year)
src_year <- CJ(src_id = neighbor_pairs$src_id, year = years)
nbr_year <- CJ(nbr_id = neighbor_pairs$nbr_id, year = years)

# Fast merge friendlier approach
neighbor_dt <- neighbor_pairs[, .(src_id, nbr_id)][, year := rep(years, each = .N)]
neighbor_dt[, src_key := paste(src_id, year, sep = "_")]
neighbor_dt[, nbr_key := paste(nbr_id, year, sep = "_")]

# Join neighbor keys to cell_data indices
idx_map <- data.table(key = cell_data$key, idx = seq_len(nrow(cell_data)))
neighbor_dt <- neighbor_dt[idx_map, on = c("src_key=key"), nomatch=0][, src_idx := idx][, idx := NULL]
neighbor_dt <- neighbor_dt[idx_map, on = c("nbr_key=key"), nomatch=0][, nbr_idx := idx][, idx := NULL]
neighbor_dt <- neighbor_dt[!is.na(src_idx) & !is.na(nbr_idx), .(src_idx, nbr_idx)]

# Step 2: Compute stats for each variable
compute_neighbor_stats_dt <- function(var) {
  vals <- cell_data[[var]]
  dt <- data.table(src_idx = neighbor_dt$src_idx,
                   val = vals[neighbor_dt$nbr_idx])
  dt <- dt[!is.na(val), .(max = max(val),
                           min = min(val),
                           mean = mean(val)), by = src_idx]
  colnames(dt)[-1] <- paste(var, c("nbr_max","nbr_min","nbr_mean"), sep = "_")
  dt
}

# Apply for all source variables
res_list <- lapply(neighbor_source_vars, compute_neighbor_stats_dt)
res_all <- Reduce(function(...) merge(..., by="src_idx", all=TRUE), res_list)

# Merge back into original data
cell_data <- cbind(cell_data, res_all[order(src_idx)][, src_idx := NULL])
```

---

**Why this works:**  
- Single precomputed `(src_idx, nbr_idx)` table replaces billions of micro-operations.  
- `data.table` aggregation is batch-optimized and uses C under the hood.  
- Only minimal joins and merges rather than repetitive R loops.  

**Time estimate:**  
From **~86 hours** down to **< 1 hour** on 16 GB RAM with `data.table`.  

**Next step:** After adding features, feed directly to the existing Random Forest predict step—no retraining required.