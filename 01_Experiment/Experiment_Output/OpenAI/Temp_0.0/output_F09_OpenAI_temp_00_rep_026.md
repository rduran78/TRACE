 **Diagnosis**  
The current pipeline is extremely slow because for each of the ~6.46M rows, `build_neighbor_lookup` and `compute_neighbor_stats` repeatedly perform expensive list lookups and string concatenations. The neighbor relationships are static across years, but the code recomputes neighbor indices for every row-year combination. This results in massive redundant computation and memory overhead.

**Optimization Strategy**  
- Precompute a reusable neighbor lookup table **once** at the cell level (not per row-year).
- For each year, join cell-year attributes to this static neighbor table and compute neighbor statistics using fast vectorized operations or `data.table`.
- Avoid repeated string concatenation and list traversals inside loops.
- Use `data.table` for efficient joins and aggregations.
- Keep the Random Forest model unchanged and preserve the original estimand.

---

### **Optimized R Code**

```r
library(data.table)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer vectors (neighbors per cell)
# id_order: vector of cell IDs in same order as rook_neighbors_unique

# Convert to data.table
setDT(cell_data)

# Precompute neighbor pairs (static across years)
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Set keys for fast joins
setkey(cell_data, id, year)
setkey(neighbor_dt, id)

# Function to compute neighbor stats for one variable
compute_neighbor_stats_dt <- function(var_name) {
  # Join neighbor pairs with cell_data for neighbor values
  joined <- neighbor_dt[cell_data, on = .(neighbor_id = id), allow.cartesian = TRUE]
  # joined now has: id (focal), neighbor_id, year, var_name
  setnames(joined, var_name, "neighbor_val")
  
  # Aggregate by focal id-year
  stats <- joined[, .(
    max_val = if (.N > 0) max(neighbor_val, na.rm = TRUE) else NA_real_,
    min_val = if (.N > 0) min(neighbor_val, na.rm = TRUE) else NA_real_,
    mean_val = if (.N > 0) mean(neighbor_val, na.rm = TRUE) else NA_real_
  ), by = .(id, year)]
  
  # Merge back to cell_data
  setnames(stats, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  cell_data[stats, on = .(id, year), 
            (paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))) := 
              mget(paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))]
}

# Apply for all neighbor source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  compute_neighbor_stats_dt(var_name)
}

# cell_data now has neighbor features added
# Proceed with Random Forest prediction using the existing trained model
```

---

**Why This is Faster**  
- Neighbor relationships are computed **once** (1.37M pairs) instead of 6.46M times.
- Uses `data.table` joins and group operations, which are highly optimized in C.
- Eliminates repeated string concatenation and list indexing.
- Memory footprint is reduced because we avoid creating massive intermediate lists.

**Expected Performance**  
This approach should reduce runtime from 86+ hours to a few hours or less on a standard laptop, depending on I/O and CPU speed.