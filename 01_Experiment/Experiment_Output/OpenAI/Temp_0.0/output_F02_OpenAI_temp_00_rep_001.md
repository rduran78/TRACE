 **Diagnosis**  
The current implementation is extremely slow and memory-intensive because:  
- `build_neighbor_lookup` creates a large list of length 6.46M (one per cell-year), each storing neighbor indices. This is highly inefficient in both memory and time.  
- `compute_neighbor_stats` repeatedly traverses large lists and subsets vectors, causing heavy R-level overhead.  
- The outer loop calls `compute_and_add_neighbor_features` five times, repeating expensive operations.  
- Everything is done in pure R with `lapply` and `do.call(rbind, ...)` on millions of rows, which is not scalable.  

**Optimization Strategy**  
1. **Avoid per-row neighbor lookups**: Instead of building a 6.46M-element list, use a long-format join approach.  
2. **Exploit vectorization and data.table**: Convert data to `data.table` and compute neighbor stats via grouped operations.  
3. **Precompute neighbor relationships once**: Expand neighbor relationships across years in a long table and join with values.  
4. **Compute all neighbor stats in one pass**: Instead of looping over variables, melt and compute in a grouped manner.  
5. **Memory efficiency**: Use integer keys and avoid large intermediate lists.  

**Working R Code (Optimized)**  
```r
library(data.table)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer vectors (spdep nb object)
# id_order: vector of cell ids in same order as rook_neighbors_unique

# Convert to data.table
setDT(cell_data)

# Build neighbor pairs (cell_id -> neighbor_id)
neighbor_dt <- rbindlist(
  lapply(seq_along(rook_neighbors_unique), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(id = id_order[i], neighbor_id = id_order[rook_neighbors_unique[[i]]])
  })
)

# Expand across years
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(id = rep(id, each = length(years)),
                                neighbor_id = rep(neighbor_id, each = length(years)),
                                year = rep(years, times = .N))]

# Merge neighbor values
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_dt <- merge(neighbor_dt, cell_data[, c("id", "year", vars), with = FALSE],
                     by.x = c("neighbor_id", "year"), by.y = c("id", "year"),
                     all.x = TRUE)

# Compute neighbor stats by (id, year)
stat_dt <- neighbor_dt[, .(
  ntl_max = max(ntl, na.rm = TRUE),
  ntl_min = min(ntl, na.rm = TRUE),
  ntl_mean = mean(ntl, na.rm = TRUE),
  ec_max = max(ec, na.rm = TRUE),
  ec_min = min(ec, na.rm = TRUE),
  ec_mean = mean(ec, na.rm = TRUE),
  pop_density_max = max(pop_density, na.rm = TRUE),
  pop_density_min = min(pop_density, na.rm = TRUE),
  pop_density_mean = mean(pop_density, na.rm = TRUE),
  def_max = max(def, na.rm = TRUE),
  def_min = min(def, na.rm = TRUE),
  def_mean = mean(def, na.rm = TRUE),
  usd_est_n2_max = max(usd_est_n2, na.rm = TRUE),
  usd_est_n2_min = min(usd_est_n2, na.rm = TRUE),
  usd_est_n2_mean = mean(usd_est_n2, na.rm = TRUE)
), by = .(id, year)]

# Replace NaN with NA
for (j in names(stat_dt)) set(stat_dt, which(is.nan(stat_dt[[j]])), j, NA)

# Merge back to cell_data
cell_data <- merge(cell_data, stat_dt, by = c("id", "year"), all.x = TRUE)
```

**Why this is faster**  
- Eliminates 6.46M list elements and repeated `lapply`.  
- Uses `data.table` joins and grouped aggregation in C, which is orders of magnitude faster.  
- Computes all neighbor-based features in one grouped pass.  

**Expected performance**  
On 6.46M rows and ~1.37M neighbor links, this approach should complete in minutes to a few hours on a 16 GB laptop, instead of 86+ hours.  

**Preserves**  
- Original estimand and trained Random Forest model.  
- All neighbor-based features (max, min, mean) for the five variables.