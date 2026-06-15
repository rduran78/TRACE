 **Diagnosis**  
The main bottleneck is `build_neighbor_lookup` constructing a list of length 6.46M (one element per row), where each element stores integer indices for neighbor observations per year. The costly `paste()` and repeated list allocations inside `lapply` make this step extremely slow and memory-heavy. Additionally, `compute_neighbor_stats` repeatedly traverses this large structure for each variable. The complexity scales with `O(n_rows × avg_neighbors × n_vars)` using lists.

**Optimization Strategy**  
- Avoid building a massive per-row list. Instead, compute neighbor statistics using **vectorized joins**.
- Represent panel data as a keyed `data.table` and perform a self-join on `(neighbor_id, year)` to fetch all neighbor values in one shot, then aggregate.
- This drastically reduces memory use and eliminates R-level loops.

---

### Optimized R Code

```r
library(data.table)

# Convert to data.table
dt <- as.data.table(cell_data)
setkey(dt, id, year)

# Prepare neighbor mapping expanded over years
neighbor_dt <- data.table(from_id = rep(id_order, lengths(rook_neighbors_unique)),
                           to_id = unlist(rook_neighbors_unique))
# Cartesian join with years to cover panel structure
years <- sort(unique(dt$year))
neighbor_dt <- neighbor_dt[, .(from_id = rep(from_id, each = length(years)),
                                to_id   = rep(to_id, each = length(years)),
                                year    = rep(years, times = .N))]
setkey(neighbor_dt, to_id, year)

# Join neighbor values for each target variable and compute stats
for (var_name in c("ntl", "ec", "pop_density", "def", "usd_est_n2")) {

  # Join neighbor values
  joined <- neighbor_dt[dt, on = .(to_id = id, year), nomatch = 0]
  # Compute stats by from_id-year
  stats <- joined[, .(
    max_val = max(get(var_name), na.rm = TRUE),
    min_val = min(get(var_name), na.rm = TRUE),
    mean_val = mean(get(var_name), na.rm = TRUE)
  ), by = .(from_id, year)]

  # Replace Inf/-Inf from all-NA with NA
  is.nan.or.infinite <- function(x) is.nan(x) | is.infinite(x)
  for (col in names(stats)[-c(1,2)]) set(stats, which(is.nan.or.infinite(stats[[col]])), col, NA)
  
  # Merge back to main dt
  setkey(stats, from_id, year)
  dt <- stats[dt, on = .(from_id = id, year)]
  setnames(dt, c("max_val", "min_val", "mean_val"),
               paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean")))
}

# Result: dt now has neighbor-based features appended
cell_data <- as.data.frame(dt)
```

---

**Why this works**  
- Eliminates per-row `paste` and `lapply`.
- Uses `data.table` for efficient keyed joins and grouped aggregation in C, not R loops.
- Single pass per variable, no giant list objects.
- Memory footprint is dominated by join tables (~pairs × years), still feasible within 16 GB.

**Expected Speed-Up**  
From 86+ hours to a few hours or less on a modern laptop because operations are vectorized and implemented in optimized C code instead of nested R lists.