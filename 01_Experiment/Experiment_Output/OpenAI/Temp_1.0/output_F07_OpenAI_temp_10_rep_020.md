 **Diagnosis:**  
The bottleneck comes from the naive nested loop with `lapply` over 6.46M rows, repeatedly traversing neighbor lists and computing summary stats in R. This results in huge R-level overhead, poor memory locality, and no vectorization. Each row independently looks up neighbor indices and computes statistics, which is extremely costly for millions of iterations.

**Optimization Strategy:**  
- Precompute **neighbor lookup** once (already done), but store in a memory-efficient vectorized structure (e.g., integer matrix or compressed list).
- Avoid repeated `lapply` calls for every cell by doing **vectorized aggregation** or highly optimized loops in C++ (via `data.table` groups, `Rcpp`, or `matrixStats`).
- Use `data.table` keyed joins or a **long format edge table** (cell–neighbor pairs), then compute max/min/mean for each variable with fast grouping (`fread`, `:=`, `by`).
- Do everything in memory-efficient chunks if needed.
- Preserve original estimand by ensuring no imputation or approximation.
  
**Vectorized Workflow:**  
Convert neighbor relationships into a long table, replicate across years, join with `data`, and aggregate:

```r
library(data.table)

# Convert to long edge list (directed)
edges <- data.table(
  from = rep(id_order, lengths(rook_neighbors_unique)),
  to   = unlist(rook_neighbors_unique)
)

# Cross with years
years <- sort(unique(cell_data$year))
edges <- edges[CJ(year = years), on = .(dummy = NULL)][
  , from_key := paste(from, year, sep = "_")
][
  , to_key := paste(to, year, sep = "_")
]

# Convert cell_data to data.table and key by id_year
setDT(cell_data)
cell_data[, id_year := paste(id, year, sep = "_")]
setkey(cell_data, id_year)

# Merge neighbor values long-form
edges <- merge(edges, cell_data[, .(id_year, ntl, ec, pop_density, def, usd_est_n2)],
               by.x = "to_key", by.y = "id_year", all.x = TRUE)

# Compute aggregates by 'from_key'
neighbor_stats <- edges[, .(
  ntl_max = max(ntl, na.rm=TRUE),
  ntl_min = min(ntl, na.rm=TRUE),
  ntl_mean = mean(ntl, na.rm=TRUE),
  
  ec_max = max(ec, na.rm=TRUE),
  ec_min = min(ec, na.rm=TRUE),
  ec_mean = mean(ec, na.rm=TRUE),
  
  pop_max = max(pop_density, na.rm=TRUE),
  pop_min = min(pop_density, na.rm=TRUE),
  pop_mean = mean(pop_density, na.rm=TRUE),
  
  def_max = max(def, na.rm=TRUE),
  def_min = min(def, na.rm=TRUE),
  def_mean = mean(def, na.rm=TRUE),
  
  usd_max = max(usd_est_n2, na.rm=TRUE),
  usd_min = min(usd_est_n2, na.rm=TRUE),
  usd_mean = mean(usd_est_n2, na.rm=TRUE)
), by = from_key]

# Merge back
setkey(neighbor_stats, from_key)
cell_data <- merge(cell_data, neighbor_stats, by.x = "id_year", by.y = "from_key", all.x = TRUE)
```

**Why this is fast & memory-friendly:**  
- `data.table` grouping on ~6.46M rows and ~1.37M neighbor pairs is efficient and uses optimized C code.
- Eliminates 6.46M `lapply` calls.
- Single wide join plus vectorized summarization in one step.
- Fits in 16GB RAM with careful memory usage; if not, chunk by year.

**Expected Performance Gain:**  
From >86 hours → **minutes** or low hour range, depending on disk speed and system memory, since operations are fully vectorized and optimized. The trained Random Forest remains untouched, and numerical estimands stay exact.