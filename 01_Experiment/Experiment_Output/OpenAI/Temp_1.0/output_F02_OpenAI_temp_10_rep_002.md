 **Diagnosis**  
The bottleneck is from nested `lapply` calls over ~6.5M rows and repeated R-level operations. Memory pressure comes from building huge lists (`neighbor_lookup` of length 6.46M) and repeatedly computing neighbor statistics in pure R loops. This approach is inefficient because:

- `build_neighbor_lookup` expands neighbors per row repeatedly by string concatenations (`paste`) and hashmap lookups.
- `compute_neighbor_stats` loops per row again and aggregates at R level.
- No vectorization; all row-wise operations are interpreted.
- Large intermediate lists (millions of elements) blow up RAM.

---

**Optimization Strategy**  
Key targets:
1. **Avoid row-wise loops**: Restructure the pipeline to compute features using *joins/grouped summarization* rather than per-row lists.
2. **Precompute neighbor mappings in a long format once** and join back using `data.table` for speed and low memory.
3. Use **fast aggregation** with `data.table` (`fread`, keyed joins) to compute max/min/mean in bulk.
4. Do all operations for all variables in a single grouped pass to reduce repeated scans.
5. Ensure chunk-wise/stream processing if memory remains tight.

---

**Optimized Implementation with `data.table`**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2

# Convert neighbors list (rook_neighbors_unique) into long table
# id_order: vector of cell IDs in same order as rook_neighbors_unique
neighbor_dt <- rbindlist(
  lapply(seq_along(rook_neighbors_unique), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(src_id = id_order[i], nb_id = id_order[rook_neighbors_unique[[i]]])
  })
)

# Cartesian expansion across years
years <- sort(unique(cell_data$year))
neighbor_dt <- neighbor_dt[, .(id = src_id, nb_id, year = years), by = seq_len(nrow(neighbor_dt))]
neighbor_dt[, seq_len := NULL]

# Merge neighbor values
setkey(cell_data, id, year)
setkey(neighbor_dt, nb_id, year)
merged <- cell_data[neighbor_dt, on = .(id = nb_id, year), nomatch = 0L]

# Compute stats for all neighbor_source_vars at once
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

agg <- merged[, .(
  ntl_max  = max(ntl, na.rm=TRUE),
  ntl_min  = min(ntl, na.rm=TRUE),
  ntl_mean = mean(ntl, na.rm=TRUE),
  ec_max   = max(ec, na.rm=TRUE),
  ec_min   = min(ec, na.rm=TRUE),
  ec_mean  = mean(ec, na.rm=TRUE),
  pop_max  = max(pop_density, na.rm=TRUE),
  pop_min  = min(pop_density, na.rm=TRUE),
  pop_mean = mean(pop_density, na.rm=TRUE),
  def_max  = max(def, na.rm=TRUE),
  def_min  = min(def, na.rm=TRUE),
  def_mean = mean(def, na.rm=TRUE),
  usd_max  = max(usd_est_n2, na.rm=TRUE),
  usd_min  = min(usd_est_n2, na.rm=TRUE),
  usd_mean = mean(usd_est_n2, na.rm=TRUE)
), by = .(id, year)]

# Merge back to cell_data
cell_data <- agg[cell_data, on = .(id, year)]

# Clean inf values if any NA-only groups existed
replace_inf <- function(x) fifelse(is.infinite(x), NA_real_, x)
cell_data[, names(agg)[-(1:2)] := lapply(.SD, replace_inf), .SDcols = names(agg)[-(1:2)]]
```

---

**Why this is faster**:  
- All neighbor relationships represented as a single long table (`neighbor_dt`) instead of millions of small lists.
- Single keyed join + grouped aggregation instead of millions of function calls.
- `data.table` uses C-level operations for grouping → orders of magnitude faster and memory efficient.
- Handles all variables in one scan.

**Expected performance**: Minutes to low hours on 16 GB RAM, compared to 86+ hours.

**Preserves**:  
- Original estimand (same max/min/mean per (id, year) over neighbors).
- Random Forest model remains untouched.

This is the practical, scalable approach on a standard laptop.