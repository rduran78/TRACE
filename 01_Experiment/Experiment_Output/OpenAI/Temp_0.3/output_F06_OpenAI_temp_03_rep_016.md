 **Diagnosis**  
The bottleneck is the nested `lapply` over ~6.46M rows, repeatedly scanning neighbor lists and computing stats. Each row recomputes neighbor indices and aggregates, causing massive overhead. Memory is stressed by large lists and repeated allocations. The algorithm is effectively doing a join + group aggregation in pure R loops, which is extremely slow for millions of rows.

---

**Optimization Strategy**  
- Precompute neighbor relationships as a long table (cell-year → neighbor-year).
- Use **vectorized joins** and **grouped aggregation** via `data.table` for speed and memory efficiency.
- Compute all 3 stats (max, min, mean) in one grouped operation per variable.
- Avoid per-row `lapply`; instead, leverage fast keyed joins.
- Keep the Random Forest model unchanged; only speed up feature engineering.

---

**Optimized R Code (data.table approach)**  
```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute neighbor pairs (cell_id -> neighbor_id)
# rook_neighbors_unique: list of integer vectors
neighbor_pairs <- rbindlist(
  lapply(seq_along(rook_neighbors_unique), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(id = id_order[i], neighbor_id = id_order[rook_neighbors_unique[[i]]])
  })
)

# Expand to cell-year pairs
years <- unique(cell_data$year)
neighbor_dt <- CJ(year = years, id = neighbor_pairs$id)[
  , neighbor_id := neighbor_pairs$neighbor_id[match(id, neighbor_pairs$id)]
]

# Melt cell_data for join
setkey(cell_data, id, year)
setkey(neighbor_dt, neighbor_id, year)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  # Join neighbor values
  joined <- neighbor_dt[cell_data, on = .(neighbor_id = id, year), nomatch = 0]
  setnames(joined, var_name, "neighbor_val")
  
  # Aggregate by focal cell-year
  agg <- joined[
    !is.na(neighbor_val),
    .(
      max_val = max(neighbor_val),
      min_val = min(neighbor_val),
      mean_val = mean(neighbor_val)
    ),
    by = .(id, year)
  ]
  
  # Merge back into cell_data
  setnames(agg, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  cell_data <- merge(cell_data, agg, by = c("id", "year"), all.x = TRUE)
}
```

---

**Why This Works**  
- Converts neighbor relationships into a normalized table and uses `data.table` joins and aggregations (highly optimized in C).
- Eliminates per-row loops and repeated lookups.
- Handles all years in one pass using `CJ` (cross join) and keyed joins.
- Preserves original estimand (same max, min, mean definitions).
- Memory-friendly: processes variable-by-variable, not all at once.

---

**Expected Performance**  
From 86+ hours to **minutes** on a 16 GB laptop, as `data.table` can handle tens of millions of rows efficiently.