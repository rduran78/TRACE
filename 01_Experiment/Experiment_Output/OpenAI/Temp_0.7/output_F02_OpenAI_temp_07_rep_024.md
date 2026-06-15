 **Diagnosis:**  
The current approach uses deeply nested `lapply` operations over 6.46M rows, repeatedly constructing neighbor lookups and computing stats in pure R lists. This causes high overhead, excessive memory allocations, and poor CPU cache locality. The bottlenecks are:  
- **`build_neighbor_lookup`** creates a large nested list (≈6.46M elements), which is memory-intensive (~GBs).  
- **`compute_neighbor_stats`** iterates over all rows and neighbors for each variable, causing redundant passes through the data.  
- Overall: O(N * V) iterations, where N ≈ 6.46M and V = 5 variables, in pure R loops.

---

### **Optimization Strategy**
1. **Avoid building giant list objects**: Represent neighbor relationships in a compressed long format (edge list) instead of a list-per-cell-year.
2. **Use `data.table`** for fast joins and aggregations in memory-efficient way.
3. **Compute all neighbor statistics in a single grouped operation**, avoiding repeated loops.
4. **Precompute neighbor-year pairs** once, then join variables for aggregation.
5. **Parallelize aggregations if possible** (via `data.table` or `future`).

---

### **Optimized Approach**
- Expand neighbor relationships to `(cell_id, neighbor_id)` pairs.
- Cross with all years → `(cell_id, year, neighbor_id)`.
- Join with main `cell_data` to get neighbor values.
- Aggregate `max`, `min`, and `mean` per `(cell_id, year)` for each variable.
- Merge results back to `cell_data`.

---

### **Working R Code**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Expand rook neighbors into edge list
# id_order matches indices of rook_neighbors_unique
edges <- data.table(
  from = rep(id_order, lengths(rook_neighbors_unique)),
  to   = unlist(rook_neighbors_unique)
)

# Create a grid of all (cell_id, year)
years <- unique(cell_data$year)
cell_year <- CJ(id = id_order, year = years)

# Join edges to cell_year: replicate neighbor relationships for each year
neighbor_pairs <- edges[cell_year, on = .(from = id), allow.cartesian = TRUE]
setnames(neighbor_pairs, c("from", "to", "id", "year"), c("from", "to", "cell_id", "year"))
neighbor_pairs[, from := NULL]

# Add neighbor values from cell_data
cell_data_keyed <- copy(cell_data)
setkey(cell_data_keyed, id, year)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  # Join neighbor variable values
  neighbor_pairs <- neighbor_pairs[cell_data_keyed[, .(id, year, val = get(var_name))],
                                   on = .(to = id, year), nomatch = 0]

  # Compute stats: max, min, mean by (cell_id, year)
  stats <- neighbor_pairs[, .(
    max_val = max(val, na.rm = TRUE),
    min_val = min(val, na.rm = TRUE),
    mean_val = mean(val, na.rm = TRUE)
  ), by = .(cell_id, year)]

  # Merge back to cell_data
  setnames(stats, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  cell_data <- merge(cell_data, stats, by.x = c("id", "year"), by.y = c("cell_id", "year"), all.x = TRUE)
}

# Final optimized cell_data now contains neighbor features
```

---

### **Why This Is Faster**
- Eliminates nested `lapply` over millions of rows.
- Uses vectorized joins and aggregations in `data.table`.
- Single pass per variable for aggregation.
- Memory footprint is minimized by using long format rather than huge nested lists.

---

**Expected performance:**  
- On 6.46M rows, `data.table` joins and aggregations with ~1.4M edges are feasible on a 16 GB laptop.
- Estimated runtime: minutes to a few hours (depending on disk speed), not 86+ hours.

This preserves the **trained Random Forest model** and numerical estimand, only optimizing feature engineering.