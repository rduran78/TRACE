 **Diagnosis**  
The bottleneck is the repeated R-level `lapply` loops across ~6.5 million rows for five variables. For each row, the code constructs neighbor stats by indexing into a large vector (`vals`). This approach causes excessive memory allocations and function calls, leading to extremely poor performance on large panels. The complexity is roughly `O(N × avg_degree)` where `N` is 6.5M, and each iteration runs pure R overhead.  

**Optimization Strategy**  
- Pre-flatten neighbor relationships into a long format (`from_row`, `to_row`) representing all directed edges with correct year alignment.
- Use **vectorized aggregation** (`data.table` or `collapse`) to compute `max`, `min`, `mean` in one grouped operation.
- Avoid R loops over 6.5M elements; leverage efficient C-backed grouping.
- Preserve original estimand by computing stats per `(cell_id, year)` from neighbors in the same year.

**Working R Code (data.table approach)**  
```r
library(data.table)

# Assume cell_data is a data.table
setDT(cell_data)

# Build neighbor lookup in long edge form (cell-year pairs)
# id_order: vector of ordered cell ids
# rook_neighbors_unique: list of neighbor indices (spdep nb)

# Step 1: Build edges between cell ids (static)
edges <- data.table(
  from_id = rep(id_order, times = lengths(rook_neighbors_unique)),
  to_id   = unlist(id_order[unlist(rook_neighbors_unique)], use.names = FALSE)
)

# Step 2: Expand for years by joining on cell-year rows
# cell_data has columns: id, year
# Map each from_id/year to to_id/year
edges_exp <- cell_data[, .(id, year, row_id = .I)][edges, on = .(id = from_id), allow.cartesian = TRUE]
setnames(edges_exp, "row_id", "from_row")

edges_exp <- cell_data[, .(id, year, row_id = .I)][edges_exp, on = .(id = to_id, year), allow.cartesian = TRUE]
setnames(edges_exp, "row_id", "to_row")

# Drop edges where no matching year neighbor
edges_exp <- edges_exp[!is.na(to_row), .(from_row, to_row)]

# Free memory from intermediate mapping columns
rm(edges)

# Step 3: Compute neighbor stats for each variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  vals <- cell_data[[var_name]]
  tmp  <- data.table(
    from_row = edges_exp$from_row,
    val      = vals[edges_exp$to_row]
  )

  # Aggregate
  agg <- tmp[, .(
    max_val = max(val, na.rm = TRUE),
    min_val = min(val, na.rm = TRUE),
    mean_val = mean(val, na.rm = TRUE)
  ), by = from_row]

  # Merge back to cell_data
  setorder(agg, from_row)
  set(cell_data, i = agg$from_row, j = paste0(var_name, "_nbr_max"), agg$max_val)
  set(cell_data, i = agg$from_row, j = paste0(var_name, "_nbr_min"), agg$min_val)
  set(cell_data, i = agg$from_row, j = paste0(var_name, "_nbr_mean"), agg$mean_val)
}

# At this stage, cell_data includes neighbor summary columns for RF prediction
```

**Why This is Faster**  
- Only one pass per variable using fast in-memory grouping.
- Avoids millions of small list allocations.
- Data.table operations are implemented in C and optimized for large datasets.
- Memory footprint controlled because we only store `from_row` → `to_row` once for all years.

This approach preserves:
- **Same estimand**: stats computed over rook neighbors for the same year.
- **Pre-trained RF model**: untouched (only feature engineering changes).
- **Scalability**: reduces execution time from 86+ hours to likely under 1 hour on a 16 GB laptop (with efficient disk/memory handling).