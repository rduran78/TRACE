 **Diagnosis**  
The bottleneck is the nested `lapply` operations over ~6.46M rows, repeatedly subsetting large vectors and computing max/min/mean for each row’s neighbors. This is highly inefficient in R due to high interpreter overhead, repeated memory allocations, and lack of vectorization. The current design does not scale for millions of rows on a standard laptop.

---

**Optimization Strategy**  
1. **Avoid per-row `lapply`**: Replace row-wise loops with a grouped aggregation approach.
2. **Reshape neighbor relationships into an edge list**: Each edge is `(cell_idx, neighbor_idx, year)`.
3. **Join values once and compute `max`, `min`, `mean` using `data.table` group operations**, which are highly optimized in C.
4. **Precompute for all variables in one pass** to minimize repeated joins.
5. **Preserve the estimand**: Results for each `(cell_id, year)` must match the original logic (ignoring NAs).

---

**Working R Code**

```r
library(data.table)

# Convert input to data.table
cell_dt <- as.data.table(cell_data)

# Add row index for fast joins
cell_dt[, row_idx := .I]

# Build neighbor edge list
# rook_neighbors_unique: list of neighbor indices for each id_order element
neighbor_edges <- rbindlist(
  lapply(seq_along(rook_neighbors_unique), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(
      from_id = id_order[i],
      to_id   = id_order[rook_neighbors_unique[[i]]]
    )
  })
)

# Expand for all years
years <- unique(cell_dt$year)
edges_expanded <- neighbor_edges[, .(id = from_id, neighbor_id = to_id), by = years]
setnames(edges_expanded, "years", "year")

# Map to row indices
id_year_key <- cell_dt[, .(id, year, row_idx)]
edges_expanded <- merge(edges_expanded, id_year_key, by = c("id","year"), all.x = TRUE)
setnames(edges_expanded, "row_idx", "row_from")
edges_expanded <- merge(edges_expanded, id_year_key, by.x = c("neighbor_id","year"),
                         by.y = c("id","year"), all.x = TRUE)
setnames(edges_expanded, "row_idx", "row_to")

# Keep only valid neighbor pairs
edges_expanded <- edges_expanded[!is.na(row_from) & !is.na(row_to)]

# Compute neighbor stats for each variable
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (v in vars) {
  val_dt <- cell_dt[, .(row_idx, val = get(v))]
  edges_vals <- merge(edges_expanded, val_dt, by.x = "row_to", by.y = "row_idx", all.x = TRUE)

  # Compute stats grouped by row_from
  stats <- edges_vals[, .(
    max_val = if (.N > 0) max(val, na.rm = TRUE) else NA_real_,
    min_val = if (.N > 0) min(val, na.rm = TRUE) else NA_real_,
    mean_val = if (.N > 0) mean(val, na.rm = TRUE) else NA_real_
  ), by = row_from]

  # Attach to main table
  setnames(stats, c("row_from","max_val","min_val","mean_val"),
           c("row_idx", paste0(v,"_nbr_max"), paste0(v,"_nbr_min"), paste0(v,"_nbr_mean")))
  cell_dt <- merge(cell_dt, stats, by = "row_idx", all.x = TRUE)
}

# Remove helper column
cell_dt[, row_idx := NULL]

# Result: cell_dt now contains neighbor features for all variables
```

---

**Expected Performance**  
- Complexity reduced from ~6.46M * `lapply` to efficient `data.table` joins and grouped aggregations.
- Memory footprint manageable because edge list (~1.37M pairs × 28 years ≈ 38M rows) is large but feasible on 16 GB with `data.table`.
- Wall time expected to drop from 86+ hours to under 1 hour (depending on disk I/O and CPU).

**Preserves**:  
- Original estimator (max, min, mean of rook neighbors by year).
- Pipeline compatibility (adds columns to `cell_dt` as before).
- Trained Random Forest model remains unchanged.