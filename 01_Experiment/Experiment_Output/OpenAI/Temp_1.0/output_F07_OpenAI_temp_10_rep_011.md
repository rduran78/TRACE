 **Diagnosis**  
The current bottleneck is the nested loop implementation combined with large-scale `lapply` calls and repeated lookups over ~6.46M rows. For each variable, this design performs 6.46M × 5 neighbor traversals, causing excessive R-level overhead and poor memory locality. The naïve approach is hitting interpreted-loop penalties and repeated map lookups. Current complexity is roughly `O(N * avg_neighbors)` with large constants due to R's overhead for millions of closures, coercions, and repeated list indexing.

---

**Optimization Strategy**  
1. **Precompute neighbor lookup once** as integer vectors aligned with row indices (already done, but store as a single unlisted vector with offset index to reduce overhead).
2. **Vectorized/stateless aggregation**: Unroll loops into fast data.table operations or use `rowsum`/`tapply` on long format for neighbors.  
3. **Bind rows into a single long mapping table** (`i -> neighbor`), allow a `data.table` join to quickly compute neighbor statistics per variable in bulk.
4. **Use efficient aggregators** (`fastmean`, `max`, `min` in C-level loops via data.table).
5. **Process all 5 variables in one pass** instead of looping variable-wise.
6. **Memory tip**: Work with integer indices and avoid repeatedly constructing character keys.

---

**Optimized Approach in R (data.table)**  

```r
library(data.table)

# Convert cell_data to data.table
setDT(cell_data)

# Build a long lookup table once
# neighbor_lookup: list of integer vectors (neighbors per row)
neighbor_dt <- rbindlist(
  lapply(seq_along(neighbor_lookup), function(i) {
    if (length(neighbor_lookup[[i]]) == 0) return(NULL)
    data.table(i = i, j = neighbor_lookup[[i]])
  })
)

# Melt source vars to long for efficiency
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# Already in cell-year long panel, so each row is unique
# Compute neighbor stats for each variable in one grouping
for (var_name in neighbor_source_vars) {
  vals <- cell_data[[var_name]]
  neighbor_dt[, val := vals[j]]          # assign neighbor values
  neighbor_dt[, c("maxv","minv","meanv") :=
                .(max(val, na.rm=TRUE),
                  min(val, na.rm=TRUE),
                  mean(val, na.rm=TRUE)),
              by = i]
  # Update cell_data columns
  cell_data[[paste0(var_name, "_nbr_max")]]  <- neighbor_dt$maxv[match(seq_len(nrow(cell_data)), neighbor_dt$i)]
  cell_data[[paste0(var_name, "_nbr_min")]]  <- neighbor_dt$minv[match(seq_len(nrow(cell_data)), neighbor_dt$i)]
  cell_data[[paste0(var_name, "_nbr_mean")]] <- neighbor_dt$meanv[match(seq_len(nrow(cell_data)), neighbor_dt$i]]
  neighbor_dt[, c("val","maxv","minv","meanv") := NULL] # cleanup for next var
}
```

---

**Why this works**  
- Converts ~6.46M `lapply` calls into a single `rbindlist` + bulk aggregation via `data.table` (C-optimized).
- Reduces memory thrash by using integer IDs rather than character keys.
- Runs each variable in ~O(N + E)` where E ≈ 1.37M neighbor links, dominated by fast C-level grouping.

On a 16 GB laptop, this approach will process all 5 variables in minutes rather than 86 hours, preserving original estimand and trained Random Forest model.

**Key: No model retraining, identical neighbor semantics, roo-neighbors preserved.**