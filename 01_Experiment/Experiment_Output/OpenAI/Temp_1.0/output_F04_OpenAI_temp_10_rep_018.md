 **Diagnosis**  
The bottleneck is the construction and repeated traversal of `neighbor_lookup`. The current implementation:  
- Uses `lapply` for **every row (~6.46M)** → extremely slow in R due to interpreter overhead.  
- Builds a large list (`neighbor_lookup`) of length equal to rows (6.46M), huge memory footprint (~hundreds of MBs) and expensive GC.  
- Calls `compute_neighbor_stats` five times sequentially traversing the same lookup repeatedly → multiplies overhead.  

**Root cause:** Neighbor feature calculation is **row-wise in pure R loops**; no vectorization, no data.table use, and duplicate work per variable.  

---

### **Optimization Strategy**
1. **Precompute neighbor relationships once in vectorized long format** instead of storing 6.46M neighbor lists.  
   - Convert cell-level `id` and `year` to one unique key index.  
   - Expand neighbors into a long table:  
     `source_row, neighbor_row`.  
2. Join values of source and neighbor in **`data.table`** and compute `max`, `min`, `mean` grouped by `source_row` using **fast aggregation**.  
3. Repeat **only aggregation per variable** without regenerating structure.  
4. Do not retrain Random Forest; only enrich `cell_data`.  
5. Use `data.table` for efficient memory and speed (highly recommended vs. base R).  

Expected speed-up: **hours → minutes** on 6.5M rows if aggregated in compiled code.

---

## **Optimized R Code (data.table)**

```r
library(data.table)

# Convert to data.table
dt <- as.data.table(cell_data)
dt[, row_id := .I]  # unique row index

# STEP 1: Build long neighbor table once
# "id_order" is vector of all cell IDs; rook_neighbors_unique is list of neighbors per ref_idx
neighbor_dt_list <- vector("list", length(id_order))

for (i in seq_along(id_order)) {
  cell_id <- id_order[i]
  neigh_ids <- id_order[rook_neighbors_unique[[i]]]
  if (length(neigh_ids) > 0) {
    neighbor_dt_list[[i]] <- data.table(source_id = cell_id, neighbor_id = neigh_ids)
  }
}
neighbor_pairs <- rbindlist(neighbor_dt_list)
rm(neighbor_dt_list)  # free memory

# Add year to pair with repeated years (cross join across 28 years)
years <- unique(dt$year)
neighbor_pairs <- neighbor_pairs[, .(source_id, neighbor_id, year = rep(years, each = .N)), by = seq_len(nrow(neighbor_pairs))]
neighbor_pairs[, seq_len := NULL]

# STEP 2: Map to row indices in dt
setkey(dt, id, year)
neighbor_pairs[, source_row := dt[.(source_id, year), row_id]]
neighbor_pairs[, neighbor_row := dt[.(neighbor_id, year), row_id]]

# Drop unused columns
neighbor_pairs[, c("source_id", "neighbor_id") := NULL]

# Remove NAs (some combos may not exist for all years)
neighbor_pairs <- neighbor_pairs[!is.na(source_row) & !is.na(neighbor_row)]

# STEP 3: Compute neighbor stats for all needed variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var in neighbor_source_vars) {
  vals <- dt[[var]]
  
  # Create a working data.table of neighbor values
  tmp <- neighbor_pairs[, .(source_row, nval = vals[neighbor_row])]
  tmp <- tmp[!is.na(nval)]
  
  # Aggregate max, min, mean by source_row
  stats <- tmp[, .(
    paste0(var, "_nbr_max") = max(nval),
    paste0(var, "_nbr_min") = min(nval),
    paste0(var, "_nbr_mean") = mean(nval)
  ), by = source_row]
  
  # Merge back to dt
  setkey(stats, source_row)
  dt[stats, c(paste0(var, "_nbr_max"), paste0(var, "_nbr_min"), paste0(var, "_nbr_mean")) :=
       .(i.get(1L), i.get(2L), i.get(3L))]
}

# Remove helper columns
dt[, row_id := NULL]

# Done
cell_data <- as.data.frame(dt)
```

---

### **Why This Works**
- **Single neighbor mapping** built once, reused for all variables (no repeated lapply).  
- **Vectorized aggregation** via `data.table`, optimized in C.  
- Avoids materializing huge lists; uses integer row indices and joins.  
- Scales much better than millions of R closures.  

---

**Expected Impact:**  
- Original 86+ hours → estimated under 1 hour (often <15 minutes with `data.table` on a 16 GB laptop).  
- Preserves Random Forest model and final estimand exactly.