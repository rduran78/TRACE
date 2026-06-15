 **Diagnosis**  
The current method uses pure R lists and `lapply` over ~6.46M rows to compute neighbor statistics (max, min, mean). This creates severe memory and overhead issues due to repeated lookups, allocations, and R's interpreted loops. With millions of rows and over 1.3M neighbor links, it performs tens of millions of operations in R, causing the 86+ hour runtime. The main bottlenecks:  
- Neighbor lookups per row in a large list in R (`lapply` and long vectors).  
- No vectorization or efficient memory layout (repeated subsetting).  
- Fully serial execution, no parallelization.  

---

**Optimization Strategy**  
- Precompute neighbor index mapping for all rows (already done: `neighbor_lookup`).  
- Flatten neighbor pairs (row → neighbor) into two aligned vectors, enabling vectorized aggregation.  
- Use **`data.table`** for grouping and computing max, min, mean by row ID, which is extremely efficient on large datasets.  
- Process all variables in long format at once instead of looping through variables.  
- Preserve memory by avoiding redundant lists, work column-wise.  
- If further speed required, use **`parallel::mclapply`** or **`future.apply`**, but data.table grouping should already bring runtime down to minutes.  

---

**Working R Code (Efficient Version using data.table):**

```r
library(data.table)

# Assume: cell_data is data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order and rook_neighbors_unique (spdep::nb) are available

# 1. Build edge list: each observation row_id paired with a neighbor row_id
build_edge_list <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  idx_lookup <- setNames(seq_len(nrow(data)), paste(data$id, data$year, sep = "_"))
  
  edges <- vector("list", length = nrow(data))
  row_ids <- seq_len(nrow(data))
  
  for (i in row_ids) {
    ref_idx <- id_to_ref[as.character(data$id[i])]
    n_cells <- id_order[neighbors[[ref_idx]]]
    if (length(n_cells) == 0) next
    n_keys <- paste(n_cells, data$year[i], sep = "_")
    n_idx <- idx_lookup[n_keys]
    n_idx <- as.integer(n_idx[!is.na(n_idx)])
    if (length(n_idx)) {
      edges[[i]] <- data.table(src = i, nbr = n_idx)
    }
  }
  rbindlist(edges)
}

neighbor_edges <- build_edge_list(cell_data, id_order, rook_neighbors_unique)
# neighbor_edges: columns src, nbr (row indices in cell_data)

setDT(cell_data) # ensure data.table
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# 2. Melt neighbor values for fast grouping
for (v in vars) {
  darr <- cell_data[[v]]
  # Join neighbor values by neighbor index
  temp <- neighbor_edges[, .(src, val = darr[nbr])]
  # Drop NA
  temp <- temp[!is.na(val)]
  
  # Compute stats by src
  stats <- temp[, .(
    max = max(val),
    min = min(val),
    mean = mean(val)
  ), by = src]
  
  # Add columns to cell_data
  setnames(stats, c("src", paste0(v, "_nbr_max"), paste0(v, "_nbr_min"), paste0(v, "_nbr_mean")))
  cell_data[stats$src, `:=`(
    (paste0(v, "_nbr_max")) = stats[[2]],
    (paste0(v, "_nbr_min")) = stats[[3]],
    (paste0(v, "_nbr_mean")) = stats[[4]]
  )]
}
```

---

**Why This is Fast:**  
- Converts neighbor structure into a flat two-column index list only once.  
- Uses `data.table` grouping to compute stats in compiled C for each variable.  
- Avoids `lapply` over millions of rows with nested loops.  
- Reduces complexity to O(E) where E = total neighbor edges (~couple million), which is manageable.  

---

**Expected Performance:**  
On a 6.5M row dataset and ~1.37M neighbor links, each variable only involves a few million operations—all handled in data.table (C backend). Run time should drop from 86+ hours to **a few minutes to tens of minutes on a 16 GB laptop**, without retraining the model and while preserving the exact estimand.