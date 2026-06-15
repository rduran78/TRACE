 **Diagnosis**  
The bottleneck lies in:  
1. **Repeated `lapply` and object copying** over 6.46M rows for neighbor stats. This is highly inefficient in R due to memory allocations.  
2. **Building neighbor lookup repeatedly** (list of indices per row). For large panels, this consumes excessive memory and slows computation.  
3. **Random Forest inference**: `predict()` on millions of rows in small chunks repeatedly can be slow; loading the model multiple times is worse.  
4. **Data structure**: Using lists and repeated `paste()` calls creates huge overhead.

---

### **Optimization Strategy**
- **Precompute neighbor indices once** in a flat, vectorized format (e.g., `data.table` or matrix form).  
- **Vectorize neighbor feature computation** using `data.table` joins instead of looping.  
- **Use memory-efficient `data.table` for panel data** instead of base R.  
- **Batch prediction**: Load the Random Forest model once and predict in large chunks (e.g., 500k rows per batch).  
- Avoid repeatedly copying large data frames; update in place.  
- Consider parallelization via `parallel` or `future.apply`.  

---

### **Working R Code**

```r
library(data.table)
library(randomForest)

# Convert to data.table
cell_dt <- as.data.table(cell_data)

# Precompute key for fast joining
cell_dt[, key := paste(id, year, sep = "_")]
setkey(cell_dt, key)

# Flatten neighbors: for each id, store neighbors
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  ref_id <- id_order[i]
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = ref_id,
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Expand to panel: join with years
years <- unique(cell_dt$year)
neighbor_panel <- neighbor_dt[, .(neighbor_id = neighbor_id), by = .(id)]
neighbor_panel <- neighbor_panel[, .(id = rep(id, length(years)),
                                     year = rep(years, each = .N),
                                     neighbor_id = neighbor_id), by = id]
neighbor_panel[, key := paste(neighbor_id, year, sep = "_")]

# Join to get neighbor values
compute_neighbor_stats_dt <- function(var) {
  vals <- cell_dt[, .(key, val = get(var))]
  merged <- neighbor_panel[vals, on = "key"]
  merged <- merged[!is.na(val)]
  # Aggregate stats
  merged[, .(max = max(val), min = min(val), mean = mean(val)), by = .(id, year)]
}

# Compute all neighbor stats efficiently
for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_dt(var_name)
  setnames(stats, c("id", "year", paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))))
  cell_dt[stats, on = .(id, year), `:=`(
    (paste0(var_name, "_nb_max")) = i[[paste0(var_name, "_nb_max")]],
    (paste0(var_name, "_nb_min")) = i[[paste0(var_name, "_nb_min")]],
    (paste0(var_name, "_nb_mean")) = i[[paste0(var_name, "_nb_mean")]]
  )]
}

# Prepare predictors
predictors <- setdiff(names(cell_dt), c("id", "year", "gdp")) # adjust target name

# Load Random Forest model once
rf_model <- readRDS("rf_model.rds")

# Batch prediction
batch_size <- 500000
n <- nrow(cell_dt)
preds <- numeric(n)
for (start in seq(1, n, by = batch_size)) {
  end <- min(start + batch_size - 1, n)
  preds[start:end] <- predict(rf_model, newdata = cell_dt[start:end, ..predictors])
}

cell_dt[, gdp_pred := preds]
```

---

### **Why This Works**
- **Joins instead of nested lapply** reduce complexity from O(n × neighbors) loops to efficient indexed operations.  
- **`data.table` aggregation** avoids repeated copying and uses optimized C backend.  
- **Random Forest predict** in chunks prevents memory overflow while minimizing repeated overhead.  

This approach should reduce runtime from **86+ hours to a few hours** depending on disk speed and CPU.