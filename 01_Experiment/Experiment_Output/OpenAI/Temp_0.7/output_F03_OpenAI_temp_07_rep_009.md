 **Diagnosis**  
The major bottlenecks are:  
1. **Repeated lapply loops and object copying**: `compute_neighbor_stats` and `compute_and_add_neighbor_features` repeatedly traverse large lists and copy data frames.  
2. **List-of-lists neighbor lookup**: Building and iterating over large R lists (6.46M entries) is memory-heavy and slow.  
3. **Prediction loop inefficiency**: If predictions are applied in small chunks with `predict()`, overhead accumulates.  
4. **Random Forest inference on full data**: Large input matrix (~6.46M × 110) stresses memory; conversion to `data.frame` is costly.  

**Optimization Strategy**  
- Precompute neighbor relationships in a **long table** (edges) instead of list-of-lists.  
- Compute neighbor stats with **data.table** group operations, avoiding repeated `lapply`.  
- Use **matrix or data.table** for features to minimize copies.  
- Run predictions in **large batches** or all at once.  
- Keep model in memory; avoid reloading.  
- Use **parallelization** for neighbor stats and prediction if possible.  

---

### **Optimized Workflow**

```r
library(data.table)
library(ranger) # Fast RF inference (alternatively randomForest)

# Assume: cell_data (data.table) with columns id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer vectors (nb object)
# id_order: vector mapping index to id

# Convert to data.table
setDT(cell_data)

# Build long neighbor table once
build_neighbor_table <- function(id_order, neighbors) {
  src <- rep(seq_along(neighbors), lengths(neighbors))
  dst <- unlist(neighbors, use.names = FALSE)
  data.table(src_id = id_order[src], nb_id = id_order[dst])
}

neighbor_dt <- build_neighbor_table(id_order, rook_neighbors_unique)

# Join with years: expand for all years (cartesian join)
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[CJ(year = years), on = .(dummy = NULL)]
setnames(neighbor_dt, "V1", "year") # after CJ

# Compute neighbor stats efficiently
compute_neighbor_stats_dt <- function(cell_data, neighbor_dt, var_name) {
  # Prepare source values keyed by id-year
  vals <- cell_data[, .(id, year, value = get(var_name))]
  setkey(vals, id, year)
  
  # Join neighbors on nb_id-year to get neighbor values
  joined <- neighbor_dt[vals, on = .(nb_id = id, year), nomatch = 0]
  
  # Compute stats by src_id-year
  stats <- joined[, .(
    max_val = max(value, na.rm = TRUE),
    min_val = min(value, na.rm = TRUE),
    mean_val = mean(value, na.rm = TRUE)
  ), by = .(src_id, year)]
  
  setnames(stats, c("src_id", "year", paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))))
  stats
}

# Apply for all neighbor source vars
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
all_stats <- lapply(neighbor_source_vars, function(v) compute_neighbor_stats_dt(cell_data, neighbor_dt, v))
all_stats_dt <- Reduce(function(x, y) merge(x, y, by = c("src_id", "year"), all = TRUE), all_stats)

# Merge back into main cell_data
setnames(all_stats_dt, "src_id", "id")
cell_data <- merge(cell_data, all_stats_dt, by = c("id", "year"), all.x = TRUE)

# Prepare predictors as matrix for fast prediction
predictor_vars <- setdiff(names(cell_data), c("id", "year", "gdp")) # adjust target name
X <- as.matrix(cell_data[, ..predictor_vars])

# Load pre-trained Random Forest model
# model <- readRDS("rf_model.rds")

# Predict in large chunks using ranger (fastest)
pred <- predict(model, data = X)$predictions

cell_data[, gdp_pred := pred]
```

---

**Why This Is Faster**  
- Neighbor stats computed in **vectorized, grouped operations** via `data.table`, eliminating millions of `lapply` calls.  
- One **long join-based approach** for all neighbors and years.  
- Minimal data copies; merges happen in-place.  
- Random Forest inference uses **matrix input** and `ranger` for speed (or `predict` in bulk if using `randomForest`).  

**Expected Impact**  
From 86+ hours to a few hours (dominated by joins and prediction), well within 16 GB RAM if processed in chunks (e.g., per 5M rows for prediction).