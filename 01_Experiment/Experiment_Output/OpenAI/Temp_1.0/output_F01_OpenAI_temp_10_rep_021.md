 **Diagnosis**  
The main performance bottleneck comes from:  
- Iterating through years with a `for` loop and repeatedly subsetting `joined_data` and `pred_db`.  
- Using `dplyr::filter` inside the loop for large datasets (slow for hundreds of thousands of rows).  
- Large object copies due to `tidyverse` operations.  
- Single-threaded `predict` calls on large chunks instead of batching.  

The models are fixed, so we cannot retrain, but predictions can be vectorized and memory-optimized.

---

### **Optimization Strategy**
1. **Convert to `data.table`** for efficient subsetting and updates (in-place operations, no full copies).
2. **Pre-split data by year** using `split` or `data.table` grouping to avoid repetitive filtering.
3. **Parallelize predictions** across years with `parallel::mclapply` or `future.apply` (if OS supports).
4. **Batch predictions** for very large datasets by splitting each year’s subset into smaller chunks.
5. Use `fwrite` from `data.table` instead of `write.csv` for speed.

---

### **Optimized Working R Code**
```r
library(data.table)
library(parallel)

# Ensure pred_db and joined_data are data.table objects
setDT(pred_db)
setDT(joined_data)

# Pre-split joined_data by year for faster access
data_by_year <- split(joined_data, joined_data$year)

# Number of cores to use (adjust as needed)
n_cores <- max(1, detectCores() - 1)

# Function to predict for one year with batching
predict_year <- function(year) {
  if (!as.character(year) %in% names(rf_models_per_year)) {
    cat("Warning: No model found for year", year, "\n")
    return(NULL)
  }
  
  rf_model <- rf_models_per_year[[as.character(year)]]
  test_set <- data_by_year[[as.character(year)]]
  
  # If test_set is huge, batch it (e.g., 50,000 rows per batch)
  batch_size <- 50000
  n <- nrow(test_set)
  preds <- numeric(n)
  
  for (i in seq(1, n, by = batch_size)) {
    j <- min(i + batch_size - 1, n)
    preds[i:j] <- predict(rf_model, newdata = test_set[i:j])
  }
  
  list(year = year, preds = preds, idx = which(pred_db$year == year))
}

# Run predictions in parallel
results <- mclapply(unique(pred_db$year), predict_year, mc.cores = n_cores)

# Assign predictions back without full copies
for (res in results) {
  if (!is.null(res)) {
    pred_db$consolidated[res$idx] <- res$preds
  }
}

# Write out efficiently
fwrite(pred_db, "RF_imputated_db.csv")
```

---

### **Why This Works**
- **`data.table`** eliminates unnecessary copying → huge speed gain.
- **Batching** avoids memory spikes.
- **Parallelization** utilizes multiple cores for independent years.
- **`fwrite`** saves output 10x faster.

**Expected improvement:** Hours reduced to minutes depending on hardware and dataset size, while keeping Random Forest models unchanged.