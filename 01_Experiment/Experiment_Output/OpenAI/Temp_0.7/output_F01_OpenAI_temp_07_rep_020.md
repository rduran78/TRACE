 **Diagnosis:**  
The current bottleneck is the loop-based prediction on a very large dataset (`pred_db` and `joined_data`) with many predictors across multiple years. Each iteration filters and predicts year by year, which is slow and memory-intensive. Additionally, `dplyr` operations on large tables and repeated subsetting add overhead.

---

### **Optimization Strategy**
1. **Switch to `data.table` for fast filtering and assignment** – avoids overhead from `dplyr`.
2. **Chunk predictions** to prevent loading the entire dataset in memory at once.
3. **Avoid creating intermediate large objects** – predict and write results incrementally.
4. **Leverage `parallel` prediction** for speed if multiple CPU cores are available.
5. **Preserve existing models** – no retraining.

---

### **Optimized R Code**
```r
library(data.table)
library(parallel)

# Convert to data.table
setDT(pred_db)
setDT(joined_data)

# List of years present in both data and models
years <- intersect(unique(pred_db$year), names(rf_models_per_year))

# Prepare output file
out_file <- "RF_imputated_db.csv"
fwrite(pred_db[0], out_file)  # write header only

# Function to process one year in chunks
predict_year <- function(year) {
  cat("Predicting for year", year, "\n")
  
  rf_model <- rf_models_per_year[[as.character(year)]]
  
  test_set <- joined_data[year == !!year]
  idx <- which(pred_db$year == year)
  
  # Chunk size to control memory
  chunk_size <- 50000
  preds <- numeric(length(idx))
  
  for (i in seq(1, nrow(test_set), by = chunk_size)) {
    rows <- i:min(i + chunk_size - 1, nrow(test_set))
    preds[rows] <- predict(rf_model, newdata = test_set[rows])
  }
  
  pred_db[idx, consolidated := preds]
  
  # Append to file incrementally (optional)
  fwrite(pred_db[idx], out_file, append = TRUE)
  
  invisible(NULL)
}

# Use parallel processing if desired
mclapply(years, predict_year, mc.cores = detectCores() - 1)

cat("Done. Predictions saved to", out_file, "\n")
```

---

### **Why This Works**
- `data.table` drastically speeds up filtering and assignment.
- Chunked prediction avoids memory overflow.
- Parallelization utilizes available CPU cores efficiently.
- Incremental writing prevents holding the full output in memory.

**Result:** Computation becomes feasible on a normal machine without retraining models.