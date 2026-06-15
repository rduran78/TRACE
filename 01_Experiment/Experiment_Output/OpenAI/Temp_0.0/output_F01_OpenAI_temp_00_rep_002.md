 **Diagnosis**  
The bottleneck is the loop predicting year by year on large data frames using `dplyr::filter` and base assignment. This repeatedly subsets and copies large objects in memory, which is inefficient for hundreds of thousands of rows. Additionally, `predict()` on large chunks without batching can cause memory spikes.

---

**Optimization Strategy**  
1. **Use `data.table` for fast filtering and assignment** instead of `dplyr`.
2. **Process in chunks** to avoid loading all rows for a year at once if memory is tight.
3. **Pre-allocate and work in-place** to minimize copies.
4. **Avoid repeated coercion** by ensuring `year` is character or factor consistently.
5. **Parallelize predictions** if multiple cores are available (optional).

---

**Optimized R Code**

```r
library(data.table)

# Convert to data.table for efficiency
setDT(pred_db)
setDT(joined_data)

# Ensure year is character for matching
pred_db[, year := as.character(year)]
joined_data[, year := as.character(year)]

# Pre-allocate consolidated column if not present
if (!"consolidated" %in% names(pred_db)) {
  pred_db[, consolidated := NA_real_]
}

# Loop through available models only
for (yr in names(rf_models_per_year)) {
  cat("Predicting for year", yr, "\n")
  
  # Subset test set efficiently
  test_set <- joined_data[year == yr]
  
  if (nrow(test_set) > 0) {
    rf_model <- rf_models_per_year[[yr]]
    
    # Predict in chunks to save memory
    chunk_size <- 50000
    preds <- numeric(nrow(test_set))
    for (i in seq(1, nrow(test_set), by = chunk_size)) {
      idx <- i:min(i + chunk_size - 1, nrow(test_set))
      preds[idx] <- predict(rf_model, newdata = test_set[idx])
    }
    
    # Assign predictions back efficiently
    pred_db[year == yr, consolidated := preds]
  } else {
    cat("Warning: No data for year", yr, "\n")
  }
}

# Write output
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Why This Works**  
- `data.table` filtering and assignment are in-place and memory-efficient.
- Chunked prediction prevents memory overload.
- Avoids unnecessary loops over years without models.
- Scales better for hundreds of thousands of rows on a personal machine.