I am working with this R script for cell-level GDP prediction. The process is too slow or too memory intensive on a personal computer. Please review the code and propose a practical optimization strategy. Preserve the trained Random Forest models and do not retrain them.

```r
library(blockCV)
library(zoo)
library(LongituRF)
library(randomForest)
library(sf)
library(spdep)
library(tidyverse)
library(data.table)
library(terra)
library(plm)
library(utils)
library(fixest)
library(scales)
library(stringi)

prep_data <- st_read('/Volumes/Toshi 1Tb/Amaz/geographic_cell_data/geographic_cell_data.shp')

load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

pred_db$consolidated <- NA

for (year in unique(pred_db$year)) {
  cat(paste0("Predicting for year ", year, "\n"))

  test_set <- joined_data %>% filter(year == year)

  if (as.character(year) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(year)]]
    pred_db$consolidated[pred_db$year == year] <- predict(rf_model, newdata = test_set)
  } else {
    cat(paste0("Warning: No model found for year ", year, "\n"))
  }
}

write.csv(pred_db, "RF_imputated_db.csv")
```

Context:
- The data contain hundreds of thousands of cells per year and many predictor variables.
- The model is already trained.
- The goal is to make the process computationally feasible on a normal machine.

Provide a diagnosis, an optimization strategy, and working R code.
