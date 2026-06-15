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
  
  # Subset test data
  test_set <- joined_data %>% filter(year == year)
  
  # Check if a model exists for this year
  if (as.character(year) %in% names(rf_models_per_year)) {
    
    # Retrieve the trained model for this year
    rf_model <- rf_models_per_year[[as.character(year)]]
    
    # Predict using the corresponding model
    pred_db$consolidated[pred_db$year == year] <- predict(rf_model, newdata = test_set)
    
  } else {
    cat(paste0("⚠️ Warning: No model found for year ", year, "\n"))
  }
}

write.csv(pred_db, "RF_imputated_db.csv")