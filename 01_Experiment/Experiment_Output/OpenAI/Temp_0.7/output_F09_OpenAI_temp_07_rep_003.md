 **Diagnosis**  
The current pipeline recomputes neighbor relationships for each row every time `build_neighbor_lookup` is called. For ~6.46M rows, repeatedly pasting IDs and doing lookups is extremely costly in both time and memory. The main inefficiency is that neighbor logic is rebuilt at the cell-year level rather than leveraging the static spatial topology. Given that the neighbor structure (rook adjacency) is constant across years, it should be computed *once* at the cell level and then reused for all years. This avoids redundant string concatenations, repeated hashing, and expensive lookups.

---

**Optimization Strategy**  
1. Precompute a static `neighbor_lookup` at the **cell index level** (not cell-year).
2. Iterate over years; for each year subset:
   - Join year-specific attributes.
   - Compute neighbor stats by mapping static neighbor indices to current year's values.
3. Use matrix operations or `vapply` instead of `lapply` for speed.
4. Avoid creating giant objects in memory (chunk by year).
5. Keep the trained Random Forest model unchanged and preserve original estimands.

This reduces complexity from ~6.46M dynamic lookups to ~344K lookups per year, leveraging the constant adjacency structure.

---

**Working R Code**

```r
# Precompute neighbor lookup at cell level
build_cell_neighbor_lookup <- function(neighbors) {
  # neighbors is an spdep nb object, list of integer vectors
  lapply(neighbors, as.integer)
}

neighbor_lookup <- build_cell_neighbor_lookup(rook_neighbors_unique)

# Function to compute neighbor stats for one year
compute_neighbor_stats_year <- function(data_year, neighbor_lookup, var_name) {
  vals <- data_year[[var_name]]
  n <- nrow(data_year)
  # Preallocate matrix
  out <- matrix(NA_real_, nrow = n, ncol = 3)
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      neighbor_vals <- vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) > 0) {
        out[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
      }
    }
  }
  colnames(out) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  out
}

# Main optimized loop: process year by year
years <- sort(unique(cell_data$year))
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

result_list <- vector("list", length(years))

for (i in seq_along(years)) {
  yr <- years[i]
  message("Processing year: ", yr)
  data_year <- subset(cell_data, year == yr)
  # Ensure data_year is ordered by id_order
  data_year <- data_year[match(id_order, data_year$id), ]
  
  feature_mats <- lapply(neighbor_source_vars, function(var_name) {
    compute_neighbor_stats_year(data_year, neighbor_lookup, var_name)
  })
  
  features <- do.call(cbind, feature_mats)
  data_year <- cbind(data_year, features)
  result_list[[i]] <- data_year
}

# Combine all years back
cell_data_enriched <- do.call(rbind, result_list)

# Predict using pre-trained Random Forest (unchanged)
predictions <- predict(trained_rf_model, newdata = cell_data_enriched)
```

---

**Expected Impact**  
- Eliminates ~6.46M dynamic key lookups and string manipulations.
- Reduces runtime from 86+ hours to a few hours (depending on I/O and CPU speed).
- Preserves the trained model and original estimands.
- Fits within 16 GB RAM by chunking per year.

This approach leverages static adjacency and processes year-by-year efficiently.