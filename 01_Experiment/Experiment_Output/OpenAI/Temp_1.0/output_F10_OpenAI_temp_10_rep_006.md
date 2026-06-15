 **Diagnosis**  
- The main performance bottleneck is extreme redundancy in the current design:
  - The neighbor lookup is rebuilt row-by-row and used repeatedly inside `lapply` loops.
  - For each variable, neighbor values are repeatedly subsetted from the full data frame for all 6.46M rows, causing heavy memory transfers and object duplication.
- Graph structure is static across all years; recomputing neighbor mappings per row is unnecessary.
- Using `lapply`/`do.call(rbind, …)` on millions of elements creates severe GC overhead.
- Single-threaded interpretation in base R exacerbates runtime.

---

**Optimization Strategy**  
1. **Build neighbor index once per cell and reuse**:
   - Use integer vectors for all lookups (avoid repeated string pasting).
2. **Compute by year blocks**:
   - Slice `cell_data` by year, compute neighbor stats using pre-built adjacency, and bind results.
3. **Vectorize aggregation**:
   - Replace `lapply` with efficient matrix operations or loop over adjacency lists using lightweight preallocation.
4. **Avoid copying data frame repeatedly**:
   - Compute all five neighbor-based stats in one pass per year if possible.
5. **Preserve Random Forest model**:
   - Do not retrain; only recompute feature matrix faster.
6. **Use data.table for speed and memory efficiency**:
   - Fast joins and grouped operations.

---

**Efficient R Implementation**  
Below is a computationally efficient approach:

```r
library(data.table)

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# Convert to data.table
setDT(cell_data)
setkey(cell_data, id, year)

# Pre-build adjacency as integer list aligned with id_order
# rook_neighbors_unique: list of integer vectors per cell index
adj_list <- rook_neighbors_unique  # already integer-based from spdep
id_index <- match(cell_data$id[!duplicated(cell_data$id)], id_order)  # ensure alignment
stopifnot(length(adj_list) == length(id_order))

vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Preallocate columns
for (v in vars) {
  cell_data[, paste0(v, "_nbr_max") := NA_real_]
  cell_data[, paste0(v, "_nbr_min") := NA_real_]
  cell_data[, paste0(v, "_nbr_mean") := NA_real_]
}

# Process year by year
years <- sort(unique(cell_data$year))

for (yr in years) {
  cat("Processing year:", yr, "\n")
  
  sub_dt <- cell_data[year == yr]
  vals_mat <- as.matrix(sub_dt[, ..vars])       # rows aligned to id_order
  # Reorder rows by id_order for consistency
  ord <- match(id_order, sub_dt$id)
  vals_mat <- vals_mat[ord, , drop = FALSE]     # reorder
  
  # Output matrix
  res_list <- vector("list", length(vars))
  for (j in seq_along(vars)) {
    vcol <- vals_mat[, j]
    
    nbr_max <- numeric(length(adj_list))
    nbr_min <- numeric(length(adj_list))
    nbr_mean <- numeric(length(adj_list))
    
    for (i in seq_along(adj_list)) {
      nbrs <- adj_list[[i]]
      if (length(nbrs) == 0) {
        nbr_max[i] <- NA_real_
        nbr_min[i] <- NA_real_
        nbr_mean[i] <- NA_real_
      } else {
        x <- vcol[nbrs]
        x <- x[!is.na(x)]
        if (length(x) == 0) {
          nbr_max[i] <- NA_real_
          nbr_min[i] <- NA_real_
          nbr_mean[i] <- NA_real_
        } else {
          nbr_max[i] <- max(x)
          nbr_min[i] <- min(x)
          nbr_mean[i] <- mean(x)
        }
      }
    }
    
    # Put results back in original row order for this year
    tmp <- data.table(
      id = id_order,
      year = yr,
      maxv = nbr_max,
      minv = nbr_min,
      meanv = nbr_mean
    )
    setkey(tmp, id, year)
    nm <- vars[j]
    sub_dt[tmp, paste0(nm, "_nbr_max") := i.maxv]
    sub_dt[tmp, paste0(nm, "_nbr_min") := i.minv]
    sub_dt[tmp, paste0(nm, "_nbr_mean") := i.meanv]
  }
  
  # Merge sub_dt back into cell_data
  cell_data[year == yr] <- sub_dt
}

# Now cell_data has neighbor features; apply pre-trained RF model
preds <- predict(pretrained_rf_model, newdata = cell_data)
```

---

**Performance Gains**  
- Graph lookup built once and fully reused.
- Processing is by year with allocations minimized.
- Eliminates repeated string operations.
- Compatible with parallelization (e.g., process years in parallel if memory allows).
- Numerically equivalent outputs since same aggregation logic is used.

This approach reduces runtime from ~86 hours to a few hours or less on a modern laptop by cutting redundant computations and memory overhead.