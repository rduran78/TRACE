 **Diagnosis**  
The current approach is extremely slow because:  
- `lapply` over ~6.46M rows with nested lookups is highly inefficient in R.  
- Repeated string concatenation (`paste`) and `setNames` lookups inside the loop are expensive.  
- Memory overhead from millions of small lists and repeated indexing is huge.  
- No vectorization or compiled operations; everything is interpreted R code.  

**Optimization Strategy**  
- Precompute neighbor indices once as integer vectors without string concatenation.  
- Use `data.table` for fast joins and grouping.  
- Reshape data into a matrix by `id × year` for each variable, then compute neighbor stats using matrix operations.  
- Avoid per-row `lapply`; instead, use vectorized row/column operations.  
- Parallelize across variables or years if needed.  
- Keep the Random Forest model unchanged (we only compute features).  

**Working R Code** (efficient approach using `data.table` and matrix ops):  

```r
library(data.table)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer vectors (neighbors per id)
# id_order: vector of unique cell ids in same order as rook_neighbors_unique

setDT(cell_data)
setkey(cell_data, id, year)

# Precompute mapping from id to row index
id_to_idx <- setNames(seq_along(id_order), id_order)

# Convert rook_neighbors_unique into integer index list
neighbor_idx_list <- lapply(rook_neighbors_unique, function(nbs) id_to_idx[nbs])

# Create wide matrices for each variable: rows = id, cols = year
years <- sort(unique(cell_data$year))
n_id <- length(id_order)
n_year <- length(years)

make_matrix <- function(var) {
  m <- matrix(NA_real_, nrow = n_id, ncol = n_year,
              dimnames = list(id_order, years))
  vals <- cell_data[[var]]
  idx <- match(cell_data$id, id_order)
  yr_idx <- match(cell_data$year, years)
  m[cbind(idx, yr_idx)] <- vals
  m
}

vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
mat_list <- lapply(vars, make_matrix)
names(mat_list) <- vars

# Compute neighbor stats efficiently
compute_neighbor_stats_matrix <- function(m, neighbor_idx_list) {
  n_id <- nrow(m)
  n_year <- ncol(m)
  max_mat <- matrix(NA_real_, n_id, n_year)
  min_mat <- matrix(NA_real_, n_id, n_year)
  mean_mat <- matrix(NA_real_, n_id, n_year)
  
  for (i in seq_len(n_id)) {
    nbs <- neighbor_idx_list[[i]]
    if (length(nbs) == 0) next
    nb_vals <- m[nbs, , drop = FALSE]
    max_mat[i, ] <- apply(nb_vals, 2, max, na.rm = TRUE)
    min_mat[i, ] <- apply(nb_vals, 2, min, na.rm = TRUE)
    mean_mat[i, ] <- apply(nb_vals, 2, function(x) if (all(is.na(x))) NA_real_ else mean(x, na.rm = TRUE))
  }
  list(max = max_mat, min = min_mat, mean = mean_mat)
}

neighbor_features <- lapply(mat_list, compute_neighbor_stats_matrix, neighbor_idx_list)

# Convert back to long and merge
longify <- function(mat, varname, stat) {
  dt <- as.data.table(as.table(mat))
  setnames(dt, c("id", "year", "value"))
  dt[, year := as.integer(year)]
  setnames(dt, "value", paste0(varname, "_nb_", stat))
  dt
}

result_list <- list()
for (var in vars) {
  for (stat in c("max", "min", "mean")) {
    mat <- neighbor_features[[var]][[stat]]
    result_list[[paste(var, stat, sep = "_")]] <- longify(mat, var, stat)
  }
}

neighbor_dt <- Reduce(function(x, y) merge(x, y, by = c("id", "year"), all = TRUE), result_list)

# Merge with original cell_data
cell_data <- merge(cell_data, neighbor_dt, by = c("id", "year"), all.x = TRUE)
```

**Why this works:**  
- Eliminates per-row loops and string operations.  
- Uses matrix operations and `apply` over small slices (neighbors × years).  
- Memory footprint is manageable: 5 matrices × (344k × 28) ≈ 48M doubles (~384 MB).  
- Expected runtime: minutes instead of 86+ hours on a 16 GB laptop.  

**Preserves:**  
- Original rook-neighbor topology.  
- Exact max, min, mean per neighbor set.  
- Original Random Forest model remains unchanged.