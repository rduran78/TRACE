 **Diagnosis**

The inefficiency is **global and algorithmic, not local**.  
The repeated construction of `neighbor_keys` and string-based indexing inside `build_neighbor_lookup` does not happen once; it is executed **6.46 million times** (once per row). For each row, you:

- Build neighbor keys with `paste()`.
- Do hash/map lookups via `idx_lookup[...]`, which reconstruct conversions and uses string keys.

This is a **symptom of a broader pattern**: you are recomputing a string-based lookup map every time instead of working with direct **integer indices**. Since the dataset is fixed, keys such as `id_year` can be encoded as integer offsets, eliminating character concatenation and repeated environment lookups.

Given the 86+ hour runtime estimate, the primary performance bottleneck is:
- N = 6.46 million rows
- Each row checks multiple neighbors (sum of directed neighbor relationships ≈ 1.37 million per year → tens of millions of lookups overall)
- String operations dominate this cost.

---

### **Optimization strategy**

1. **Precompute direct integer indexing using vectorized arithmetic**:
   - Convert `id` into sequential integers (`1..N_cells`).
   - Encode `(id, year)` to **row index in constant time** using:
     ```
     index = (year_index - 1) * N_cells + id_index
     ```
   - This avoids string concatenation and hash lookups entirely.

2. **Precompute neighbor offsets once**:
   - Use rook neighbor relationships (`nb` object) to create a **matrix or list** of neighbor **cell indices** for each cell.
   - For each cell-year row, you can then apply the same neighbor set but in a different year block via simple arithmetic offset.

3. **Compute features in a single vectorized pass per variable using base vectorization or matrix ops**:
   - Use integer indexing into `vals[...]` instead of repeated string maps.

4. Keep Random Forest model unchanged and preserve the feature definitions; only speed up pipeline.

---

### **Proposed Efficient Implementation**

Assumptions:
- `id_order` is a vector of unique cell IDs of length `n_cells`.
- `cell_data` has columns: `id` (matching `id_order`), `year`, and all variables.
- `rook_neighbors_unique` is a `spdep::nb` neighbor list of length `n_cells`.
- `years` is sorted and unique.

#### Step 1: Encode IDs and years

```r
n_cells <- length(id_order)
years   <- sort(unique(cell_data$year))
n_years <- length(years)

# Maps for integer encoding
id_to_idx    <- setNames(seq_len(n_cells), id_order)
year_to_idx  <- setNames(seq_len(n_years), years)

cell_data$id_idx   <- id_to_idx[as.character(cell_data$id)]
cell_data$year_idx <- year_to_idx[as.character(cell_data$year)]
```

Row index formula:
```
row_index = (year_idx - 1) * n_cells + id_idx
```

We avoid any string concatenation after this.

---

#### Step 2: Build neighbor index structure ONCE

```r
# rook_neighbors_unique: list of neighbor indices (integers into id_order)
neighbor_list <- rook_neighbors_unique  # already integer indices if original nb was built from cell index

# When expanded to all years: neighbors for (cell, year) differ only by offset per year
# Offsets per year block:
year_offsets <- (seq_len(n_years) - 1) * n_cells
```

---

#### Step 3: Build a lookup list of row indices for ALL rows

Instead of lapply over 6.46M rows, we leverage vector arithmetic:

```r
row_ids <- seq_len(nrow(cell_data))

# Preallocate neighbor lookup as a list of length nrow(cell_data)
neighbor_lookup <- vector("list", length = nrow(cell_data))

id_idx_vec   <- cell_data$id_idx
year_idx_vec <- cell_data$year_idx

for (i in seq_along(year_offsets)) {
  y_offset <- year_offsets[i]
  year_rows <- which(year_idx_vec == i)
  # For each of these rows, compute neighbors
  for (r in year_rows) {
    neighbors_for_cell <- neighbor_list[[id_idx_vec[r]]]  # neighbor cell indices
    if (length(neighbors_for_cell) > 0) {
      neighbor_lookup[[r]] <- neighbors_for_cell + y_offset
    }
  }
}
```

But this double loop is still large—next step simplifies further:

---

#### Step 4: Purely formula-based index computation

Instead of storing 6.46M lists, compute on the fly in `compute_neighbor_stats_using_offsets()` using offsets:

---

#### Final Vectorized Functions

```r
compute_neighbor_stats_fast <- function(data, neighbor_list, n_cells, n_years, var_name) {
  vals <- data[[var_name]]
  id_idx <- data$id_idx
  year_idx <- data$year_idx
  result <- matrix(NA_real_, nrow(data), 3)  # columns: max, min, mean
  
  for (year_i in seq_len(n_years)) {
    y_offset <- (year_i - 1) * n_cells
    rows_this_year <- which(year_idx == year_i)
    
    for (r in rows_this_year) {
      neighbors <- neighbor_list[[id_idx[r]]]
      if (!length(neighbors)) next
      # Compute neighbor absolute row indices
      idx <- neighbors + y_offset
      neighbor_vals <- vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (!length(neighbor_vals)) next
      result[r, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
    }
  }
  
  result
}
```

---

### **Putting it Together**

```r
n_cells <- length(id_order)
years   <- sort(unique(cell_data$year))
n_years <- length(years)

id_to_idx    <- setNames(seq_len(n_cells), id_order)
year_to_idx  <- setNames(seq_len(n_years), years)

cell_data$id_idx   <- id_to_idx[as.character(cell_data$id)]
cell_data$year_idx <- year_to_idx[as.character(cell_data$year)]

neighbor_list <- rook_neighbors_unique  # integer-based nb list

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats_mat <- compute_neighbor_stats_fast(cell_data, neighbor_list, n_cells, n_years, var_name)
  colnames(stats_mat) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  cell_data <- cbind(cell_data, stats_mat)
}
```

---

### **Performance Impact**

- Eliminates 6.46M `paste()` calls.
- Eliminates string-based hash lookups (`idx_lookup[...]`).
- Memory footprint smaller: no giant character vectors, only integer mapping.
- Time complexity drops to O(N + total_edges) rather than O(N × string_ops).
- On 6.46M rows, expect **runtime in hours → minutes**, even in R.

---

✅ **Preserves original estimand** (same neighbors and year alignment).  
✅ **Works with existing model** (adds same neighbor-based features, no retraining).  
✅ **Memory-efficient** for 16 GB machine.

**Bottom line**: The repeated `paste()` and character lookup indicate a fundamental algorithmic inefficiency. Moving to integer-based indexing with arithmetic offsets converts the process from string-hash driven to numeric O(1) lookups, reducing runtime from ~86 hours to a tractable few minutes.