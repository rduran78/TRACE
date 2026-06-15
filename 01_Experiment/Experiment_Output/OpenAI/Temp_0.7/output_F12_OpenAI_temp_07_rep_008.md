 **Diagnosis:**  
The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` is the main bottleneck is **incorrect**. Binding a few million rows after computing statistics is relatively cheap compared to the cost of repeatedly scanning neighbor indexes and subsetting vectors inside the `lapply` for 6.46M rows. The **true bottleneck** is the repeated R-level iteration and vector subsetting (`vals[idx]`) inside `compute_neighbor_stats()`. Each iteration incurs overhead in R's interpreter and repeated memory allocations. With ~6.46M rows × 5 variables, this dominates runtime.

---

### **Optimization Strategy**
- Precompute neighbor indices once (already done via `neighbor_lookup`).
- Replace the R `lapply` loop with a **vectorized or compiled approach** (e.g., `data.table`, `Rcpp`, or `vapply` with preallocated matrix).
- Compute all neighbor stats in **C-level loops** via `Rcpp` for speed.
- Avoid repeated copying of `vals[idx]`; instead, iterate over numeric vectors directly in compiled code.

---

### **Optimized Rcpp Implementation**

**Step 1: Install Rcpp if needed**
```r
install.packages("Rcpp")
```

**Step 2: Implement C++ function**
```r
library(Rcpp)

cppFunction('
NumericMatrix computeNeighborStatsCpp(List neighbor_lookup, NumericVector vals) {
  int n = neighbor_lookup.size();
  NumericMatrix result(n, 3); // cols: max, min, mean

  for (int i = 0; i < n; i++) {
    IntegerVector idx = neighbor_lookup[i];
    int m = idx.size();

    double maxv = R_NegInf;
    double minv = R_PosInf;
    double sumv = 0.0;
    int count = 0;

    for (int j = 0; j < m; j++) {
      int pos = idx[j] - 1; // R is 1-based
      if (pos >= 0 && pos < vals.size()) {
        double v = vals[pos];
        if (!R_IsNA(v)) {
          if (v > maxv) maxv = v;
          if (v < minv) minv = v;
          sumv += v;
          count++;
        }
      }
    }

    if (count == 0) {
      result(i, 0) = NA_REAL;
      result(i, 1) = NA_REAL;
      result(i, 2) = NA_REAL;
    } else {
      result(i, 0) = maxv;
      result(i, 1) = minv;
      result(i, 2) = sumv / count;
    }
  }
  return result;
}
')
```

---

### **Step 3: Replace compute_neighbor_stats**
```r
compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  computeNeighborStatsCpp(neighbor_lookup, vals)
}
```

---

### **Step 4: Apply in loop**
```r
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  colnames(stats) <- paste(var_name, c("max", "min", "mean"), sep = "_")
  cell_data <- cbind(cell_data, stats)
}
```

---

### **Expected Performance Gain**
- Eliminates 6.46M R-level loops → replaced by a single efficient C++ loop.
- Drastically reduces overhead from function calls and memory allocations.
- Brings computation down from **86+ hours to a few minutes** on a standard laptop.

---

✅ **Summary:** Reject colleague’s diagnosis. The real bottleneck is R-level per-row neighbor aggregation inside `compute_neighbor_stats()`, not `do.call(rbind, ...)`. Correct fix is to move that logic into a compiled routine (Rcpp) for efficient looping and aggregation while preserving the original estimand and trained model.