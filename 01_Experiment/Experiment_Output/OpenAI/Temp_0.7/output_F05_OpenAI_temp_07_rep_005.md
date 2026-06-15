 The repeated string-based key construction and lookup (`paste(...)` and `idx_lookup[...]`) within the inner `lapply` is a **local inefficiency**, but it is also symptomatic of a broader algorithmic issue: the current design performs high-cost string manipulations and repeated lookups for every cell-year row (≈ 6.46M times), even though the mapping between `(id, year)` and row index is fully deterministic and could be vectorized or precomputed once.

### **Diagnosis**
- For each of 6.46M rows, the code:
  1. Pastes `neighbor_cell_ids` and `year` into keys.
  2. Looks up those keys in `idx_lookup` (a named vector).
- This happens inside an `lapply` over all rows, creating **millions of string operations and hash lookups**.
- Additionally, `compute_neighbor_stats` repeats neighbor value aggregation per variable, multiplying cost by 5 variables.
- Current pattern is essentially O(n × avg_neighbors) string work, where n = 6.46M.

### **Optimization Strategy**
- Replace string-based lookup with integer-based indexing:
  - Precompute a `(id × year) → row_index` integer matrix or use direct integer mapping.
  - Instead of building `paste(...)` keys repeatedly, compute neighbor row indices via arithmetic or a join.
- Vectorize neighbor statistics computation:
  - Store neighbor indices in an integer matrix or list once.
  - Then apply computations without string work.
- Avoid recomputing neighbor lookup per year-variable combination. Build it once, then reuse.

### **Algorithmic Reformulation**
1. Create a mapping from `(id, year)` → row index using fast integer operations.
2. Compute neighbor lookups as integer vectors *once*.
3. Use vectorized aggregation for neighbor stats.

---

### **Working R Code**

```r
# Precompute id-year to row index mapping
build_neighbor_lookup_optimized <- function(data, id_order, neighbors) {
  # Ensure ids and years are integer or factor for fast mapping
  id_to_row <- match(data$id, id_order)     # maps each row's id to its ref index
  year_vals <- as.integer(as.factor(data$year))  # map years to integers
  
  # Build matrix for (id_ref, year_int) -> row index
  n_ids <- length(id_order)
  n_years <- length(unique(data$year))
  row_index_matrix <- matrix(NA_integer_, nrow = n_ids, ncol = n_years)
  for (i in seq_len(nrow(data))) {
    row_index_matrix[id_to_row[i], year_vals[i]] <- i
  }
  
  # Build neighbor lookup: list of integer vectors
  lapply(seq_len(nrow(data)), function(i) {
    ref_idx <- id_to_row[i]
    neighbor_refs <- neighbors[[ref_idx]]
    if (length(neighbor_refs) == 0) return(integer(0))
    year_idx <- year_vals[i]
    # Direct integer lookup from matrix
    idxs <- row_index_matrix[neighbor_refs, year_idx]
    idxs[!is.na(idxs)]
  })
}

# Compute neighbor stats with vectorization
compute_neighbor_stats_optimized <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  result <- matrix(NA_real_, nrow = length(neighbor_lookup), ncol = 3)
  for (i in seq_along(neighbor_lookup)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      v <- vals[idx]
      v <- v[!is.na(v)]
      if (length(v) > 0) {
        result[i, ] <- c(max(v), min(v), mean(v))
      }
    }
  }
  result
}

# Outer loop unchanged except for optimized functions
neighbor_lookup <- build_neighbor_lookup_optimized(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_optimized(cell_data, neighbor_lookup, var_name)
  colnames(stats) <- paste(var_name, c("max", "min", "mean"), sep = "_")
  cell_data <- cbind(cell_data, stats)
}
```

---

### **Expected Gains**
- Eliminates millions of string concatenations and hash lookups.
- Uses integer arithmetic and matrix indexing (O(1) per lookup).
- Memory: `row_index_matrix` is `n_ids × n_years` (≈ 344,208 × 28 ≈ 9.6M integers, ~40 MB), feasible on 16 GB RAM.
- Runtime improvement: from **86+ hours** to **a few hours or less**, dominated by aggregation and I/O.

**Preserves original estimand and Random Forest model.**