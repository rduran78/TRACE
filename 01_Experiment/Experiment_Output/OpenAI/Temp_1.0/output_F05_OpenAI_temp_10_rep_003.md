 **Diagnosis**  
The repeated creation of `neighbor_keys` (`paste(neighbor_cell_ids, data$year[i], sep = "_")`) and subsequent lookups in `idx_lookup` inside the inner `lapply` loop are only symptoms of a **larger algorithmic inefficiency**:

- You are building neighbor index lookups on-the-fly **for every row** (`6.46M` rows) by repeatedly doing:
  - String concatenation (`paste`) per neighbor relation.
  - Name-based lookup in `idx_lookup`, which creates unnecessary overhead at this scale.
- This effectively scales with  
  `O(#rows × avg_neighbors)` ≈ `6.46M × 4-8` → tens of millions of map lookups.
- Because the neighbor topology **does not change across years**, the entire repeated-string logic is avoidable: compute a block-based index once and reuse it across all years.

The bottleneck is therefore *algorithmic*, not just micro-inefficiency. Optimal plan:  
→ Precompute neighbor ID → row-index relationships structured as integer arrays **without string keys**, leveraging the fact that data is sorted by `(id, year)` or can be ordered so.  

---

### **Optimization Strategy**
1. **Sort `data` by (id, year)** so that rows for each `id` form a contiguous block of `n_years` rows.
2. Compute:
   - `years <- sort(unique(data$year))`
   - `n_years <- length(years)`
   - `id_pos <- match(data$id, id_order)`
3. Build a **static neighbor index matrix** (`neighbor_lookup`) using integer indices instead of repeated string pastes:
   - For cell `i` at year `t` (`row_index r`), its neighbors are:
     ```
     neighbor_idx = neighbors[[id_pos[i]]]
     neighbor_rows = (neighbor_idx - 1) * n_years + year_index[t]
     ```
   - This is pure integer math: `row = (id_index - 1)*n_years + year_idx`.
4. Use these indices once for all variable computations.

Memory check:  
- 6.46M rows → integer indices = ~ 25MB for a list of vectors (manageable with 16GB RAM).

---

### **Working Optimized R Code**

```r
# Assume: data has columns id, year, sorted by id then year
optimize_neighbor_lookup <- function(data, id_order, neighbors) {
  years <- sort(unique(data$year))
  n_years <- length(years)

  # Mapping id -> position in id_order
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # For fast year index lookup
  year_index <- match(data$year, years)

  # Confirm data is sorted by id, then year (if not, do order)
  o <- order(data$id, data$year)
  if (!isTRUE(all.equal(o, seq_len(nrow(data))))) {
    data <- data[o, ]
    year_index <- year_index[o]
  }

  # Build lookup: row i -> integer vector of neighbor row indices
  id_ref_vec <- id_to_ref[as.character(data$id)]
  n_rows <- nrow(data)

  neighbor_lookup <- vector("list", n_rows)

  # Precompute multiplier for block offset
  block_offset <- (seq_along(id_order) - 1L) * n_years

  for (r in seq_len(n_rows)) {
    ref_idx <- id_ref_vec[r]
    n_ids <- neighbors[[ref_idx]]
    if (length(n_ids) == 0L) {
      neighbor_lookup[[r]] <- integer(0L)
    } else {
      neighbor_rows <- block_offset[n_ids] + year_index[r]
      neighbor_lookup[[r]] <- neighbor_rows
    }
  }

  neighbor_lookup
}

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  result <- matrix(NA_real_, nrow = length(neighbor_lookup), ncol = 3)
  for (i in seq_along(neighbor_lookup)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0L) {
      neighbor_vals <- vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) > 0L) {
        result[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
      }
    }
  }
  result
}

# Usage:
neighbor_lookup <- optimize_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  colnames(stats) <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  cell_data <- cbind(cell_data, stats)
}
```

---

### **Expected Runtime Reduction**
- Removes repeated string concatenation and name-based lookup.
- Each neighbor-index computation now is pure integer arithmetic done once.
- For 6.46M rows and moderate neighbor density, this should reduce runtime from **86+ hours → a few hours (or less)** on a standard 16 GB machine.

**Core Principle:** Precompute numeric neighbor indices once; reuse across all target variables, cutting complexity drastically while preserving the trained model and original estimand.