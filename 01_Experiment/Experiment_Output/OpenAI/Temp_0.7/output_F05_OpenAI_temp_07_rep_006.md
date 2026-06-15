 **Diagnosis:**  
The repeated `paste()` and string-based lookups inside `build_neighbor_lookup()` are **only a local inefficiency, but symptomatic of a broader algorithmic issue**. The current design repeatedly maps `(id, year)` → row index via a character-keyed `idx_lookup`. This involves:

- Constructing millions of string keys (`id_year`) and hashing them.
- Doing this for **every row** across ~6.46M observations.
- Repeating the computation for each of the 5 neighbor source variables.

This is extremely expensive because the lookup is recalculated for each row and depends on string processing rather than integer indexing. The underlying problem is that the algorithm is operating at the row level instead of exploiting the **panel structure** (id × year) and **static neighbor relationships**.

---

### **Optimization Strategy**
- **Precompute an integer-based neighbor index matrix once** for all rows, eliminating repeated string concatenation and hash lookups.
- Use **matrix operations** or `vapply` instead of repeated `lapply` calls with dynamic string handling.
- Leverage the fact that `neighbors` is static across years: for each cell `id`, its neighbors in each year correspond to a fixed set of row offsets.
- Build a **neighbor index matrix** of size `nrow(data)` × `max_degree` (fill with `NA` for missing neighbors). Then reuse this for all variables.

This reduces complexity from repeated string-based hashing to pure integer lookups, cutting runtime from days to hours (or minutes).

---

### **Reformulated R Code**

```r
# Precompute neighbor index matrix once
build_neighbor_index_matrix <- function(data, id_order, neighbors) {
  # Map id -> row indices by year
  n_ids   <- length(id_order)
  n_years <- length(unique(data$year))
  id_to_rows <- split(seq_len(nrow(data)), data$id)  # list: id -> row indices

  max_deg <- max(lengths(neighbors))
  n_rows  <- nrow(data)
  
  # Preallocate integer matrix (n_rows x max_deg)
  neighbor_matrix <- matrix(NA_integer_, nrow = n_rows, ncol = max_deg)
  
  for (ref_idx in seq_along(id_order)) {
    ref_id <- id_order[ref_idx]
    ref_rows <- id_to_rows[[as.character(ref_id)]]
    if (is.null(ref_rows)) next
    
    # Neighbor IDs for this cell
    nb_ids <- id_order[neighbors[[ref_idx]]]
    if (length(nb_ids) == 0) next
    
    # For each year (row in ref_rows), map to neighbor rows of same year
    for (rpos in seq_along(ref_rows)) {
      ref_row <- ref_rows[rpos]
      yr <- data$year[ref_row]
      
      # Find neighbor rows for this year
      nb_rows <- integer(0)
      for (nb_id in nb_ids) {
        nb_set <- id_to_rows[[as.character(nb_id)]]
        if (!is.null(nb_set)) {
          # Find row for this year (same index as yr)
          # Assuming each id has one row per year in same order
          nb_row <- nb_set[rpos]  # position matches year ordering
          nb_rows <- c(nb_rows, nb_row)
        }
      }
      if (length(nb_rows) > 0) {
        neighbor_matrix[ref_row, seq_along(nb_rows)] <- nb_rows
      }
    }
  }
  
  neighbor_matrix
}

# Compute neighbor stats using precomputed matrix
compute_neighbor_stats_matrix <- function(data, neighbor_matrix, var_name) {
  vals <- data[[var_name]]
  apply(neighbor_matrix, 1, function(idx) {
    idx <- idx[!is.na(idx)]
    if (length(idx) == 0) return(c(NA, NA, NA))
    neighbor_vals <- vals[idx]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) return(c(NA, NA, NA))
    c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  }) |> t()
}

# Pipeline
neighbor_matrix <- build_neighbor_index_matrix(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_matrix(cell_data, neighbor_matrix, var_name)
  colnames(stats) <- paste(var_name, c("max", "min", "mean"), sep = "_")
  cell_data <- cbind(cell_data, stats)
}
```

---

### **Why this works**
- **No repeated string concatenation or hashing.**
- Precomputes neighbor indices once using integer positions.
- Each neighbor set is reused across all variables.
- Reduces complexity from O(n × neighbors × vars) with hashing to O(n × neighbors + n × vars).

This approach should **cut runtime from 86+ hours to a few hours or less** on a 16 GB laptop, without changing the estimand or retraining the Random Forest model.