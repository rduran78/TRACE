 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` — O(n) list of per-row lookups via `lapply`

- With ~6.46 million rows, this creates a list of 6.46M elements. Each iteration does string pasting, hash-table lookups (`idx_lookup[neighbor_keys]`), and NA filtering. The string-based key construction (`paste(id, year, sep="_")`) and named-vector lookup is extremely slow at this scale — effectively O(n × k) with large constant factors from string operations, where k ≈ average number of rook neighbors (~4).
- The lookup is **year-invariant in structure** (the same cell always has the same neighbors), yet the code rebuilds neighbor indices per cell-year row rather than exploiting the fact that neighbor topology is constant across years.

### Bottleneck B: `compute_neighbor_stats` — `lapply` over 6.46M elements

- Each call iterates over all 6.46M rows, extracts neighbor values, removes NAs, and computes max/min/mean. This is called 5 times (once per variable), so ~32.3M iterations total.
- `lapply` + `do.call(rbind, ...)` on a 6.46M-element list is inherently slow in R.

### Why raster focal/kernel operations are **not** directly applicable

The comment in the prompt about raster focal operations is a useful analogy — focal operations (e.g., `terra::focal`) compute neighborhood statistics on regular grids extremely efficiently. However:

- The data is in **long panel format** (cell × year), not a raster stack.
- The neighbor structure (`spdep::nb`) may not correspond to a perfectly regular grid (boundary cells, missing cells).
- Focal operations would require reshaping into raster layers per year, running focal, then reshaping back — and would silently change results at boundaries or for irregular grids.

**The correct strategy is to stay in tabular form but replace all R-level loops and string operations with vectorized/matrix operations using `data.table`.**

---

## 2. Optimization Strategy

| Step | What changes | Why it's faster |
|------|-------------|-----------------|
| **1. Use `data.table` keyed joins** | Replace string-paste + named-vector lookup with integer-keyed merge | Eliminates millions of `paste()` and hash lookups |
| **2. Expand neighbor pairs into an edge table** | Build a single `data.table` of `(id, neighbor_id)` from the `nb` object once | Vectorized, no per-row `lapply` |
| **3. Join variable values onto edge table by `(neighbor_id, year)`** | Single keyed join brings all neighbor values in one operation | O(n log n) binary search join vs. O(n×k) string lookup |
| **4. Grouped aggregation** | `edge_dt[, .(max, min, mean), by = .(id, year)]` | `data.table` C-level grouped aggregation — extremely fast |
| **5. Process all 5 variables in one pass** | Aggregate all 5 variables simultaneously in one grouped operation | Avoids 5× repeated joins |

**Expected speedup**: From 86+ hours to roughly **5–15 minutes** on the same laptop.

---

## 3. Working R Code

```r
library(data.table)

# ============================================================
# STEP 1: Build the edge table from the nb object (once)
# ============================================================
# rook_neighbors_unique is a list of integer vectors (spdep::nb object).
# id_order is the vector mapping position -> cell id.

build_edge_table <- function(id_order, nb_obj) {
  # nb_obj[[i]] contains the indices (into id_order) of neighbors of cell i.
  # We expand this into a two-column data.table of (id, neighbor_id).
  n <- length(nb_obj)
  
  # Pre-compute lengths for pre-allocation
  lens <- vapply(nb_obj, length, integer(1))
  # Handle the spdep convention: nb objects use 0L to indicate no neighbors
  lens[lens == 1L & vapply(nb_obj, function(x) x[1] == 0L, logical(1))] <- 0L
  
  total_edges <- sum(lens)
  
  from_id <- integer(total_edges)
  to_id   <- integer(total_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    k <- lens[i]
    if (k == 0L) next
    idx_range <- pos:(pos + k - 1L)
    from_id[idx_range] <- id_order[i]
    to_id[idx_range]   <- id_order[nb_obj[[i]]]
    pos <- pos + k
  }
  
  data.table(id = from_id, neighbor_id = to_id)
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
cat("Edge table rows:", nrow(edge_dt), "\n")

# ============================================================
# STEP 2: Convert cell_data to data.table (if not already)
# ============================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ============================================================
# STEP 3: Compute all neighbor stats in one vectorized pass
# ============================================================
compute_all_neighbor_features <- function(cell_data, edge_dt, source_vars) {
  
  # Columns we need from cell_data for the join
  join_cols <- c("id", "year", source_vars)
  
  # Subset to only needed columns for the neighbor value lookup
  neighbor_vals <- cell_data[, ..join_cols]
  
  # Key for joining: we want to look up neighbor values by (neighbor_id, year)
  setnames_map <- c(id = "neighbor_id")
  
  # Create the join table: edge_dt expanded by year

  # Instead of a full cross join (which would be huge), we merge edge_dt 

  # with cell_data on the "from" side to get years, then join neighbor values.
  
  # Step 3a: Get (id, year) from cell_data, join with edge_dt to get 
  #          (id, year, neighbor_id)
  id_year <- cell_data[, .(id, year)]
  setkey(id_year, id)
  setkey(edge_dt, id)
  
  # This gives us every (id, year, neighbor_id) triple
  expanded <- edge_dt[id_year, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded has columns: id, neighbor_id, year
  
  cat("Expanded edge-year rows:", nrow(expanded), "\n")
  
  # Step 3b: Join neighbor variable values by (neighbor_id, year)
  setkey(neighbor_vals, id, year)
  setkey(expanded, neighbor_id, year)
  
  expanded <- neighbor_vals[expanded, on = c(id = "neighbor_id", "year")]
  # Now expanded has: id (= original neighbor_id), year, <source_vars>, 
  #                   i.id (= the focal cell id)
  # Rename for clarity
  # After the join: 'id' column = neighbor_id, 'i.id' = focal cell id
  setnames(expanded, c("id", "i.id"), c("neighbor_id", "id"))
  
  # Step 3c: Grouped aggregation — compute max, min, mean for each variable
  #          grouped by (id, year)
  agg_exprs <- list()
  for (v in source_vars) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("n_max_", v)]]  <- substitute(max(x, na.rm = TRUE),  list(x = v_sym))
    agg_exprs[[paste0("n_min_", v)]]  <- substitute(min(x, na.rm = TRUE),  list(x = v_sym))
    agg_exprs[[paste0("n_mean_", v)]] <- substitute(mean(x, na.rm = TRUE), list(x = v_sym))
  }
  
  # Build the aggregation call
  agg_result <- expanded[, 
    {
      out <- list()
      for (v in source_vars) {
        nv <- get(v)
        nv <- nv[!is.na(nv)]
        if (length(nv) == 0L) {
          out[[paste0("n_max_", v)]]  <- NA_real_
          out[[paste0("n_min_", v)]]  <- NA_real_
          out[[paste0("n_mean_", v)]] <- NA_real_
        } else {
          out[[paste0("n_max_", v)]]  <- max(nv)
          out[[paste0("n_min_", v)]]  <- min(nv)
          out[[paste0("n_mean_", v)]] <- mean(nv)
        }
      }
      out
    },
    by = .(id, year)
  ]
  
  return(agg_result)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
t0 <- proc.time()

neighbor_features <- compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)

cat("Neighbor features computed in", (proc.time() - t0)[3], "seconds\n")

# ============================================================
# STEP 4: Merge neighbor features back into cell_data
# ============================================================
setkey(cell_data, id, year)
setkey(neighbor_features, id, year)

cell_data <- neighbor_features[cell_data, on = .(id, year)]

# Cells with no neighbors (e.g., islands) will have NA — this matches 
# the original implementation's behavior.

# Fix -Inf/Inf from max/min on empty sets (shouldn't occur due to 
# the NA handling above, but as a safety net):
inf_cols <- grep("^n_max_|^n_min_", names(cell_data), value = TRUE)
for (col in inf_cols) {
  set(cell_data, which(is.infinite(cell_data[[col]])), col, NA_real_)
}

cat("Final cell_data rows:", nrow(cell_data), "\n")
cat("Final cell_data cols:", ncol(cell_data), "\n")

# ============================================================
# STEP 5: Predict with the pre-trained Random Forest (unchanged)
# ============================================================
# The trained model object is assumed to already exist in the environment.
# For example:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## 4. Why This Preserves the Original Numerical Estimand

| Concern | Guarantee |
|---------|-----------|
| **Same neighbor definition** | The edge table is built from the identical `rook_neighbors_unique` nb object. |
| **Same statistics** | `max`, `min`, `mean` with `na.rm = TRUE` after filtering `NA`s — identical logic. |
| **Same NA handling** | Cells with zero valid neighbors get `NA` for all three stats, exactly as the original `c(NA, NA, NA)` return. |
| **No model retraining** | The Random Forest model is loaded and used for `predict()` only — no fitting occurs. |
| **Column names** | The output columns follow the `n_max_<var>`, `n_min_<var>`, `n_mean_<var>` naming convention, which should match whatever `compute_and_add_neighbor_features` originally produced (adjust the prefix if the original used a different convention). |

---

## 5. Performance Estimate

| Component | Original | Optimized |
|-----------|----------|-----------|
| Neighbor lookup construction | ~hours (6.46M string-paste + hash lookups) | ~10–30 sec (integer edge table) |
| Neighbor stats (5 vars × 6.46M rows) | ~80+ hours (R-level lapply) | ~2–10 min (data.table grouped agg) |
| **Total** | **86+ hours** | **~5–15 minutes** |

The key insight: **the neighbor topology is time-invariant**, so we build the edge structure once on ~344K cells, then let `data.table`'s binary-search join and C-level grouped aggregation handle the 6.46M-row expansion — replacing billions of R-interpreter-level operations with a handful of vectorized calls.