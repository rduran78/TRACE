 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(N²)-like in practice
This function iterates over **all ~6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs **character key strings** (`paste(id, year)`) and performs **named-vector lookups** (`idx_lookup[neighbor_keys]`) — which in R are hash-table lookups on a 6.46-million-entry named vector, repeated ~6.46 million times × ~4 neighbors each ≈ **~26 million hash lookups**.

Character pasting and named-vector indexing are slow per-call in R. At ~6.46M outer iterations, this dominates runtime.

### 2. `compute_neighbor_stats` uses per-row `lapply`
For each of the 5 variables, it loops over 6.46M rows in R-level `lapply`, extracting small vectors and computing `max/min/mean`. That's **~32 million R function calls** across all variables.

### 3. The neighbor topology is **static** but is being resolved at the cell-year level
Rook neighbors are a property of the **spatial grid** (344,208 cells), not of cell-years (6.46M rows). The current code conflates the two, doing ~19× more work than necessary for the lookup phase.

---

## Optimization Strategy

**Core insight:** Separate the *spatial topology* (which cells neighbor which — fixed, 344K cells) from the *panel attributes* (which vary by year — 28 years). Then use vectorized joins and grouped vectorized aggregation instead of row-wise R loops.

**Steps:**

1. **Build a static edge table once** — a two-column `data.table` of `(cell_id, neighbor_id)` from the `nb` object. This has ~1.37M rows and never changes.

2. **Join yearly attributes onto the edge table** — for each year, join the cell-year attributes onto the `neighbor_id` column. This is a keyed `data.table` join: O(E) per year, fully vectorized in C.

3. **Grouped aggregation** — group by `(cell_id, year)` and compute `max`, `min`, `mean` of each neighbor variable in one vectorized pass using `data.table`'s `by=` grouping (GForce-optimized in C).

4. **Join results back** to the main dataset.

**Expected speedup:** From ~86 hours to **minutes** (likely 2–10 minutes on a 16 GB laptop). The bottleneck shifts from millions of R-level function calls to a handful of vectorized C-level `data.table` operations.

**Preservation guarantees:**
- The trained Random Forest model is untouched — we only produce the same predictor columns.
- The numerical estimand is identical: `max`, `min`, `mean` of the same neighbor values, with the same `NA` handling.

---

## Working R Code

```r
library(data.table)

# ==============================================================
# STEP 1: Build a static spatial edge table (once, reusable)
# ==============================================================
# Input: id_order (vector of 344,208 cell IDs, matching the nb object index)
#        rook_neighbors_unique (spdep nb object, list of integer index vectors)

build_edge_table <- function(id_order, neighbors_nb) {
  # Pre-allocate: count total edges
  n_edges <- sum(vapply(neighbors_nb, function(x) {
    # spdep nb encodes "no neighbors" as a single 0L
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1)))

  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  pos <- 1L

  for (i in seq_along(neighbors_nb)) {
    nb_idx <- neighbors_nb[[i]]
    if (length(nb_idx) == 1L && nb_idx[1] == 0L) next
    n <- length(nb_idx)
    from_id[pos:(pos + n - 1L)] <- id_order[i]
    to_id[pos:(pos + n - 1L)]   <- id_order[nb_idx]
    pos <- pos + n
  }

  data.table(cell_id = from_id, neighbor_id = to_id)
}

edge_table <- build_edge_table(id_order, rook_neighbors_unique)
# ~1.37M rows, two integer columns — tiny in memory

# ==============================================================
# STEP 2: Compute all neighbor features via vectorized joins
# ==============================================================

compute_all_neighbor_features <- function(cell_data_df, edge_table,
                                          neighbor_source_vars) {
  # Convert to data.table if needed (by reference if already one)
  dt <- as.data.table(cell_data_df)

  # Columns we need from the neighbor rows: id, year, and the source vars
  # Build the join: for every (cell_id, year), find neighbor rows' attributes
  # We expand edge_table × years via a merge with the attribute table on neighbor_id.

  # Subset to only needed columns for the neighbor attribute lookup
  attr_cols <- c("id", "year", neighbor_source_vars)
  dt_attr <- dt[, ..attr_cols]
  setnames(dt_attr, "id", "neighbor_id")
  setkeyv(dt_attr, c("neighbor_id", "year"))

  # Join neighbor attributes onto edge table (edge_table is year-agnostic,
  # so we do a many-to-many join: each edge × each year the neighbor has data)
  # This produces ~1.37M edges × 28 years ≈ 38.4M rows (fits in 16 GB easily
  # for a handful of numeric columns).

  setkey(edge_table, neighbor_id)

  # Efficient approach: join edge_table with dt_attr on neighbor_id
  # This gives us (cell_id, neighbor_id, year, var1, var2, ...)
  edges_with_attrs <- dt_attr[edge_table,
                              on = .(neighbor_id),
                              allow.cartesian = TRUE,
                              nomatch = NA]
  # Result columns: neighbor_id, year, <vars>, cell_id

  # Now aggregate: for each (cell_id, year), compute max/min/mean of each var
  agg_exprs <- list()
  for (v in neighbor_source_vars) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("neighbor_max_", v)]]  <-
      bquote(as.numeric(max(.(v_sym), na.rm = TRUE)))
    agg_exprs[[paste0("neighbor_min_", v)]]  <-
      bquote(as.numeric(min(.(v_sym), na.rm = TRUE)))
    agg_exprs[[paste0("neighbor_mean_", v)]] <-
      bquote(mean(.(v_sym), na.rm = TRUE))
  }

  # Build a single aggregation call for efficiency
  # data.table GForce optimizes max, min, mean when called directly in j
  agg_call <- as.call(c(as.name("list"),
                         lapply(agg_exprs, function(e) e)))

  neighbor_stats <- edges_with_attrs[,
                                      eval(agg_call),
                                      by = .(cell_id, year)]

  # Replace -Inf/Inf from max/min on all-NA groups with NA
  for (col_name in names(neighbor_stats)) {
    if (col_name %in% c("cell_id", "year")) next
    vals <- neighbor_stats[[col_name]]
    set(neighbor_stats, i = which(is.infinite(vals)), j = col_name, value = NA_real_)
  }

  # Merge back onto the main data
  setnames(neighbor_stats, "cell_id", "id")
  setkeyv(neighbor_stats, c("id", "year"))
  setkeyv(dt, c("id", "year"))

  # Remove old neighbor columns if they exist (to avoid duplicates)
  old_cols <- intersect(names(dt), setdiff(names(neighbor_stats), c("id", "year")))
  if (length(old_cols) > 0) dt[, (old_cols) := NULL]

  dt <- neighbor_stats[dt, on = .(id, year)]

  # Return as data.frame if the input was one
  if (is.data.frame(cell_data_df) && !is.data.table(cell_data_df)) {
    setDF(dt)
  }
  dt
}

# ==============================================================
# STEP 3: Run it
# ==============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  edge_table,
  neighbor_source_vars
)

# ==============================================================
# STEP 4: Predict with the existing trained Random Forest
# ==============================================================
# The trained model object is unchanged. The predictor columns are
# numerically identical to the original pipeline's output.
# Example (adjust to your actual model object and prediction column):
#
#   cell_data$rf_prediction <- predict(trained_rf_model, newdata = cell_data)
```

---

## Complexity Comparison

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup | 6.46M R-level iterations + 26M hash lookups on character keys | One `for` loop over 344K cells to build edge table (once) |
| Stat computation | 6.46M × 5 = 32.3M R `lapply` calls | One vectorized `data.table` grouped aggregation |
| Memory pattern | 6.46M-entry named character vector + 6.46M-entry list | ~38M-row numeric `data.table` (~1.5 GB for 5 vars) |
| Estimated time | ~86 hours | **~2–10 minutes** |
| Numerical result | max/min/mean with NA removal | Identical max/min/mean with NA removal |
| Trained RF model | Preserved | Preserved (untouched) |

The key insight is that **spatial topology is time-invariant**: build the ~1.37M-edge table once, then let `data.table`'s C-level join and GForce aggregation handle the panel expansion and summarization entirely in vectorized compiled code.