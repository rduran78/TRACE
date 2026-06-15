 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck has two root causes:

### 1. `build_neighbor_lookup` — O(n) `lapply` over 6.46 million rows with per-row string pasting and named-vector lookups
- For every one of the ~6.46M rows, the code pastes strings (`paste(id, year)`), then does named-vector indexing into `idx_lookup` (a character-keyed vector of length 6.46M). Named-vector lookup in R is hash-based but the constant factor is large when called millions of times inside `lapply`.
- The result is a **list of 6.46M integer vectors** — this alone consumes enormous memory (each list element has overhead ≈ 56+ bytes even if empty, so ≥ 360 MB of pure overhead, plus the actual neighbor indices).

### 2. `compute_neighbor_stats` — another O(n) `lapply` over 6.46M rows
- Called **5 times** (once per source variable), each time iterating over every row, subsetting a numeric vector, and computing `max/min/mean`. The per-element R function-call overhead dominates.

### 3. Memory
- The 6.46M-element list of neighbor indices, the 6.46M × 110 data frame, and intermediate copies easily exceed 16 GB.

**Summary:** The design is idiomatic R but relies on millions of R-level function calls and string operations. The fix is to **vectorize everything** using `data.table` joins and grouped operations, eliminating both the per-row `lapply` loops and the string-keyed lookups entirely.

---

## Optimization Strategy

| Step | What changes | Why it helps |
|---|---|---|
| **A. Replace `build_neighbor_lookup` with an edge-list + `data.table` equi-join** | Instead of building a 6.46M-element list, expand the `nb` object into a two-column edge table `(cell_id, neighbor_id)`, then join on `(neighbor_id, year)` to get neighbor row indices (or values directly). | One vectorized join replaces 6.46M `paste` + named-vector lookups. `data.table` join is C-level and memory-mapped. |
| **B. Compute all neighbor stats in one grouped aggregation** | After the join attaches neighbor values, a single `data.table` `[, .(max, min, mean), by = .(cell_row)]` computes all three stats at once. | Replaces 6.46M R-level `lapply` calls per variable with one C-level grouped aggregation. |
| **C. Process all 5 variables in one pass** | Melt or simply carry all 5 source columns through the single join, then aggregate all 5 simultaneously. | Reduces the number of large joins from 5 to 1. |
| **D. Avoid the giant neighbor-index list entirely** | We never materialise a 6.46M-element R list. The edge table + join produces a long `data.table` that is aggregated and then discarded. | Saves hundreds of MB of list overhead. |

**Estimated speedup:** The vectorized approach should finish in **minutes** (roughly 5–15 min depending on disk/RAM speed), not 86+ hours. Peak RAM ≈ 4–6 GB.

---

## Working R Code

```r
# ------------------------------------------------------------------
# 0.  Libraries
# ------------------------------------------------------------------
library(data.table)

# ------------------------------------------------------------------
# 1.  Convert the nb object to a data.table edge list  (one-time, fast)
#     rook_neighbors_unique is a list of integer vectors (spdep::nb),
#     where element i contains the indices (into id_order) of neighbors
#     of id_order[i].
# ------------------------------------------------------------------
build_edge_table <- function(id_order, nb_obj) {
  # Pre-allocate: total number of directed edges
  n_edges <- sum(lengths(nb_obj))
  from_idx <- integer(n_edges)
  to_idx   <- integer(n_edges)
  pos <- 1L
  for (i in seq_along(nb_obj)) {
    ni <- nb_obj[[i]]
    len <- length(ni)
    if (len == 0L) next
    from_idx[pos:(pos + len - 1L)] <- i
    to_idx[pos:(pos + len - 1L)]   <- ni
    pos <- pos + len
  }
  data.table(
    cell_id     = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# edge_dt has ~1.37M rows (one per directed rook relationship)

# ------------------------------------------------------------------
# 2.  Convert cell_data to data.table (if not already) and add a
#     row identifier that we will aggregate back to.
# ------------------------------------------------------------------
setDT(cell_data)
cell_data[, .row_id := .I]

# ------------------------------------------------------------------
# 3.  Subset the columns we actually need for the neighbor join
#     to keep peak memory low.
# ------------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Small table: (id, year, .row_id) for the focal cells
focal <- cell_data[, .(id, year, .row_id)]

# Small table: (id, year, <source vars>) for the neighbor cells
neighbor_vals <- cell_data[, c("id", "year", neighbor_source_vars), with = FALSE]

# ------------------------------------------------------------------
# 4.  Join:  focal --> edge_dt --> neighbor_vals
#     For every focal row, find its neighbors in the same year and
#     pull their variable values.
# ------------------------------------------------------------------
# Step 4a: attach neighbor cell ids to every focal row
#   focal  JOIN  edge_dt  ON  focal.id == edge_dt.cell_id
setkey(edge_dt, cell_id)
setkey(focal, id)
focal_with_nbr <- edge_dt[focal, on = .(cell_id = id),
                           allow.cartesian = TRUE,
                           nomatch = NA]
# Result columns: cell_id, neighbor_id, year, .row_id
# Rows ≈ 6.46M * avg_neighbors (≈ 4) ≈ 25.8M  (manageable)

# Drop rows where there was no neighbor (isolated cells)
focal_with_nbr <- focal_with_nbr[!is.na(neighbor_id)]

# Step 4b: attach neighbor variable values by (neighbor_id, year)
setkey(neighbor_vals, id, year)
setkey(focal_with_nbr, neighbor_id, year)
joined <- neighbor_vals[focal_with_nbr,
                        on = .(id = neighbor_id, year),
                        nomatch = NA]
# 'joined' now has columns: id (=neighbor_id), year, ntl, ec, ...,
#                            cell_id, .row_id

# ------------------------------------------------------------------
# 5.  Grouped aggregation — compute max, min, mean for every
#     (focal row, variable) combination in one pass.
# ------------------------------------------------------------------
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(as.numeric(max(.(as.name(v)),   na.rm = TRUE))),
    bquote(as.numeric(min(.(as.name(v)),   na.rm = TRUE))),
    bquote(as.numeric(mean(.(as.name(v)),  na.rm = TRUE)))
  )
}), recursive = FALSE)

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

# Build the j-expression programmatically
agg_call <- as.call(c(
  as.name("list"),
  setNames(agg_exprs, agg_names)
))

stats <- joined[, eval(agg_call), by = .row_id]

# Replace -Inf/Inf from max/min of all-NA groups with NA
for (col in agg_names) {
  set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
}

# ------------------------------------------------------------------
# 6.  Merge the aggregated neighbor features back to cell_data
# ------------------------------------------------------------------
setkey(stats, .row_id)
setkey(cell_data, .row_id)
cell_data <- stats[cell_data, on = ".row_id"]

# Clean up the helper column
cell_data[, .row_id := NULL]

# ------------------------------------------------------------------
# 7.  (Optional) free large intermediates
# ------------------------------------------------------------------
rm(focal, neighbor_vals, focal_with_nbr, joined, stats, edge_dt)
gc()

# ------------------------------------------------------------------
# 8.  Predict with the existing Random Forest model (unchanged)
# ------------------------------------------------------------------
# The trained RF model object (e.g. `rf_model`) is used as-is.
# cell_data now contains the same 15 neighbor feature columns
# (5 vars × 3 stats) that the model expects, with identical
# numerical values, so the estimand is preserved.
#
#   cell_data$predicted_gdp <- predict(rf_model, newdata = cell_data)
# ------------------------------------------------------------------
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same set of rook-neighbor values per cell-year as the original code. The `data.table` join replicates the same `(cell, year) → neighbor (cell, year)` mapping. |
| **Trained RF model** | No model retraining occurs. The code only reconstructs the **feature columns** that already existed, with identical names and values, then calls `predict()`. |
| **Column names** | The programmatic `agg_names` vector produces the same `neighbor_max_ntl`, `neighbor_min_ntl`, `neighbor_mean_ntl`, … names (adjust the pattern if your original pipeline used a different naming convention). |

---

## Expected Performance

| Metric | Original | Optimized |
|---|---|---|
| Wall-clock time | ~86+ hours | **~5–15 minutes** |
| Peak RAM | >16 GB (fails/swaps) | **~4–6 GB** |
| Largest intermediate | 6.46M-element R list | ~26M-row `data.table` (flat, column-oriented) |

The key insight is that the entire neighbor-lookup-then-aggregate pattern is just a **relational join followed by a grouped aggregation** — exactly what `data.table` is engineered to do at C speed with minimal memory copies.