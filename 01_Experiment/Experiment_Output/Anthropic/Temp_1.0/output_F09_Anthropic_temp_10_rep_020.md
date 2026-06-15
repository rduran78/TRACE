 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of two fundamental bottlenecks:

1. **`build_neighbor_lookup` constructs a row-level lookup across all 6.46 million cell-year rows using `lapply` with string-pasting and named-vector indexing.** For each of the 6.46M rows, it pastes neighbor cell IDs with the row's year, then looks them up in a named character vector of length 6.46M. Named vector lookup in R is O(n) per query in the worst case; doing this ~6.46M × ~4 neighbors ≈ 26M string lookups against a 6.46M-length named vector is catastrophically slow.

2. **The lookup is *year-redundant*.** The spatial neighbor topology is identical across all 28 years. A cell's rook neighbors in 1992 are the same cells as in 2019. Yet `build_neighbor_lookup` rebuilds the full mapping for every cell-year row, embedding the year into the key and re-resolving it. This inflates the problem from ~344K cells × ~4 neighbors to ~6.46M rows × ~4 neighbors.

3. **`compute_neighbor_stats` then iterates over all 6.46M entries in `neighbor_lookup` with `lapply`, computing max/min/mean per row.** This is pure R-level looping—no vectorization.

**Core insight:** The neighbor *topology* is a static spatial property. It should be built **once** as a simple cell-to-cell adjacency table (~1.37M rows), then **joined** to the panel data by year to resolve neighbor attribute values, then **aggregated** with vectorized grouped operations. This converts the entire pipeline from O(rows × neighbors × string-ops) to a few fast data.table joins and grouped aggregations.

---

## Optimization Strategy

1. **Build a static directed-edge table once** from `rook_neighbors_unique` (the `nb` object): a two-column data.table with `focal_id` and `neighbor_id`. This has ~1.37M rows and never changes.

2. **For each year and each variable**, join the edge table to the cell-year attribute data on `(neighbor_id, year)` to attach each neighbor's variable value to each directed edge. Then group by `(focal_id, year)` and compute `max`, `min`, `mean` in a single vectorized pass.

3. **Join the resulting summary statistics back** onto `cell_data` by `(id, year)`.

4. This replaces 6.46M-iteration `lapply` calls with a handful of keyed data.table merges and `[, .(…), by=…]` aggregations—orders of magnitude faster.

**Expected speedup:** The entire neighbor-feature computation should drop from ~86+ hours to **minutes** (typically 2–10 minutes on a 16 GB laptop).

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build the static cell-neighbor edge table ONCE
# ──────────────────────────────────────────────────────────────────────
# Inputs:
#   id_order             — vector of cell IDs (length 344,208), in the
#                          same order as rook_neighbors_unique
#   rook_neighbors_unique — an nb object (list of integer index vectors)
#
# Output:
#   edge_dt — a data.table with columns: focal_id, neighbor_id
#             (~1,373,394 rows)

build_edge_table <- function(id_order, neighbors_nb) {
  # Pre-allocate vectors
  n_edges <- sum(lengths(neighbors_nb))
  focal_id    <- integer(n_edges)
  neighbor_id <- integer(n_edges)

  pos <- 1L
  for (i in seq_along(neighbors_nb)) {
    nb_idx <- neighbors_nb[[i]]
    # spdep nb objects use 0L to denote "no neighbors"
    nb_idx <- nb_idx[nb_idx != 0L]
    n_nb   <- length(nb_idx)
    if (n_nb > 0L) {
      focal_id[pos:(pos + n_nb - 1L)]    <- id_order[i]
      neighbor_id[pos:(pos + n_nb - 1L)] <- id_order[nb_idx]
      pos <- pos + n_nb
    }
  }
  # Trim if any 0-neighbor cells reduced the count
  data.table(focal_id = focal_id[1:(pos - 1L)],
             neighbor_id = neighbor_id[1:(pos - 1L)])
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

# ──────────────────────────────────────────────────────────────────────
# STEP 2: Convert cell_data to data.table (non-destructive)
# ──────────────────────────────────────────────────────────────────────
# Preserve original row order for downstream prediction
cell_dt <- as.data.table(cell_data)
cell_dt[, .row_order := .I]   # bookmark original order

# ──────────────────────────────────────────────────────────────────────
# STEP 3: For each neighbor source variable, compute neighbor
#         max / min / mean via join + grouped aggregation
# ──────────────────────────────────────────────────────────────────────

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Key the attribute table for fast join on (id, year)
# We create a slim lookup: just id, year, and the five source vars
attr_cols <- c("id", "year", neighbor_source_vars)
attr_dt   <- cell_dt[, ..attr_cols]
setkey(attr_dt, id, year)

# We will accumulate new columns in a results table keyed by (id, year)
results_dt <- cell_dt[, .(id, year, .row_order)]
setkey(results_dt, id, year)

for (var_name in neighbor_source_vars) {

  message("Computing neighbor stats for: ", var_name)

  # Slim neighbor-attribute table: just id (as neighbor_id), year, value
  nb_attr <- attr_dt[, .(neighbor_id = id, year, value = get(var_name))]
  setkey(nb_attr, neighbor_id, year)

  # Expand edges × years: join neighbor attributes onto every edge for

  # every year.
  #
  # Strategy: cross-join edge_dt with unique years, then look up the
  # neighbor's value.  But that would create edge_dt × 28 rows first.
  # More memory-efficient: join edges onto the long cell-year table.

  # For each row in cell_dt, we need that row's focal_id's neighbors.
  # Approach: join cell_dt (as focal) → edge_dt → nb_attr in two steps.

  # Step A: attach neighbors to each focal cell-year
  #   focal_rows has one row per (focal_id, year, neighbor_id)
  focal_key <- cell_dt[, .(focal_id = id, year)]
  setkey(focal_key, focal_id)
  setkey(edge_dt, focal_id)

  # This is an equi-join: every focal cell-year gets replicated for each

  # of its neighbors.  ~6.46M × avg_neighbors ≈ 26M rows — fits in 16 GB.
  expanded <- edge_dt[focal_key, on = .(focal_id), allow.cartesian = TRUE, nomatch = 0L]
  # expanded columns: focal_id, neighbor_id, year

  # Step B: attach the neighbor's variable value
  setkey(expanded, neighbor_id, year)
  expanded[nb_attr, on = .(neighbor_id, year), value := i.value]

  # Step C: aggregate by (focal_id, year)
  stats <- expanded[!is.na(value),
                    .(nb_max  = max(value),
                      nb_min  = min(value),
                      nb_mean = mean(value)),
                    by = .(focal_id, year)]

  # Rename columns to match original pipeline naming convention
  #   e.g. ntl_neighbor_max, ntl_neighbor_min, ntl_neighbor_mean
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  setnames(stats, c("nb_max", "nb_min", "nb_mean"),
                  c(max_col, min_col, mean_col))

  # Join back to results
  setkey(stats, focal_id, year)
  results_dt <- stats[results_dt, on = .(focal_id = id, year)]
  setnames(results_dt, "focal_id", "id")
  setkey(results_dt, id, year)

  # Clean up per-iteration large objects
  rm(nb_attr, expanded, stats)
  gc()
}

# ──────────────────────────────────────────────────────────────────────
# STEP 4: Merge neighbor features back into cell_data (original frame)
# ──────────────────────────────────────────────────────────────────────
# Restore original row order
setorder(results_dt, .row_order)

# Identify the new neighbor-stat columns
new_cols <- setdiff(names(results_dt), c("id", "year", ".row_order"))

# Bind new columns onto the original cell_data (preserves class: data.frame / tibble)
for (col in new_cols) {
  cell_data[[col]] <- results_dt[[col]]
}

# ──────────────────────────────────────────────────────────────────────
# STEP 5: Predict with the existing trained Random Forest (unchanged)
# ──────────────────────────────────────────────────────────────────────
# The trained model object (e.g. `rf_model`) is used as-is.
# cell_data now contains the same neighbor feature columns with the
# same numerical values as the original pipeline would have produced.
#
# Example (adjust to your actual model object name):
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Why This Is Correct and Equivalent

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | The join-then-aggregate approach computes `max`, `min`, `mean` over exactly the same set of non-NA neighbor values as the original `compute_neighbor_stats`. No rounding or approximation is introduced. |
| **Trained RF model preserved** | The model object is never touched. Only the input feature columns are recomputed (identically). |
| **Column names preserved** | The `setnames` call produces the same `{var}_neighbor_max/min/mean` columns the original `compute_and_add_neighbor_features` would have created. |
| **NA handling** | Rows with no neighbors (or all-NA neighbor values) receive `NA` for all three stats via the `nomatch` / left-join semantics, matching the original `c(NA, NA, NA)` return. |

## Expected Performance

| Step | Estimated Time |
|---|---|
| `build_edge_table` (once) | < 5 seconds |
| Per-variable join + aggregation | ~30–90 seconds each |
| Total for 5 variables | **~3–8 minutes** |
| **Original implementation** | **~86+ hours** |

The speedup factor is roughly **600–1,700×**, achieved entirely by eliminating per-row string operations and replacing R-level `lapply` loops with vectorized data.table keyed joins and grouped aggregations.