 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two compounding reasons:

### 1. `build_neighbor_lookup` is O(N) over all cell-year rows (~6.46M), but it rebuilds neighbor relationships redundantly across years.

The neighbor topology is **purely spatial** — cell A's neighbors are the same in 1992 as in 2019. Yet `build_neighbor_lookup` creates a lookup entry **per cell-year row**, iterating over ~6.46M rows, performing character key construction and hash lookups for each. This means the same spatial neighbor set is resolved **28 times** (once per year) for each of the 344,208 cells. That's ~9.6 million redundant key-paste-and-match operations.

### 2. `compute_neighbor_stats` uses an R-level `lapply` over ~6.46M entries.

Each iteration indexes into a numeric vector and computes `max`, `min`, `mean` — all in interpreted R. With 5 variables × 6.46M rows, that's ~32.3 million R-level function calls with per-element vector subsetting. This is the dominant bottleneck.

### 3. Memory bloat from the lookup structure.

`neighbor_lookup` is a list of 6.46M integer vectors. The list overhead alone (~6.46M SEXP pointers + individual vector allocations) can consume several GB of RAM on a 16 GB laptop, causing GC pressure and swapping.

**Root cause summary:** The spatial topology is conflated with the temporal panel. The code treats each cell-year as a unique entity needing its own neighbor resolution, when in fact the neighbor graph is time-invariant.

---

## Optimization Strategy

**Core idea:** Separate the *spatial topology* (built once) from the *temporal attributes* (joined per year), then compute neighbor statistics using vectorized `data.table` operations instead of row-wise R loops.

### Step-by-step plan:

1. **Build a spatial edge table once.** Convert the `spdep::nb` object into a two-column `data.table` of `(cell_id, neighbor_id)` — roughly 1.37M rows. This is done once and is year-invariant.

2. **For each year, join cell attributes onto the edge table.** This gives each edge the neighbor's attribute value. Then group by `(cell_id, year)` and compute `max`, `min`, `mean` in one vectorized pass.

3. **Join the resulting statistics back to the main data.** This replaces the per-row R-level `lapply`.

### Complexity comparison:

| | Current | Optimized |
|---|---|---|
| Neighbor resolution | ~6.46M R-level iterations | ~1.37M-row edge table (built once) |
| Stats computation | ~32.3M R-level `lapply` calls | 5 vectorized `data.table` group-by operations over ~38.4M edge-year rows |
| Expected time | ~86+ hours | **~2–10 minutes** |
| RAM for lookup | Several GB (list of 6.46M vectors) | ~200–400 MB (edge table + joins) |

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 0: Ensure cell_data is a data.table with columns: id, year, and all
#         predictor variables. The trained RF model object is untouched.
# ==============================================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ==============================================================================
# STEP 1: Build the time-invariant spatial edge table ONCE from the nb object.
#
#   rook_neighbors_unique : spdep nb object (list of integer index vectors)
#   id_order              : vector of cell IDs in the same order as the nb object
#
#   Result: edges_dt — a data.table with columns (cell_id, neighbor_id)
#           representing every directed rook-neighbor pair.
# ==============================================================================

build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate by computing total number of edges
  n_cells <- length(id_order)
  n_edges <- sum(vapply(neighbors, length, integer(1)))

  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)

  pos <- 1L
  for (i in seq_len(n_cells)) {
    nb_idx <- neighbors[[i]]
    n_nb   <- length(nb_idx)
    if (n_nb > 0L) {
      from_id[pos:(pos + n_nb - 1L)] <- id_order[i]
      to_id[pos:(pos + n_nb - 1L)]   <- id_order[nb_idx]
      pos <- pos + n_nb
    }
  }

  data.table(cell_id = from_id, neighbor_id = to_id)
}

edges_dt <- build_edge_table(id_order, rook_neighbors_unique)

cat(sprintf(
  "Edge table built: %s directed neighbor pairs for %s cells.\n",
  format(nrow(edges_dt), big.mark = ","),
  format(length(id_order), big.mark = ",")
))

# ==============================================================================
# STEP 2: For each neighbor source variable, compute neighbor max, min, mean
#         by joining yearly attributes onto the edge table, then grouping.
#
#   This replaces build_neighbor_lookup + compute_neighbor_stats entirely.
# ==============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Extract only the columns we need for the join: id, year, and the source vars.
# This keeps memory lean during the join operations.
join_cols <- unique(c("id", "year", neighbor_source_vars))
attr_dt   <- cell_data[, ..join_cols]

# We will join edges_dt (cell_id -> neighbor_id) with attr_dt on neighbor_id + year.
# First, get the unique years to iterate over (avoids a massive cross-join).
all_years <- sort(unique(attr_dt$year))

# Pre-set keys for fast joins
setkey(edges_dt, neighbor_id)

# Function to compute neighbor stats for one variable across all years
compute_neighbor_features_fast <- function(attr_dt, edges_dt, var_name, all_years) {

  cat(sprintf("  Computing neighbor stats for: %s ...\n", var_name))

  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  # Subset to just id, year, and the variable of interest
  sub_dt <- attr_dt[, .(id, year, val = get(var_name))]
  setkey(sub_dt, id)

  # For each year, join neighbor values and aggregate
  results_list <- vector("list", length(all_years))

  for (yi in seq_along(all_years)) {
    yr <- all_years[yi]

    # Get attribute values for this year
    yr_vals <- sub_dt[year == yr, .(neighbor_id = id, neighbor_val = val)]
    setkey(yr_vals, neighbor_id)

    # Join: for each edge, attach the neighbor's value in this year
    edge_vals <- edges_dt[yr_vals, on = "neighbor_id", nomatch = 0L, allow.cartesian = FALSE]
    # edge_vals now has columns: cell_id, neighbor_id, neighbor_val

    # Remove NA neighbor values before aggregation
    edge_vals <- edge_vals[!is.na(neighbor_val)]

    # Aggregate by cell_id
    if (nrow(edge_vals) > 0L) {
      agg <- edge_vals[, .(
        nmax  = max(neighbor_val),
        nmin  = min(neighbor_val),
        nmean = mean(neighbor_val)
      ), by = cell_id]
      agg[, year := yr]
      results_list[[yi]] <- agg
    }
  }

  result_dt <- rbindlist(results_list, use.names = TRUE)
  setnames(result_dt, c("nmax", "nmin", "nmean"), c(max_col, min_col, mean_col))
  setnames(result_dt, "cell_id", "id")

  return(result_dt)
}

# Compute and merge all neighbor features
for (var_name in neighbor_source_vars) {

  # Remove old columns if they exist (idempotent re-runs)
  old_cols <- paste0(c("neighbor_max_", "neighbor_min_", "neighbor_mean_"), var_name)
  old_cols_present <- old_cols[old_cols %in% names(cell_data)]
  if (length(old_cols_present) > 0L) {
    cell_data[, (old_cols_present) := NULL]
  }

  feat_dt <- compute_neighbor_features_fast(attr_dt, edges_dt, var_name, all_years)

  # Merge back onto cell_data by id + year
  cell_data <- merge(cell_data, feat_dt, by = c("id", "year"), all.x = TRUE)
}

cat("All neighbor features computed and merged.\n")

# ==============================================================================
# STEP 3: Predict with the existing trained Random Forest model.
#
#   The model object (e.g., rf_model) is UNCHANGED. We simply feed the updated
#   cell_data (now containing the neighbor_max_*, neighbor_min_*, neighbor_mean_*
#   columns) into predict().
#
#   The numerical estimand is preserved because the neighbor statistics are
#   computed identically: max, min, and mean of non-NA neighbor values per
#   cell-year, exactly as the original code produced.
# ==============================================================================

# Example prediction call (adjust 'rf_model' and predictor column selection
# to match your existing pipeline):
#
# predictor_cols <- setdiff(names(cell_data), c("id", "year", "response_var"))
# cell_data[, prediction := predict(rf_model, newdata = cell_data[, ..predictor_cols])]

cat("Pipeline complete. Trained RF model is preserved; apply predict() as before.\n")
```

---

## Why This Preserves the Original Numerical Estimand

The original code computes, for each cell-year row, the `max`, `min`, and `mean` of each neighbor source variable across that cell's rook neighbors **in the same year**, excluding `NA` values. The optimized code does exactly the same thing:

1. The edge table encodes the identical spatial neighbor pairs derived from `rook_neighbors_unique` and `id_order`.
2. The join on `neighbor_id` within a single year retrieves exactly the same neighbor attribute values.
3. `NA` removal (`!is.na(neighbor_val)`) and the `max`/`min`/`mean` aggregation match the original logic.
4. Cells with no valid neighbors in a given year receive `NA` via the `all.x = TRUE` merge, matching the original `c(NA, NA, NA)` return.

The trained Random Forest model is never modified — only the feature-engineering preprocessing is accelerated.

### Expected performance on a 16 GB laptop:

- **Edge table construction:** < 5 seconds (single loop over 344K cells).
- **Neighbor stats for all 5 variables × 28 years:** ~2–8 minutes total (vectorized `data.table` joins and grouped aggregations).
- **Peak RAM:** ~3–5 GB (well within 16 GB).
- **Total speedup:** roughly **500–2,500×** versus the original 86+ hour estimate.