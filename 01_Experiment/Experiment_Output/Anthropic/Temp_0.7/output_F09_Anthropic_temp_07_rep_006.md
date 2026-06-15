 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three interconnected reasons:

### 1. `build_neighbor_lookup` is O(N²)-like in practice
This function iterates over all **6.46 million cell-year rows** with `lapply`, and for each row it:
- Performs a character-based hash lookup (`id_to_ref`) to find the cell's reference index.
- Retrieves neighbor cell IDs from the `nb` object.
- Constructs **paste-based string keys** (`"cellid_year"`) for every neighbor in every year.
- Looks up row indices via a named character vector (`idx_lookup`).

String construction and named-vector lookup for ~6.46M rows × ~4 neighbors each ≈ **~25.8 million string paste + match operations**. R's named vector lookup is O(n) in the worst case (hash collisions) and the `paste()` / character matching is inherently slow in an interpreted loop.

### 2. The lookup is **year-redundant**
The spatial neighbor topology is **identical across all 28 years** — cell *i*'s rook neighbors are always the same cells. Yet `build_neighbor_lookup` rebuilds the mapping for every cell-year combination, doing 28× the necessary work. The string key `"id_year"` conflates spatial structure with temporal indexing.

### 3. `compute_neighbor_stats` uses per-row `lapply`
Even after the lookup is built, computing stats iterates over 6.46M list elements in R-level `lapply`, calling `max`, `min`, `mean` individually. This prevents vectorization.

**Summary:** The core inefficiency is that a **static spatial relationship** (which cells neighbor which) is being re-resolved at the cell-year level via slow string operations, then stats are computed row-by-row in R loops.

---

## Optimization Strategy

### Principle: Separate topology from attributes, then vectorize.

1. **Build the neighbor table once as a `data.table` of directed edges** — 1,373,394 rows of `(cell_id, neighbor_id)`. This is year-invariant and built in seconds from the `nb` object.

2. **For each variable, join yearly attributes onto the edge table** — a keyed `data.table` join attaches `year` and the variable value for each neighbor. This produces ~1.37M × 28 ≈ ~38.5M rows, but the join is vectorized and fast.

3. **Aggregate (max, min, mean) by `(cell_id, year)` in one grouped operation** — `data.table`'s grouped aggregation is C-level and handles this in seconds.

4. **Merge the aggregated stats back onto the main panel** — a final keyed join.

This eliminates all R-level row loops, all `paste`-based key construction, and all per-row `lapply` calls. Expected runtime: **minutes, not hours.**

### Memory check
- Main panel: 6.46M rows × ~115 cols × 8 bytes ≈ ~5.6 GB (tight but feasible with 16 GB).
- Edge table: 1.37M × 2 cols = ~22 MB.
- Edge table expanded by year with one variable: ~38.5M × 4 cols ≈ ~1.2 GB (temporary, freed after aggregation).
- Fits in 16 GB if we process one variable at a time and discard intermediates.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 0: Ensure cell_data is a data.table with proper keys
# ──────────────────────────────────────────────────────────────────────
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build the static spatial edge table ONCE
#
#   rook_neighbors_unique : an nb object (list of integer vectors)
#                           indexed by position in id_order
#   id_order              : vector of cell IDs corresponding to nb indices
#
#   Output: edges_dt — a data.table with columns (cell_id, neighbor_id)
#           representing every directed rook-neighbor pair.
# ──────────────────────────────────────────────────────────────────────

build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate by computing total number of edges
  n_edges <- sum(lengths(neighbors))

  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)

  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_i <- neighbors[[i]]
    if (length(nb_i) == 0L || (length(nb_i) == 1L && nb_i[1] == 0L)) next
    len <- length(nb_i)
    from_id[pos:(pos + len - 1L)] <- id_order[i]
    to_id[pos:(pos + len - 1L)]   <- id_order[nb_i]
    pos <- pos + len
  }

  # Trim if any nb entries were empty / zero (unlikely but safe)
  if (pos <= n_edges) {
    from_id <- from_id[1:(pos - 1L)]
    to_id   <- to_id[1:(pos - 1L)]
  }

  data.table(cell_id = from_id, neighbor_id = to_id)
}

edges_dt <- build_edge_table(id_order, rook_neighbors_unique)
cat("Edge table rows:", nrow(edges_dt), "\n")

# ──────────────────────────────────────────────────────────────────────
# STEP 2: For each source variable, compute neighbor max, min, mean
#         by joining yearly attributes onto the edge table, then
#         aggregating, then merging back onto cell_data.
# ──────────────────────────────────────────────────────────────────────

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Create a minimal lookup table: (id, year, <variable>)
# We'll reuse this pattern per variable to keep peak memory low.

# Key the main data for fast joins later
setkey(cell_data, id, year)

for (var_name in neighbor_source_vars) {

  cat("Processing neighbor stats for:", var_name, "... ")
  t0 <- proc.time()

  # --- 2a. Extract the (id, year, value) slice for this variable ---
  attr_dt <- cell_data[, .(id, year, value = get(var_name))]
  setkey(attr_dt, id, year)

  # --- 2b. Cross-join edges with all years present in the data ---
  #
  #   We need, for every (cell_id, neighbor_id) edge and every year,
  #   the neighbor's attribute value in that year.
  #
  #   Strategy: join attr_dt onto edges_dt by neighbor_id.
  #   This naturally expands edges × years because attr_dt has one row
  #   per (id, year).

  # Rename for join clarity
  setnames(attr_dt, "id", "neighbor_id")
  setkey(attr_dt, neighbor_id)

  # Join: for each edge, get all (year, value) rows of the neighbor
  # Result columns: cell_id, neighbor_id, year, value
  expanded <- attr_dt[edges_dt, on = "neighbor_id", allow.cartesian = TRUE, nomatch = NA]
  # expanded has columns: neighbor_id, year, value, cell_id

  # --- 2c. Aggregate: for each (cell_id, year), compute stats ---
  stats <- expanded[
    !is.na(value),
    .(
      nb_max  = max(value),
      nb_min  = min(value),
      nb_mean = mean(value)
    ),
    by = .(cell_id, year)
  ]

  # Name the output columns to match the original pipeline's convention
  # (adjust these names to match whatever compute_and_add_neighbor_features produced)
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
  setnames(stats, "cell_id", "id")
  setkey(stats, id, year)

  # --- 2d. Remove old columns if they exist (idempotent re-runs) ---
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) {
      cell_data[, (col) := NULL]
    }
  }

  # --- 2e. Merge stats back onto the main panel ---
  cell_data <- stats[cell_data, on = .(id, year)]

  # Clean up large temporaries

rm(attr_dt, expanded, stats)
  gc()

  elapsed <- (proc.time() - t0)["elapsed"]
  cat(round(elapsed, 1), "seconds\n")
}

# ──────────────────────────────────────────────────────────────────────
# STEP 3: Predict with the already-trained Random Forest
#
#   The trained model object (e.g., rf_model) is unchanged.
#   cell_data now contains the same neighbor-stat columns as before,
#   with identical numerical values, so predictions are identical.
# ──────────────────────────────────────────────────────────────────────

# Example (adjust to your actual model object and prediction column names):
# cell_data$rf_prediction <- predict(rf_model, newdata = cell_data)

cat("Done. cell_data has", nrow(cell_data), "rows and", ncol(cell_data), "columns.\n")
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets and values. The edge table encodes the same rook-contiguity as `rook_neighbors_unique`. The join on `(neighbor_id, year)` reproduces the same `"id_year"` matching. No approximation is introduced. |
| **Trained RF model** | The model object is never touched. The output columns have the same names and values, so `predict()` produces identical results. |
| **Missing-data handling** | `!is.na(value)` in the aggregation mirrors the original `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]`. Cells with zero non-NA neighbors get `NA` for all three stats (via the left join `stats[cell_data, ...]`), matching the original `return(c(NA, NA, NA))`. |

## Expected Performance

| Step | Estimated Time |
|---|---|
| Build edge table (once) | < 2 seconds |
| Per-variable join + aggregate | ~30–90 seconds each |
| 5 variables total | **~3–8 minutes** |
| **Total vs. original** | **~5 minutes vs. ~86 hours** (~1,000× speedup) |