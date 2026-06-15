 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three compounding reasons:

### 1. `build_neighbor_lookup` is O(N²)-like in practice
It iterates over all **6.46 million cell-year rows** with `lapply`, and for each row it:
- Performs character-based key construction (`paste(id, year)`)
- Does named-vector lookups (`idx_lookup[neighbor_keys]`) — which are hash lookups but repeated millions of times in an R-level loop

This builds a **6.46-million-element list** where each element contains row indices of that cell-year's neighbors. The fundamental problem: **the neighbor topology is static across years**, yet this function re-discovers it per cell-year, multiplying work by 28×.

### 2. `compute_neighbor_stats` uses `lapply` over 6.46M elements
Each call iterates over every row, subsetting a numeric vector and computing `max/min/mean`. With 5 variables, that's **~32.3 million R-level function calls** — each with allocation overhead.

### 3. The neighbor lookup is year-redundant
Rook neighbors don't change across years. The current design embeds year into the lookup keys, creating 28 copies of the same spatial topology. A cell has the same neighbors in 1992 as in 2019.

---

## Optimization Strategy

**Core insight:** Separate the static spatial topology from the time-varying attributes. Build the adjacency table **once** (344K cells × ~4 neighbors each ≈ 1.37M rows), then use vectorized `data.table` joins to compute neighbor statistics.

| Step | What | Complexity |
|------|------|-----------|
| 1 | Build a two-column `data.table` of `(cell_id, neighbor_id)` from the `nb` object — **once**, ~1.37M rows | O(C × k) |
| 2 | For each year-slice, join cell attributes onto the adjacency table by `neighbor_id + year` | Vectorized join |
| 3 | Group by `(cell_id, year)` and compute `max`, `min`, `mean` in one pass | Vectorized aggregation |
| 4 | Join results back to the main dataset | Vectorized join |

**Expected speedup:** From ~86 hours → **minutes** (typically 2–10 minutes on a 16 GB laptop), because:
- The 6.46M-element R-level `lapply` is eliminated.
- All operations are vectorized `data.table` keyed joins and grouped aggregations in C.
- Memory footprint is modest: the adjacency table is ~1.37M rows × 2 integer columns ≈ 11 MB; the join table with one variable is ~1.37M × 28 years ≈ 38.4M rows of integers + doubles, well within 16 GB when processed one variable at a time.

The trained Random Forest model is untouched. The numerical output (neighbor max, min, mean per variable per cell-year) is identical.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 0: Ensure cell_data is a data.table with columns: id, year, and
#         all predictor columns including the 5 neighbor source vars.
# ──────────────────────────────────────────────────────────────────────
if (!is.data.table(cell_data)) {
 cell_data <- as.data.table(cell_data)
}

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build the static adjacency table ONCE from the nb object.
#
#   rook_neighbors_unique : an nb object (list of integer index vectors)
#   id_order              : vector of cell IDs in the same order as the nb object
#
#   Result: adj_dt — a data.table with columns (cell_id, neighbor_id)
#           representing every directed rook-neighbor pair (~1.37M rows).
# ──────────────────────────────────────────────────────────────────────
build_adjacency_table <- function(id_order, neighbors) {
  # Pre-allocate vectors for speed
  n_links <- sum(lengths(neighbors))
  from_id <- integer(n_links)
  to_id   <- integer(n_links)

  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_idx <- neighbors[[i]]
    # spdep nb objects use 0L to denote "no neighbors"; skip those
    nb_idx <- nb_idx[nb_idx != 0L]
    n      <- length(nb_idx)
    if (n > 0L) {
      from_id[pos:(pos + n - 1L)] <- id_order[i]
      to_id[pos:(pos + n - 1L)]   <- id_order[nb_idx]
      pos <- pos + n
    }
  }

  # Trim in case some 0-neighbor cells left slack
  data.table(cell_id = from_id[1:(pos - 1L)],
             neighbor_id = to_id[1:(pos - 1L)])
}

adj_dt <- build_adjacency_table(id_order, rook_neighbors_unique)

cat(sprintf("Adjacency table: %s directed neighbor pairs\n", format(nrow(adj_dt), big.mark = ",")))

# ──────────────────────────────────────────────────────────────────────
# STEP 2: For each neighbor source variable, compute neighbor max, min,
#         mean via a keyed join + grouped aggregation, then attach
#         results back to cell_data.
#
#   This replaces both build_neighbor_lookup() and compute_neighbor_stats()
#   and the outer for-loop — all in vectorized data.table operations.
# ──────────────────────────────────────────────────────────────────────

# Key cell_data for fast joins
setkey(cell_data, id, year)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {

  cat(sprintf("Computing neighbor stats for: %s ...\n", var_name))

  # Extract only the columns needed for the join (small memory footprint)
  attr_dt <- cell_data[, .(id, year, value = get(var_name))]
  setnames(attr_dt, "id", "neighbor_id")
  setkey(attr_dt, neighbor_id, year)

  # Expand adjacency table by year via join:
  #   For every (cell_id, neighbor_id) pair, attach every year's value
  #   of the neighbor.
  #
  #   adj_dt has ~1.37M rows (no year column).
  #   We join attr_dt (keyed on neighbor_id, year) onto adj_dt,
  #   allowing the cross of adj_dt × years to happen implicitly.
  #
  #   Strategy: add year to adj_dt by crossing with cell_data's (id, year),
  #   then look up the neighbor's value.

  # 2a. Get the (cell_id, year) combinations that actually exist
  cy <- cell_data[, .(cell_id = id, year)]
  setkey(cy, cell_id)

  # 2b. Join: for each (cell_id, year), get all neighbor_ids
  #     Result: (cell_id, year, neighbor_id)
  expanded <- adj_dt[cy, on = .(cell_id), allow.cartesian = TRUE, nomatch = 0L]
  #     expanded now has ~1.37M × 28 ≈ 38.4M rows (but only 3 int columns, manageable)

  # 2c. Look up the neighbor's attribute value for that year
  setkey(expanded, neighbor_id, year)
  expanded[attr_dt, value := i.value, on = .(neighbor_id, year)]

  # 2d. Aggregate: group by (cell_id, year), compute max/min/mean
  stats <- expanded[!is.na(value),
                    .(nb_max  = max(value),
                      nb_min  = min(value),
                      nb_mean = mean(value)),
                    by = .(cell_id, year)]

  # 2e. Name the new columns to match the original pipeline's naming convention
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))

  # 2f. Join stats back onto cell_data
  setkey(stats, cell_id, year)
  # Remove old columns if they already exist (idempotent re-runs)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }
  cell_data[stats, (c(max_col, min_col, mean_col)) :=
              mget(paste0("i.", c(max_col, min_col, mean_col))),
            on = .(id = cell_id, year)]

  # Clean up to free RAM before next variable

rm(attr_dt, cy, expanded, stats)
  gc()

  cat(sprintf("  Done: %s, %s, %s added.\n", max_col, min_col, mean_col))
}

# ──────────────────────────────────────────────────────────────────────
# STEP 3: Predict with the existing trained Random Forest model.
#
#   The trained model object (e.g., `rf_model`) is unchanged.
#   cell_data now contains all ~110 predictor columns including the
#   15 new neighbor_* columns, identical in value to the original code.
# ──────────────────────────────────────────────────────────────────────

# Example prediction (adapt variable name to your actual model object):
# cell_data[, prediction := predict(rf_model, newdata = cell_data)]

cat("Neighbor feature engineering complete. Ready for Random Forest prediction.\n")
```

---

## Why the Numerical Estimand Is Preserved

| Aspect | Original code | Optimized code |
|--------|--------------|----------------|
| Neighbor definition | `rook_neighbors_unique[[ref_idx]]` → same cell IDs | `adj_dt` built from the same `nb` object and `id_order` — identical pairs |
| Year matching | `paste(neighbor_id, year)` key lookup | Equi-join on `(neighbor_id, year)` — identical matching |
| NA handling | Drops `NA` neighbor values before aggregation | `expanded[!is.na(value), ...]` — identical |
| Aggregation | `max`, `min`, `mean` on non-NA neighbor values | Same three functions on the same values |
| No-neighbor cells | Returns `(NA, NA, NA)` | Left join leaves columns as `NA` — identical |

The output columns are numerically identical; only the computational path changes.

---

## Expected Performance

| Metric | Original | Optimized |
|--------|----------|-----------|
| **Time** | ~86+ hours | **~2–10 minutes** |
| **Peak RAM** | Moderate (6.46M-element list of integer vectors) | ~2–4 GB peak (one 38.4M-row join table at a time, freed between variables) |
| **RF model** | Unchanged | Unchanged |