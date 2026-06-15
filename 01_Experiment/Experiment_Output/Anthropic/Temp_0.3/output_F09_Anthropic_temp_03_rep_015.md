 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of two fundamental bottlenecks:

### 1. `build_neighbor_lookup` is O(n²)-like in practice
This function iterates over **all 6.46 million cell-year rows** and, for each row, performs:
- A named-vector lookup (`id_to_ref`) — O(1) amortized but with high constant factor for character hashing over millions of keys.
- Construction of `neighbor_keys` by pasting cell IDs with years — millions of `paste()` calls.
- A named-vector lookup (`idx_lookup`) on those keys — again, character hashing over a 6.46M-entry named vector.

The result is a **list of 6.46 million integer vectors**, each built via expensive character-key operations. This is the single largest bottleneck.

### 2. `compute_neighbor_stats` uses `lapply` over 6.46M elements
Even though each iteration is small, the R-level loop overhead across 6.46 million rows, repeated for 5 variables (= 32.3 million iterations total), is substantial.

### 3. The neighbor topology is **year-invariant** but rebuilt per cell-year
The rook-neighbor structure is purely spatial — cell A's neighbors are the same in 1992 as in 2019. Yet the lookup is constructed at the cell-year level, redundantly encoding the same spatial relationships 28 times.

---

## Optimization Strategy

**Core insight:** Separate the *spatial topology* (which cells are neighbors — fixed) from the *temporal attributes* (variable values per year — varying). Then use vectorized joins and grouped aggregations instead of row-wise R loops.

**Steps:**

1. **Build a cell-level edge table once** — a two-column `data.table` of `(cell_id, neighbor_id)` with ~1.37M rows. This is year-invariant.

2. **Join yearly attributes onto the edge table** — for each variable and year, join the variable's value from the neighbor cell onto the edge table. This is a keyed `data.table` join: O(n) and vectorized in C.

3. **Aggregate neighbor stats** — group by `(cell_id, year)` and compute `max`, `min`, `mean` in one vectorized `data.table` operation.

4. **Join results back** to the main dataset.

This replaces 6.46M × 5 R-level `lapply` iterations with a handful of vectorized `data.table` joins and group-by aggregations. Expected runtime: **minutes, not hours**.

**Memory:** The edge table is ~1.37M rows × 2 integer columns ≈ 11 MB. Joined with year expansion: 1.37M × 28 years ≈ 38.4M rows × a few columns ≈ manageable well within 16 GB.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 0: Convert cell_data to data.table (if not already)
# ──────────────────────────────────────────────────────────────────────
setDT(cell_data)

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build a year-invariant spatial edge table ONCE
#
#   rook_neighbors_unique : an nb object (list of integer index vectors)
#   id_order              : vector mapping positional index -> cell id
#
#   We produce edges_dt: data.table with columns (id, neighbor_id)
#   representing every directed rook-neighbor pair.
# ──────────────────────────────────────────────────────────────────────

build_edge_table <- function(id_order, neighbors_nb) {
  # neighbors_nb[[i]] contains integer indices into id_order for the

  # neighbors of id_order[i]. Index 0 means no neighbors (spdep convention).
  from_list <- lapply(seq_along(neighbors_nb), function(i) {
    nb_idx <- neighbors_nb[[i]]
    nb_idx <- nb_idx[nb_idx != 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(id = id_order[i], neighbor_id = id_order[nb_idx])
  })
  rbindlist(from_list, use.names = FALSE)
}

edges_dt <- build_edge_table(id_order, rook_neighbors_unique)
# edges_dt has ~1,373,394 rows: (id, neighbor_id)

cat("Edge table rows:", nrow(edges_dt), "\n")

# ──────────────────────────────────────────────────────────────────────
# STEP 2: For each neighbor source variable, compute neighbor max,
#          min, and mean via vectorized join + grouped aggregation.
# ──────────────────────────────────────────────────────────────────────

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Ensure keys are set for fast joins
setkey(cell_data, id, year)

# Extract the unique years present in the panel
all_years <- sort(unique(cell_data$year))

# Cross-join edges with years to get the full (id, neighbor_id, year) table.
# ~1.37M edges × 28 years ≈ 38.4M rows — fits in memory.
edges_by_year <- CJ_dt <- edges_dt[, .(year = all_years), by = .(id, neighbor_id)]

cat("Edge-year table rows:", nrow(edges_by_year), "\n")

# Key for joining neighbor attributes
setkey(edges_by_year, neighbor_id, year)

for (var_name in neighbor_source_vars) {

  cat("Processing neighbor features for:", var_name, "...\n")

  # Column names for the three output features
  col_max  <- paste0("n_max_",  var_name)
  col_min  <- paste0("n_min_",  var_name)
  col_mean <- paste0("n_mean_", var_name)

  # --- Join the neighbor's attribute value onto the edge-year table ---
  # We need cell_data[, .(id, year, <var_name>)] keyed by (id, year)
  # and we join on (neighbor_id, year) == (id, year)

  # Subset for the join: only the columns we need
  attr_dt <- cell_data[, .(id, year, val = get(var_name))]
  setkey(attr_dt, id, year)

  # Join: for each (id, neighbor_id, year) row in edges_by_year,
  # look up the neighbor's value by matching neighbor_id==id, year==year
  edges_by_year[attr_dt, neighbor_val := i.val, on = .(neighbor_id = id, year = year)]

  # --- Aggregate: group by (id, year), compute max/min/mean of neighbor_val ---
  agg <- edges_by_year[
    !is.na(neighbor_val),
    .(
      nmax  = max(neighbor_val),
      nmin  = min(neighbor_val),
      nmean = mean(neighbor_val)
    ),
    by = .(id, year)
  ]

  setnames(agg, c("nmax", "nmin", "nmean"), c(col_max, col_min, col_mean))
  setkey(agg, id, year)

  # --- Remove old columns if they exist (idempotency) ---
  for (cc in c(col_max, col_min, col_mean)) {
    if (cc %in% names(cell_data)) cell_data[, (cc) := NULL]
  }

  # --- Join aggregated stats back onto cell_data ---
  cell_data <- agg[cell_data, on = .(id, year)]

  # Clean up the temporary column in edges_by_year
  edges_by_year[, neighbor_val := NULL]

  cat("  Done:", col_max, col_min, col_mean, "\n")
}

# ──────────────────────────────────────────────────────────────────────
# STEP 3: Predict with the already-trained Random Forest model
#
#   The trained model object (e.g., `rf_model`) is unchanged.
#   The numerical estimand is preserved because we compute identical
#   max, min, mean statistics — just via vectorized operations
#   instead of row-wise R loops.
# ──────────────────────────────────────────────────────────────────────

# Example prediction (adjust to your actual model object name):
# cell_data[, prediction := predict(rf_model, newdata = cell_data)]

cat("All neighbor features computed. Ready for Random Forest prediction.\n")
```

---

## Why the Numerical Estimand Is Preserved

The vectorized `data.table` path computes **exactly the same** `max()`, `min()`, and `mean()` over **exactly the same** set of non-NA neighbor values for each `(cell_id, year)` combination. The only change is *how* those sets are assembled (keyed join vs. character-key named-vector lookup) and *how* the aggregation is dispatched (C-level `data.table` group-by vs. R-level `lapply`). The arithmetic is identical, so the trained Random Forest model receives the same feature values and need not be retrained.

---

## Expected Performance

| Component | Old (base R `lapply`) | New (`data.table` vectorized) |
|---|---|---|
| Build neighbor lookup | ~hours (6.46M `paste` + hash lookups) | ~seconds (1.37M-row edge table) |
| Compute stats (×5 vars) | ~hours (32.3M R-loop iterations) | ~minutes (keyed join + group-by on 38.4M rows) |
| **Total** | **~86+ hours** | **~5–15 minutes** |

The 38.4M-row `edges_by_year` table consumes roughly 38.4M × 3 columns × 8 bytes ≈ 0.9 GB, well within the 16 GB RAM budget. Peak memory during the join (with the temporary `neighbor_val` column) adds another ~300 MB, still comfortable.