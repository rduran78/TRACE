 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(N²)-like in practice
This function iterates over **all 6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs **year-specific string keys** (`paste(neighbor_id, year, sep="_")`) and matches them against a named character vector (`idx_lookup`) of length 6.46 million.

Named vector lookup in R via `idx_lookup[neighbor_keys]` is hash-based, but doing ~6.46 million calls to `paste()` and hash lookups — each producing ~4 neighbor keys — means roughly **25+ million string constructions and hash probes**, all inside a sequential `lapply` over 6.46 million iterations. The per-iteration overhead of R's interpreted loop dominates.

### 2. `compute_neighbor_stats` is similarly slow
It runs another `lapply` over 6.46 million entries, subsetting a numeric vector and computing `max/min/mean` per entry. While each operation is trivial, 6.46M R-level function calls with list allocation is very expensive.

### 3. The core architectural flaw: conflating spatial topology with temporal panel structure
The neighbor **topology** is purely spatial — it depends only on cell identity, not on year. There are only **344,208 unique cells** with ~1.37M directed neighbor pairs. But the current code rebuilds neighbor row-indices for every cell × year combination (6.46M rows), doing redundant work: the same cell has the same neighbors in every year.

---

## Optimization Strategy

**Key insight:** Separate the *spatial adjacency structure* (build once, 344K cells) from the *temporal attribute join* (vectorized, 6.46M rows).

### Step-by-step plan:

1. **Build a static edge table once** — a two-column `data.table` of `(cell_id, neighbor_id)` with ~1.37M rows. This is year-independent.

2. **For each variable, use a vectorized `data.table` join** — join cell-year attributes onto the edge table by `(neighbor_id, year)`, then group by `(cell_id, year)` to compute `max`, `min`, `mean`. This replaces both `build_neighbor_lookup` and `compute_neighbor_stats` with a single vectorized operation per variable.

3. **Join results back** to the main dataset.

This eliminates all R-level row-wise loops, all string-key construction, and all per-row `lapply` calls. The `data.table` grouped aggregation is implemented in C and will handle 1.37M × 28 ≈ 38.4M edge-year rows efficiently.

### Expected speedup:
- From ~86 hours to **minutes** (likely 5–15 minutes total for all 5 variables on a 16 GB laptop).

### Memory check:
- Edge table: ~1.37M rows × 2 integer cols ≈ 11 MB.
- Expanded edge-year table: ~38.4M rows × 3 cols (cell, neighbor, year) ≈ 460 MB.
- With one attribute column joined: ~38.4M × 4 cols ≈ 610 MB.
- Well within 16 GB.

---

## Working R Code

```r
library(data.table)

# ============================================================
# 1. Build the static spatial edge table ONCE
#    Input: id_order (vector of cell IDs in nb-object order)
#           rook_neighbors_unique (spdep nb object, length = # cells)
#    Output: edge_dt — a data.table with columns (cell_id, neighbor_id)
# ============================================================

build_edge_table <- function(id_order, neighbors) {
  # neighbors is an nb object: list of integer index vectors
  n <- length(neighbors)
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(neighbors))

  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)

  pos <- 1L
  for (i in seq_len(n)) {
    nb_idx <- neighbors[[i]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) next
    len <- length(nb_idx)
    from_id[pos:(pos + len - 1L)] <- id_order[i]
    to_id[pos:(pos + len - 1L)]   <- id_order[nb_idx]
    pos <- pos + len
  }

  # Trim if any nb entries were empty (0-neighbor cells)
  if (pos <= n_edges) {
    from_id <- from_id[seq_len(pos - 1L)]
    to_id   <- to_id[seq_len(pos - 1L)]
  }

  data.table(cell_id = from_id, neighbor_id = to_id)
}

# ============================================================
# 2. Compute neighbor stats for one variable via vectorized join
#    Input: cell_dt    — data.table with columns: id, year, <var_name>
#           edge_dt    — from step 1
#           var_name   — character, name of the source variable
#    Output: cell_dt with three new columns appended:
#            <var_name>_neighbor_max, _neighbor_min, _neighbor_mean
# ============================================================

compute_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {
  # Extract only the columns we need for the join
  # neighbor attributes: keyed by (id, year)
  attr_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setkey(attr_dt, id, year)

  # Expand edges across all years present in the data
  # Instead of a full cross-join (expensive), join edges onto data's (cell_id, year)
  # Step A: get unique (cell_id, year) pairs
  cell_years <- cell_dt[, .(cell_id = id, year)]

  # Step B: join edges to get (cell_id, year, neighbor_id)
  #         This is an inner join: for each cell-year, attach its neighbor IDs
  setkey(edge_dt, cell_id)
  setkey(cell_years, cell_id)
  edge_year <- edge_dt[cell_years, on = "cell_id", allow.cartesian = TRUE, nomatch = NULL]
  # edge_year now has columns: cell_id, neighbor_id, year
  # Rows: ~1.37M neighbors × 28 years ≈ 38.4M (but done via join, not cross)

  # Step C: join neighbor attribute values
  edge_year[attr_dt, val := i.val, on = .(neighbor_id = id, year)]

  # Step D: aggregate by (cell_id, year)
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")

  stats <- edge_year[!is.na(val),
    .(nmax = max(val), nmin = min(val), nmean = mean(val)),
    by = .(cell_id, year)
  ]

  setnames(stats, c("nmax", "nmin", "nmean"), c(max_col, min_col, mean_col))

  # Step E: merge back onto cell_dt
  # Remove old columns if they exist (idempotent re-runs)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }

  cell_dt <- merge(cell_dt, stats,
    by.x = c("id", "year"),
    by.y = c("cell_id", "year"),
    all.x = TRUE
  )

  cell_dt
}

# ============================================================
# 3. Full pipeline
# ============================================================

# Convert to data.table if not already
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# --- Build the edge table ONCE ---
edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
cat("Edge table built:", nrow(edge_dt), "directed edges\n")

# --- Compute neighbor features for each source variable ---
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "...\n")
  t0 <- proc.time()
  cell_data <- compute_neighbor_features_fast(cell_data, edge_dt, var_name)
  elapsed <- (proc.time() - t0)[["elapsed"]]
  cat("  Done in", round(elapsed, 1), "seconds\n")
}

# --- Predict with the existing trained Random Forest ---
# (The model object, e.g. `rf_model`, is already in memory and is NOT retrained)
# Ensure column order / names match what the model expects:
pred_vars <- names(cell_data)[names(cell_data) %in% rf_model$forest$independent.variable.names]
# Or however your model stores predictor names (ranger, randomForest, etc.)

cell_data[, prediction := predict(rf_model, data = cell_data[, ..pred_vars])$predictions]

cat("Pipeline complete.\n")
```

---

## Summary of Changes and Guarantees

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` are computed on exactly the same neighbor sets and the same variable values. NA handling is identical (NAs excluded before aggregation; cells with no valid neighbors get `NA`). The estimand is preserved. |
| **Trained RF model** | The model object is never modified or retrained. Only `predict()` is called. |
| **Correctness of topology** | `build_edge_table` reads the same `rook_neighbors_unique` nb object and `id_order` mapping. Each `neighbors[[i]]` indexes into `id_order` identically to the original code. |
| **Speed** | Eliminates 6.46M R-level loop iterations (×2 functions ×5 variables = ~64.6M iterations). Replaced by ~5 vectorized `data.table` grouped aggregations over ~38M rows each. Expected wall time: **5–15 minutes** vs. 86+ hours. |
| **Memory** | Peak additional memory ~1–2 GB for the expanded edge-year table, well within 16 GB. |