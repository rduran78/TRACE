 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three interconnected reasons:

### 1. Row-level `lapply` over 6.46 million rows in `build_neighbor_lookup`
For every single cell-year row, the function performs string pasting (`paste(id, year, sep="_")`), character-based lookups into named vectors (`id_to_ref`, `idx_lookup`), and `NA` filtering. Named-vector lookups in R are hash-table operations, but doing ~6.46 million of them—each touching multiple neighbors—creates billions of character-match operations. This alone can take tens of hours.

### 2. Row-level `lapply` over 6.46 million rows in `compute_neighbor_stats`
For each row, a small numeric vector is subsetted, `NA`-filtered, and then `max`/`min`/`mean` are computed. The per-call overhead of `lapply` + anonymous function + three summary calls, repeated 6.46M × 5 variables = ~32.3 million invocations, is enormous.

### 3. Redundant recomputation of the same spatial topology every row
The neighbor graph is **time-invariant**: cell A's rook neighbors are the same in 1992 as in 2019. Yet `build_neighbor_lookup` re-resolves neighbor relationships for every cell-year combination, inflating the problem from ~344K cells to ~6.46M rows. The lookup should be built **once at the cell level** (344K entries), and yearly attribute values should be **joined** onto that static topology.

### Summary
| Bottleneck | Scale | Root Cause |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `lapply` iterations | Character key lookups per cell-year instead of per cell |
| `compute_neighbor_stats` | 6.46M × 5 = 32.3M iterations | Scalar R-loop instead of vectorized group operation |
| Conceptual | 18.7× redundant work | Topology rebuilt per year instead of once |

---

## Optimization Strategy

**Core idea:** Separate the *spatial topology* (static, 344K cells) from the *temporal attributes* (yearly, 6.46M rows). Build a **directed edge table** once from the `nb` object, then use a vectorized `data.table` join-and-aggregate to compute neighbor stats for all cell-years simultaneously.

### Steps

1. **Build a static edge table** from `rook_neighbors_unique` (the `nb` object). This is a two-column data.table: `(focal_id, neighbor_id)` with ~1.37M rows. Done once, reusable forever.

2. **Join yearly attributes onto the edge table.** For each year, every edge `(focal, neighbor)` gets the neighbor's attribute value via a keyed `data.table` join. This produces an "expanded" table of ~1.37M × 28 years ≈ 38.5M rows (but processed vectorially, not in a loop).

3. **Aggregate by `(focal_id, year)`** to compute `max`, `min`, `mean` of each neighbor variable in one vectorized pass.

4. **Join the aggregated stats back** onto the main `cell_data` table.

5. **Predict** with the existing trained Random Forest model (unchanged).

### Expected speedup
- `data.table` keyed joins and grouped aggregations on ~38.5M rows run in seconds to low minutes on 16 GB RAM.
- Total estimated time: **2–10 minutes** instead of 86+ hours (roughly 500–2500× speedup).
- Memory: the edge table × years × 5 variables is manageable; peak ~2–4 GB.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 0: Ensure cell_data is a data.table with columns "id" and "year"
# ──────────────────────────────────────────────────────────────────────
cell_data <- as.data.table(cell_data)

# id_order:             character or integer vector of cell IDs in the same
#                       order as rook_neighbors_unique (the nb object)
# rook_neighbors_unique: an nb object (list of integer index vectors)

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build a static directed edge table ONCE from the nb object
#         ~1.37 M rows, two integer/character columns
# ──────────────────────────────────────────────────────────────────────
build_edge_table <- function(id_order, nb_obj) {
  # nb_obj[[i]] contains integer indices into id_order for the neighbors of

  # the i-th cell.  A 0-length integer(0) means no neighbors.
  focal_idx    <- rep(seq_along(nb_obj), lengths(nb_obj))
  neighbor_idx <- unlist(nb_obj, use.names = FALSE)

  data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# This table is time-invariant and can be serialized for future reuse:
# fst::write_fst(edge_dt, "edge_table.fst")

cat(sprintf("Edge table: %s directed edges\n", format(nrow(edge_dt), big.mark = ",")))

# ──────────────────────────────────────────────────────────────────────
# STEP 2 & 3: For each neighbor source variable, join yearly attributes
#             onto the edge table and aggregate in one vectorized pass
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Key cell_data for fast joins
setkey(cell_data, id, year)

for (var in neighbor_source_vars) {

  cat(sprintf("Computing neighbor stats for: %s\n", var))

  # Subset only the columns we need for the neighbor lookup

  # (neighbor_id will be joined on "id", plus we need "year" and the variable)
  attr_dt <- cell_data[, .(id, year, value = get(var))]
  setnames(attr_dt, "id", "neighbor_id")
  setkey(attr_dt, neighbor_id, year)

  # Expand edges × years: join neighbor attribute onto every (focal, neighbor, year)
  # This is a many-to-many join: each edge appears once per year
  expanded <- edge_dt[attr_dt, on = "neighbor_id", allow.cartesian = TRUE, nomatch = NULL]
  # expanded now has columns: focal_id, neighbor_id, year, value

  # Drop NAs in the variable (mirrors original behavior)
  expanded <- expanded[!is.na(value)]

  # Aggregate: one row per (focal_id, year)
  agg <- expanded[, .(
    nb_max  = max(value),
    nb_min  = min(value),
    nb_mean = mean(value)
  ), by = .(focal_id, year)]

  # Rename columns to match the original pipeline's naming convention
  # Original: e.g., ntl_neighbor_max, ntl_neighbor_min, ntl_neighbor_mean
  max_col  <- paste0(var, "_neighbor_max")
  min_col  <- paste0(var, "_neighbor_min")
  mean_col <- paste0(var, "_neighbor_mean")
  setnames(agg, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
  setnames(agg, "focal_id", "id")

  # ──────────────────────────────────────────────────────────────────
  # STEP 4: Join aggregated neighbor stats back onto cell_data
  # ──────────────────────────────────────────────────────────────────

  # Remove old columns if they already exist (idempotent reruns)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }

  cell_data <- merge(cell_data, agg, by = c("id", "year"), all.x = TRUE)

  # Clean up

  rm(attr_dt, expanded, agg)
}

# Re-key after all merges
setkey(cell_data, id, year)

cat("Neighbor feature engineering complete.\n")

# ──────────────────────────────────────────────────────────────────────
# STEP 5: Predict with the EXISTING trained Random Forest (unchanged)
# ──────────────────────────────────────────────────────────────────────
# The trained model object (e.g., `rf_model`) is used as-is.
# Ensure the predictor matrix matches the model's expected feature names.

# Example (adjust to your actual model object and predict call):
# cell_data[, prediction := predict(rf_model, newdata = .SD)]

# If using ranger:
# preds <- predict(rf_model, data = cell_data)
# cell_data[, prediction := preds$predictions]

# If using randomForest:
# cell_data[, prediction := predict(rf_model, newdata = cell_data)]
```

---

## Why This Preserves Correctness

| Requirement | How it is preserved |
|---|---|
| **Trained RF model untouched** | No retraining; only the feature-engineering step is replaced. The same 15 neighbor-stat columns (5 vars × 3 stats) are produced with identical names and identical numerical values. |
| **Original numerical estimand** | `max`, `min`, `mean` are computed on the same set of non-NA neighbor values per `(focal_id, year)` as the original code. The `data.table` aggregation is algebraically identical. |
| **Rook neighbor topology** | The edge table is built directly from the same `rook_neighbors_unique` nb object and `id_order`. No spatial relationships are altered. |
| **NA handling** | `expanded[!is.na(value)]` mirrors the original `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]`. Cells with zero valid neighbors get `NA` via the `all.x = TRUE` left join, matching the original `return(c(NA, NA, NA))`. |

---

## Performance Estimate on 16 GB Laptop

| Step | Estimated time | Estimated peak RAM |
|---|---|---|
| Build edge table | < 1 second | ~50 MB |
| Join + aggregate (per variable) | 30–90 seconds | ~1.5–2.5 GB |
| All 5 variables | **3–8 minutes total** | ~3–4 GB peak |
| RF prediction (unchanged) | Same as before | Same as before |

**Total: ~5–10 minutes** versus the original ~86 hours.