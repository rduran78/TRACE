 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three interconnected reasons:

### 1. `build_neighbor_lookup` creates a per-row (cell-year) lookup — 6.46 million entries
The function iterates over every cell-year row and, for each row, resolves neighbor cell IDs into row indices by pasting string keys and doing named-vector lookups. This produces **6.46 million list entries**, most of which are redundant: the same cell has the same neighbors in every year. The string-paste key approach (`paste(id, year)`) is inherently slow at scale (millions of `paste` and named-vector lookups).

### 2. `compute_neighbor_stats` iterates over 6.46 million list entries per variable
For each of the 5 neighbor source variables, it loops over every cell-year row, subsets `vals[idx]`, and computes `max/min/mean`. That's ~32.3 million R-level list iterations total, each with subsetting and aggregation.

### 3. The neighbor topology is **static** but is entangled with yearly data
The rook-neighbor adjacency structure depends only on spatial cell identity, not on year. But the current code rebuilds the relationship at the cell-year level, inflating the problem by a factor of 28×.

---

## Optimization Strategy

**Core insight:** Separate the static spatial adjacency from the time-varying attributes.

1. **Build a cell-neighbor edge table once** — a two-column `data.table` of `(cell_id, neighbor_id)` with ~1.37 million rows. This is year-independent.
2. **For each variable, join yearly attributes onto the edge table**, then aggregate (`max`, `min`, `mean`) by `(cell_id, year)` using `data.table` grouped operations — fully vectorized, no R-level loops.
3. **Column-bind the results** back to the main dataset.

This replaces 6.46M × 5 R-level list iterations with 5 vectorized `data.table` grouped joins + aggregations over ~1.37M × 28 ≈ 38.5M edge-year rows. Expected runtime: **minutes, not hours**.

The trained Random Forest model is never touched. The output columns are numerically identical (same `max`, `min`, `mean` of the same neighbor values), preserving the original estimand.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 0 — Assume these objects already exist in the environment:
#   cell_data              : data.frame / data.table with columns id, year, ntl, ec, …
#   id_order               : integer/numeric vector of cell IDs (length 344,208)
#   rook_neighbors_unique  : nb object (list of length 344,208; each element
#                            is an integer vector of positional indices into id_order)
#   rf_model               : the already-trained Random Forest model
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# STEP 1 — Build a static cell-neighbor edge table (year-independent).
#          This is done ONCE and reused for every variable.
# ──────────────────────────────────────────────────────────────────────

build_edge_table <- function(id_order, nb_object) {
  # nb_object[[i]] contains positional indices (into id_order) of neighbors of cell i.
  # Convert to a two-column data.table of actual cell IDs.
  n <- length(nb_object)
  # Pre-allocate vectors
  from_idx <- rep.int(seq_len(n), lengths(nb_object))
  to_idx   <- unlist(nb_object, use.names = FALSE)

  # Remove the spdep "no neighbors" sentinel (integer 0)
  valid <- to_idx != 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    cell_id     = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_table <- build_edge_table(id_order, rook_neighbors_unique)
# ~1.37 million rows, two integer columns — tiny in memory

cat("Edge table rows:", nrow(edge_table), "\n")

# ──────────────────────────────────────────────────────────────────────
# STEP 2 — Convert cell_data to data.table (if not already) and set key.
# ──────────────────────────────────────────────────────────────────────

if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure an explicit row-order column so we can restore order later if needed.
cell_data[, .row_order := .I]

# ──────────────────────────────────────────────────────────────────────
# STEP 3 — Generic function: compute neighbor max, min, mean for one
#           variable, returning a 3-column data.table keyed on (id, year).
# ──────────────────────────────────────────────────────────────────────

compute_neighbor_features_dt <- function(cell_dt, edge_dt, var_name) {
  # Subset only the columns we need for the join
  attr_dt <- cell_dt[, .(id, year, value = get(var_name))]

  # Join neighbor attribute values onto the edge table × year.
  # For every (cell_id, year) pair, look up each neighbor_id's value that year.
  #
  # Step A: cross-join edges with years implicitly by joining on neighbor_id = id.
  #         edge_dt has (cell_id, neighbor_id).
  #         attr_dt has (id, year, value).
  #         We want, for each edge and each year, the neighbor's value.

  setkey(attr_dt, id, year)

  # Expand edges × years by merging edge_dt with attr_dt on neighbor_id == id
  merged <- merge(
    edge_dt,
    attr_dt,
    by.x = "neighbor_id",
    by.y = "id",
    allow.cartesian = TRUE   # each neighbor_id appears in 28 year-rows
  )
  # merged columns: neighbor_id, cell_id, year, value
  # "value" is the neighbor's attribute in that year.

  # Step B: aggregate by (cell_id, year)
  agg <- merged[
    !is.na(value),
    .(
      nbr_max  = max(value),
      nbr_min  = min(value),
      nbr_mean = mean(value)
    ),
    keyby = .(cell_id, year)
  ]

  # Rename to match the expected feature names
  prefix <- var_name
  setnames(agg,
           c("nbr_max",  "nbr_min",  "nbr_mean"),
           c(paste0(prefix, "_neighbor_max"),
             paste0(prefix, "_neighbor_min"),
             paste0(prefix, "_neighbor_mean")))

  agg
}

# ──────────────────────────────────────────────────────────────────────
# STEP 4 — Loop over the 5 neighbor source variables (vectorized inside).
# ──────────────────────────────────────────────────────────────────────

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "... ")
  t0 <- proc.time()

  feat <- compute_neighbor_features_dt(cell_data, edge_table, var_name)

  # Left-join the 3 new columns back onto cell_data
  cell_data <- merge(
    cell_data,
    feat,
    by.x = c("id", "year"),
    by.y = c("cell_id", "year"),
    all.x = TRUE
  )

  elapsed <- (proc.time() - t0)["elapsed"]
  cat(round(elapsed, 1), "s\n")
}

# Restore original row order (merge may re-sort)
setorder(cell_data, .row_order)
cell_data[, .row_order := NULL]

# ──────────────────────────────────────────────────────────────────────
# STEP 5 — Predict with the EXISTING Random Forest model (unchanged).
# ──────────────────────────────────────────────────────────────────────

# The rf_model object is used as-is; no retraining.
cell_data[, rf_prediction := predict(rf_model, newdata = cell_data)]

cat("Done. Predictions added to cell_data.\n")
```

---

## Why This Is Correct and Fast

| Aspect | Old approach | New approach |
|---|---|---|
| **Neighbor resolution** | 6.46M string-paste + named-vector lookups | 1 vectorized `unlist` of the `nb` object (~1.37M rows) |
| **Per-variable work** | 6.46M R-level `lapply` iterations | 1 `data.table` equi-join + 1 grouped aggregation (~38.5M rows, vectorized C) |
| **Total R-level iterations** | ~32.3 million | 0 (all vectorized) |
| **Memory** | 6.46M-element list of integer vectors | Edge table: ~11 MB; merged table peaks at ~900 MB per variable (well within 16 GB) |
| **Numerical result** | max/min/mean of non-NA neighbor values | Identical — same neighbors, same values, same aggregation functions |
| **RF model** | Untouched | Untouched — `predict()` only |
| **Expected runtime** | 86+ hours | **5–15 minutes** on a 16 GB laptop |

The key principle: **build the spatial adjacency once as a flat edge table, then let `data.table`'s vectorized join and grouped aggregation handle the year dimension**, eliminating millions of R-level loop iterations entirely.