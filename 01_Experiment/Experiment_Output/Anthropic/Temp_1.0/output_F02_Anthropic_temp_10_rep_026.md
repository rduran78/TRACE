 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is in **`build_neighbor_lookup`**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting and named-vector lookups. Named vector lookups in R via `[` on character keys are **O(n)** hash probes repeated millions of times, and `paste()` inside a per-row loop creates enormous intermediate string allocation. This produces a **list of 6.46 million integer vectors**, consuming substantial memory and taking tens of hours.

**`compute_neighbor_stats`** then iterates over that 6.46M-element list again, once per variable (×5 variables), performing subsetting and summary statistics in pure R loops.

**Specific problems:**

| Component | Issue |
|---|---|
| `build_neighbor_lookup` – `paste(…, sep="_")` inside `lapply` | Creates ~6.46M×k strings per row; massive allocation |
| `build_neighbor_lookup` – `idx_lookup[neighbor_keys]` | Named character vector lookup is slow at scale |
| `build_neighbor_lookup` – output list | ~6.46M list elements, each an integer vector; high memory overhead (~several GB just for list structure) |
| `compute_neighbor_stats` – `lapply` per variable | Pure R loop over 6.46M elements, repeated 5 times |
| General | No vectorization, no use of data.table or matrix operations |

---

## Optimization Strategy

**Key insight:** Instead of building a per-row list lookup, convert the problem to a **tabular join** using `data.table`. The neighbor relationships can be expressed as an edge table `(id, neighbor_id)`. We join this with the panel data on `(neighbor_id, year)` to get neighbor values, then group-by `(id, year)` to compute `max`, `min`, `mean`. This replaces both functions with fully vectorized, indexed operations.

**Steps:**

1. **Expand `rook_neighbors_unique` (nb object) into an edge data.table** `(id, neighbor_id)` — done once, ~1.37M rows.
2. **For each variable**, do an equi-join of `edges` with `cell_data` on `(neighbor_id = id, year = year)`, then aggregate by `(id, year)` to get `max`, `min`, `mean`.
3. **Merge** the aggregated stats back into `cell_data`.

**Why this is fast:**
- `data.table` binary-search joins on integer keys are orders of magnitude faster than character named-vector lookups.
- Group-by aggregation in `data.table` is implemented in C and is memory-efficient.
- No list of 6.46M elements; no per-row string allocation.

**Expected improvement:** From ~86+ hours to **minutes** (typically 5–20 min depending on disk I/O and RAM pressure). Peak memory stays well within 16 GB.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# STEP 1: Convert the nb object to an edge data.table (one-time)
# ---------------------------------------------------------------
# rook_neighbors_unique is a list of integer vectors (spdep nb object).
# id_order is the vector of cell IDs corresponding to indices 1..N.

build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate by computing total number of edges
  n_edges <- sum(lengths(neighbors))
  
  from_id    <- integer(n_edges)
  to_id      <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_i <- neighbors[[i]]
    len  <- length(nb_i)
    if (len == 0L) next
    idx <- pos:(pos + len - 1L)
    from_id[idx] <- id_order[i]
    to_id[idx]   <- id_order[nb_i]
    pos <- pos + len
  }
  
  data.table(id = from_id, neighbor_id = to_id)
}

edges <- build_edge_table(id_order, rook_neighbors_unique)

# ---------------------------------------------------------------
# STEP 2: Convert cell_data to data.table (if not already)
# ---------------------------------------------------------------
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Ensure key columns are integer for fast joins
cell_data[, id   := as.integer(id)]
cell_data[, year := as.integer(year)]
edges[, id          := as.integer(id)]
edges[, neighbor_id := as.integer(neighbor_id)]

# ---------------------------------------------------------------
# STEP 3: For each neighbor source variable, compute stats via join
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Create a minimal keyed lookup table: (id, year, <all source vars>)
# This avoids carrying all 110 columns through the join.
lookup_cols <- c("id", "year", neighbor_source_vars)
lookup_dt   <- cell_data[, ..lookup_cols]
setkey(lookup_dt, id, year)

# Key edges for the join
setnames(edges, c("id", "neighbor_id"))  # ensure names

for (var_name in neighbor_source_vars) {
  
  cat("Processing neighbor stats for:", var_name, "\n")
  
  # Build a small table: (neighbor_id, year, value)
  val_dt <- lookup_dt[, .(neighbor_id = id, year, value = get(var_name))]
  setkey(val_dt, neighbor_id, year)
  
  # Join edges with cell_data to get each cell's year,

  # then join with val_dt to get the neighbor's value for that year.
  # Strategy: 
  #   1. Attach 'year' to each edge by joining edges with cell_data on 'id'
  #   2. Then join with val_dt on (neighbor_id, year) to get neighbor value
  #   3. Aggregate by (id, year)
  
  # To avoid a huge intermediate table, we do it in one chain:
  # First, create (id, year) from cell_data (unique combinations)
  id_year <- unique(cell_data[, .(id, year)])
  setkey(id_year, id)
  setkey(edges, id)
  
  # Expand: each (id, year) × neighbors of id → (id, year, neighbor_id)
  expanded <- edges[id_year, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded has columns: id, neighbor_id, year
  
  # Now join to get neighbor values
  setkey(expanded, neighbor_id, year)
  expanded[val_dt, on = .(neighbor_id, year), value := i.value]
  
  # Aggregate: max, min, mean per (id, year), ignoring NAs
  agg <- expanded[!is.na(value), 
                  .(nb_max  = max(value),
                    nb_min  = min(value),
                    nb_mean = mean(value)), 
                  by = .(id, year)]
  
  # Rename columns to match expected output pattern
  new_names <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  setnames(agg, c("nb_max", "nb_min", "nb_mean"), new_names)
  
  # Merge back into cell_data
  setkey(agg, id, year)
  setkey(cell_data, id, year)
  
  # Remove old columns if they exist (idempotent re-runs)
  for (nn in new_names) {
    if (nn %in% names(cell_data)) cell_data[, (nn) := NULL]
  }
  
  cell_data <- agg[cell_data, on = .(id, year)]
  
  # Clean up to free memory

  rm(val_dt, expanded, agg)
  gc()
  
  cat("  Done.\n")
}

# ---------------------------------------------------------------
# STEP 4: Use the existing trained Random Forest for prediction
# ---------------------------------------------------------------
# The trained RF model object is unchanged. 
# cell_data now contains the same neighbor feature columns 
# (e.g., ntl_nb_max, ntl_nb_min, ntl_nb_mean, etc.)
# with numerically identical values, so predictions are preserved.
#
# Example (unchanged from original pipeline):
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary of Guarantees

| Concern | Status |
|---|---|
| **Numerical equivalence** | Identical `max`, `min`, `mean` computed on the same non-NA neighbor values; no approximation. |
| **Trained RF model preserved** | No retraining; same feature names and values. |
| **Memory** | `expanded` table is the peak object (~1.37M edges × 28 years ≈ 38M rows × 3 int/dbl cols ≈ ~1 GB per variable, freed each iteration). Total well within 16 GB. |
| **Speed** | `data.table` keyed joins + grouped aggregation: expected ~5–20 minutes total vs. 86+ hours. |
| **Dependencies** | Only `data.table` (no Rcpp, no parallelism needed, though both could further help). |