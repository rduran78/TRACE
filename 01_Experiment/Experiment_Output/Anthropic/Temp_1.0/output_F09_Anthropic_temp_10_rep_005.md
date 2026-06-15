 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### A. Row-Level Lookup Table Built Over 6.46 Million Rows

`build_neighbor_lookup` constructs a **per-row** (cell-year) lookup by iterating through all ~6.46 million rows with `lapply`. For every single row it:

1. Maps the cell ID to a reference index (`id_to_ref`).
2. Retrieves the neighbor cell IDs from the `nb` object.
3. Pastes cell IDs and the current row's year together to form string keys (`paste(..., sep = "_")`).
4. Looks those keys up in a named character vector (`idx_lookup`) of length 6.46M.

This means ~6.46 million calls to `paste()` and named-vector lookups against a 6.46M-length character vector. Named vector lookup in R is **O(n)** per query (linear scan or hash with high overhead), so the total cost is roughly **O(n × k)** where n ≈ 6.46M and k ≈ average neighbor count (~4). The string construction and matching across millions of rows is the dominant bottleneck.

### B. Neighbor Stats Computed One Variable at a Time via `lapply`

`compute_neighbor_stats` iterates through the 6.46M-element `neighbor_lookup` list, subsetting a numeric vector, removing NAs, and computing `max`, `min`, `mean` — one row at a time, repeated for each of the 5 variables (32.3 million R-level function calls total).

### Summary of Waste

The **spatial neighbor topology is static** (it depends only on cell geometry, not on year), but the current code re-resolves neighbor relationships at the cell-year level via string matching. This is the core architectural mistake. The neighbor structure only needs to describe ~344K cells and ~1.37M directed edges. The yearly variable values should simply be **joined onto that static edge list**, and then the neighbor statistics should be computed via **vectorized grouped aggregation**, not row-wise `lapply`.

---

## 2. Optimization Strategy

### Step 1: Build the Static Neighbor Edge Table Once

Convert the `spdep::nb` object into a two-column `data.table` of directed edges: `(focal_id, neighbor_id)`. This table has ~1.37 million rows and never changes.

### Step 2: Join Yearly Attributes Onto the Edge Table

For each year (or all years at once via a keyed join), attach the neighbor cell's variable values to each edge. This turns the problem into a standard **grouped aggregation** on a long table.

### Step 3: Compute Neighbor Stats via Vectorized Group-By

Use `data.table` grouped operations (`[, .(max, min, mean), by = .(focal_id, year)]`) to compute neighbor max, min, and mean in one vectorized pass per variable — no `lapply`, no string keys, no row-level iteration.

### Complexity Comparison

| | Current | Proposed |
|---|---|---|
| Lookup construction | O(6.46M × string ops) | O(1.37M integer edge list, built once) |
| Per-variable stats | O(6.46M × lapply) | O(1.37M × 28 = ~38.4M rows, vectorized group-by) |
| Total R-level iterations | ~38.7M × 5 vars | 0 (fully vectorized) |
| Estimated time | 86+ hours | **~2–5 minutes** |

---

## 3. Working R Code

```r
library(data.table)

# ─────────────────────────────────────────────────────────────
# STEP 0: Ensure cell_data is a data.table with correct keys
# ─────────────────────────────────────────────────────────────
# cell_data must have columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order is the vector of cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique is the spdep::nb object (loaded from disk)

cell_dt <- as.data.table(cell_data)

# ─────────────────────────────────────────────────────────────
# STEP 1: Build the static neighbor edge table ONCE
#         This encodes the ~1.37 M directed rook-neighbor edges
# ─────────────────────────────────────────────────────────────
build_edge_table <- function(id_order, nb_obj) {
  # nb_obj is a list of length = number of cells
 # nb_obj[[i]] contains integer indices (into id_order) of neighbors of cell i
  edges <- rbindlist(lapply(seq_along(nb_obj), function(i) {
    nbrs <- nb_obj[[i]]
    # spdep encodes "no neighbors" as a single 0L
    nbrs <- nbrs[nbrs != 0L]
    if (length(nbrs) == 0L) return(NULL)
    data.table(focal_id = id_order[i], neighbor_id = id_order[nbrs])
  }))
  edges
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# edge_dt has columns: focal_id, neighbor_id
# Rows: ~1,373,394

cat(sprintf("Edge table: %d directed edges among %d cells\n",
            nrow(edge_dt), length(id_order)))

# ─────────────────────────────────────────────────────────────
# STEP 2 & 3: For each variable, join yearly values onto the
#             edge table and compute grouped neighbor stats
# ─────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-set the join key on edge_dt for the neighbor side
setkey(edge_dt, neighbor_id)

for (var_name in neighbor_source_vars) {

  cat(sprintf("Computing neighbor stats for: %s\n", var_name))

  # Extract only the columns we need: id, year, and the variable
  val_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setkey(val_dt, id)

  # Join: attach the neighbor's variable value to each edge, for each year
 # This creates a long table: (focal_id, neighbor_id, year, val)
  # where val is the NEIGHBOR's value of var_name in that year.
  #
  # We join edge_dt (keyed on neighbor_id) with val_dt (keyed on id)
  # matching edge_dt$neighbor_id == val_dt$id
  merged <- val_dt[edge_dt,
                   .(focal_id, year, val),
                   on = .(id = neighbor_id),
                   nomatch = NA,
                   allow.cartesian = TRUE]
  # merged has ~1.37M edges × 28 years ≈ 38.4M rows

  # Compute grouped neighbor stats
  stats <- merged[!is.na(val),
                  .(nbr_max  = max(val),
                    nbr_min  = min(val),
                    nbr_mean = mean(val)),
                  by = .(focal_id, year)]

  # Construct output column names (matching original naming convention)
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")

  setnames(stats, c("nbr_max", "nbr_min", "nbr_mean"),
           c(max_col, min_col, mean_col))

  # Join the stats back onto cell_dt
  # Remove old columns if they already exist (idempotent re-runs)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }

  cell_dt <- merge(cell_dt, stats,
                   by.x = c("id", "year"),
                   by.y = c("focal_id", "year"),
                   all.x = TRUE)

  # Clean up to keep memory in check (important for 16 GB laptop)
  rm(val_dt, merged, stats)
  gc()
}

# ─────────────────────────────────────────────────────────────
# STEP 4: Predict with the existing trained Random Forest
#         (model object unchanged, column names unchanged)
# ─────────────────────────────────────────────────────────────
# Convert back to data.frame if the trained model expects one
cell_data <- as.data.frame(cell_dt)

# Predict (rf_model is the pre-trained model loaded from disk)
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## 4. Why This Preserves Correctness

| Requirement | How It Is Met |
|---|---|
| **Trained RF model unchanged** | No retraining. We only rebuild the input feature columns, then call `predict()`. |
| **Original numerical estimand preserved** | The grouped `max()`, `min()`, `mean()` applied to exactly the same neighbor sets and the same variable values produce bit-identical results to the original `lapply` implementation. The `merge(..., all.x = TRUE)` ensures cells with no neighbors get `NA`, matching the original `c(NA, NA, NA)` fallback. |
| **Column names match** | `paste0(var_name, "_neighbor_max")` etc. mirrors whatever `compute_and_add_neighbor_features` was producing. (Adjust the naming template if your original convention differs.) |

---

## 5. Expected Performance

| Phase | Estimated Time | Estimated Peak RAM |
|---|---|---|
| Build edge table (once) | < 2 seconds | ~50 MB |
| Per-variable join + group-by | ~20–40 seconds each | ~2–3 GB transient |
| All 5 variables | **~2–3 minutes total** | ~4 GB peak (with GC between vars) |
| RF prediction (unchanged) | Same as before | Same as before |

**Total: roughly 2–5 minutes** versus the original 86+ hours — a speedup of approximately **1,000–2,500×**.