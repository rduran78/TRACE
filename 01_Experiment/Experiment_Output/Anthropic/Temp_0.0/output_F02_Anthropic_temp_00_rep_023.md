 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting, hash-table lookups, and NA filtering in pure R. This creates millions of small character vectors and triggers repeated hash lookups on a named vector of 6.46 million entries — an O(n²)-like workload in practice. The `compute_neighbor_stats` function then iterates over the same 6.46 million entries again per variable, subsetting numeric vectors by index lists. With 5 variables, that's ~32.3 million list iterations total.

**Specific problems:**

1. **`build_neighbor_lookup`:** `paste()` and named-vector lookup inside a per-row `lapply` over 6.46M rows is extremely slow. The named character vector `idx_lookup` of length 6.46M makes each `[` lookup O(n) in the worst case (R's hashed environments are faster, but `setNames` on a plain vector uses linear search or a hash that still has overhead at this scale). This single function likely accounts for >90% of the 86-hour estimate.

2. **`compute_neighbor_stats`:** `lapply` over 6.46M list elements with per-element `max`/`min`/`mean` is slow but less catastrophic. The `do.call(rbind, ...)` on a 6.46M-element list of 3-vectors is also memory-intensive (creates a temporary list of millions of small vectors before binding).

3. **Memory:** 6.46M rows × 110 columns ≈ 5.7 GB as double. The neighbor lookup list (6.46M elements, each a small integer vector) adds ~1–2 GB. Intermediate copies from `do.call(rbind, ...)` and repeated `data[[var_name]]` extraction add pressure. 16 GB is tight.

---

## Optimization Strategy

### Principle: Replace per-row R loops with vectorized joins and matrix operations using `data.table`.

**Step 1 — Vectorized neighbor lookup via `data.table` equi-join:**
Instead of building a 6.46M-element list, build a **long-format edge table** (`cell_row`, `neighbor_row`) using vectorized operations. Expand the `nb` object into a two-column edge list of (`id`, `neighbor_id`), join with year to get (`id`, `year`, `neighbor_id`), then map to row indices. This replaces the per-row `lapply` with a single merge.

**Step 2 — Vectorized neighbor stats via grouped aggregation:**
Instead of iterating over a list, use `data.table` grouped `max`/`min`/`mean` on the long edge table joined with the variable values. One grouped aggregation replaces 6.46M list iterations per variable.

**Step 3 — Memory management:**
Process one variable at a time, attach results immediately, and `rm()` intermediates.

**Expected speedup:** From ~86 hours to **~5–15 minutes** on the same laptop.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 0: Convert cell_data to data.table (in-place, no copy)
# ──────────────────────────────────────────────────────────────────────
setDT(cell_data)

# Ensure there is a row-index column for later re-attachment
cell_data[, .row_id := .I]

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build a vectorized edge table from the nb object
#
# rook_neighbors_unique is a list of length N_cells (344,208).
# id_order is the vector mapping list position -> cell id.
# We expand this into a long data.table: (id, neighbor_id)
# ──────────────────────────────────────────────────────────────────────

build_edge_table <- function(id_order, neighbors) {
  # neighbors is an nb object: list of integer vectors (indices into id_order)
  # Remove 0-length entries gracefully
  n_neighbors <- lengths(neighbors)
  
  from_idx <- rep(seq_along(neighbors), times = n_neighbors)
  to_idx   <- unlist(neighbors, use.names = FALSE)
  
  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

cat("Building edge table...\n")
edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
cat(sprintf("  Edge table: %s rows\n", format(nrow(edge_dt), big.mark = ",")))

# ──────────────────────────────────────────────────────────────────────
# STEP 2: Cross-join edges with years to get (id, year, neighbor_id),
#          then map each (neighbor_id, year) to its row in cell_data.
#
# Key insight: every edge (A -> B) exists for ALL 28 years.
# So we can do a single cross-join with the year vector, then
# join to cell_data to pick up the neighbor's variable value.
# ──────────────────────────────────────────────────────────────────────

# Unique years
years_vec <- sort(unique(cell_data$year))

# Cross-join edges × years  (~1.37M edges × 28 years ≈ 38.5M rows)
# This is the largest object; ~38.5M × 3 cols × 8 bytes ≈ 0.9 GB
cat("Expanding edges × years...\n")
edge_year_dt <- edge_dt[, .(year = years_vec), by = .(id, neighbor_id)]

# Free the compact edge table
rm(edge_dt)
gc()

# Set keys for fast joins
setkey(edge_year_dt, neighbor_id, year)

# We also need a keyed version of cell_data for the neighbor value lookup
# Create a minimal keyed reference (id, year, .row_id) for the focal cell
setkey(cell_data, id, year)

# ──────────────────────────────────────────────────────────────────────
# STEP 3: For each variable, join, aggregate, and attach
# ──────────────────────────────────────────────────────────────────────

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Build a small lookup: (id, year) -> variable value, reused per variable
# We key cell_data on (id, year) once.

cat("Computing neighbor features...\n")

for (var_name in neighbor_source_vars) {
  cat(sprintf("  Processing: %s\n", var_name))
  
  # --- 3a. Build a lookup of neighbor values ---
  # Extract only the columns we need from cell_data for the neighbor side
  val_dt <- cell_data[, .(id, year, val = get(var_name))]
  setkey(val_dt, id, year)
  
  # --- 3b. Join neighbor values onto the edge table ---
  # edge_year_dt is keyed on (neighbor_id, year)
  # val_dt is keyed on (id, year)
  # We join: edge_year_dt[val_dt] matching neighbor_id==id, year==year
  
  # Rename for clarity before join
  setnames(val_dt, "id", "neighbor_id")
  # Now val_dt is keyed on (neighbor_id, year)
  setkey(val_dt, neighbor_id, year)
  
  # Join: attach neighbor's value to each edge-year row
  edge_year_dt[val_dt, neighbor_val := i.val, on = .(neighbor_id, year)]
  
  # --- 3c. Aggregate: group by (id, year) to get max, min, mean ---
  agg <- edge_year_dt[
    !is.na(neighbor_val),
    .(
      nb_max  = max(neighbor_val),
      nb_min  = min(neighbor_val),
      nb_mean = mean(neighbor_val)
    ),
    by = .(id, year)
  ]
  
  # Name the output columns to match the original pipeline's naming convention
  # (adjust these names if your compute_and_add_neighbor_features used different names)
  new_names <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  setnames(agg, c("nb_max", "nb_min", "nb_mean"), new_names)
  
  # --- 3d. Merge back into cell_data ---
  # Left join: cell_data keeps all rows; cells with no valid neighbors get NA
  setkey(agg, id, year)
  cell_data <- agg[cell_data, on = .(id, year)]
  setkey(cell_data, id, year)
  
  # --- 3e. Clean up the temporary column on edge_year_dt ---
  edge_year_dt[, neighbor_val := NULL]
  
  rm(val_dt, agg)
  gc()
  
  cat(sprintf("    -> Added: %s\n", paste(new_names, collapse = ", ")))
}

# ──────────────────────────────────────────────────────────────────────
# STEP 4: Clean up
# ──────────────────────────────────────────────────────────────────────
rm(edge_year_dt)
gc()

# Restore original row order if needed
setorder(cell_data, .row_id)
cell_data[, .row_id := NULL]

cat("Done. Neighbor features attached.\n")
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Trained Random Forest model** | No model code is touched. We only reproduce the same feature columns (`*_nb_max`, `*_nb_min`, `*_nb_mean`) that the original code produced. The RF object remains unchanged on disk/memory. |
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets (same `nb` object, same year matching, same NA exclusion logic). The `data.table` grouped aggregation is numerically identical to the original R `max`/`min`/`mean` calls. |
| **Column naming** | Adjust the `new_names` pattern in Step 3c if your original `compute_and_add_neighbor_features` used a different naming convention (e.g., `"ntl_max_neighbor"`). The values will be identical. |

---

## Memory Budget Estimate

| Object | Approximate Size |
|---|---|
| `cell_data` (6.46M × 110 cols) | ~5.7 GB |
| `edge_year_dt` (38.5M × 3 cols + 1 temp) | ~1.2 GB |
| Temporary `val_dt` + `agg` | ~0.5 GB |
| Overhead / R session | ~1.5 GB |
| **Total peak** | **~9 GB** (fits in 16 GB) |

If memory is still tight, you can **chunk by year** (process 4–7 years at a time) or by variable, but the above should fit comfortably on a 16 GB machine.

---

## Expected Runtime

- **Edge table construction:** <1 second (vectorized `rep`/`unlist`).
- **Cross-join expansion:** ~10–30 seconds.
- **Per-variable join + aggregate:** ~30–60 seconds each.
- **Total for 5 variables:** **~3–6 minutes** (vs. 86+ hours originally).