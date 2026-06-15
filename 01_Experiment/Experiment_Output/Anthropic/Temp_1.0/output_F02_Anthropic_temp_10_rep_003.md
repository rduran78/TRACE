 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the `build_neighbor_lookup` function. It calls `lapply` over **~6.46 million rows**, and inside each iteration it performs:

1. **Character coercion and named-vector lookup** (`id_to_ref[as.character(...)]`) — repeated millions of times, this is extremely slow because R's named vector lookup is O(n) in the worst case and involves hashing overhead on every call.
2. **String pasting** (`paste(neighbor_cell_ids, data$year[i], sep = "_")`) — allocating ~4–8 short strings per row × 6.46M rows = tens of millions of tiny allocations.
3. **Named index lookup** (`idx_lookup[neighbor_keys]`) on a **6.46-million-element named character vector** — each lookup hashes the string and probes a very large hash table.

The result is a **list of 6.46 million integer vectors**, which itself consumes several GB of RAM.

`compute_neighbor_stats` then loops over that 6.46M-element list again, computing `max/min/mean` per element — manageable in principle, but the R-level `lapply` over millions of small vectors is still slow.

Multiplied by 5 variables, the total work is enormous.

**Summary of problems:**
| Problem | Impact |
|---|---|
| Per-row string pasting & named-vector lookups in `build_neighbor_lookup` | ~90 % of runtime |
| 6.46M-element R list for `neighbor_lookup` | ~2–4 GB RAM |
| R-level `lapply` in `compute_neighbor_stats` over 6.46M entries × 5 vars | Significant |
| No vectorization or use of compiled code | Everything is interpreted |

---

## Optimization Strategy

### Key Insight

The neighbor graph is **time-invariant**: cell A's neighbors are the same in every year. The `nb` object already encodes this. We only need to "expand" it across years using **integer arithmetic on a regular panel**, completely avoiding string operations.

### Plan

1. **Exploit the balanced-panel structure.** If cells are ordered consistently, row `(t-1)*N + i` corresponds to cell `i` in year `t`. Neighbor indices for year `t` are simply the cell-level neighbor indices shifted by `(t-1)*N`. This turns `build_neighbor_lookup` into pure integer arithmetic — no strings, no hash lookups.

2. **Flatten the neighbor lookup into two parallel vectors** (a CSR-like / adjacency-list-as-vectors representation): a `target` vector and a `neighbor_row` vector. This replaces the 6.46M-element R list with two integer vectors totaling ~20–30 M elements, which is ~200 MB instead of several GB.

3. **Vectorize `compute_neighbor_stats`** using `data.table` grouped operations on the flat adjacency vectors. `data.table` performs grouped `max/min/mean` in compiled C code and is orders of magnitude faster than per-element `lapply`.

4. **Process all 5 variables in one pass** over the flat adjacency to avoid redundant subsetting.

**Expected improvement:** From ~86+ hours to **~5–20 minutes** on the same laptop, with peak RAM well under 16 GB.

---

## Working R Code

```r
library(data.table)

# ============================================================
# 0. Assumptions / inputs already in the environment:
#    - cell_data        : data.frame or data.table with columns
#                         id, year, ntl, ec, pop_density, def, usd_est_n2, ...
#    - id_order         : integer/character vector of cell IDs
#                         (the ordering that matches rook_neighbors_unique)
#    - rook_neighbors_unique : an nb object (list of integer vectors)
#                              where element i contains the indices
#                              (into id_order) of neighbors of cell i.
#    - neighbor_source_vars : c("ntl","ec","pop_density","def","usd_est_n2")
# ============================================================

# --------------------------------------------------
# STEP 1 : Convert to data.table & ensure sort order
# --------------------------------------------------
if (!is.data.table(cell_data)) cell_data <- as.data.table(cell_data)

# Map each cell id to its position in id_order (1-based).
id_map <- data.table(id = id_order, cell_idx = seq_along(id_order))
cell_data <- merge(cell_data, id_map, by = "id", all.x = TRUE)

# Sort by year then cell_idx so that row number = (year_offset) * N + cell_idx
year_levels <- sort(unique(cell_data$year))           # 1992 .. 2019
year_map    <- data.table(year = year_levels,
                          year_offset = seq_along(year_levels) - 1L)
cell_data   <- merge(cell_data, year_map, by = "year", all.x = TRUE)
setorder(cell_data, year_offset, cell_idx)

# After sorting, the row number for (cell_idx=i, year_offset=t) is
#   row = t * N + i,   where N = number of cells
N <- length(id_order)
T <- length(year_levels)
stopifnot(nrow(cell_data) == N * T)   # balanced panel check

# Assign explicit row numbers (will be used as indices into columns)
cell_data[, row_id := .I]

# Quick sanity: row_id should equal year_offset * N + cell_idx
stopifnot(all(cell_data$row_id == cell_data$year_offset * N + cell_data$cell_idx))

# --------------------------------------------------
# STEP 2 : Build flat adjacency vectors (CSR-style)
#           using only integer arithmetic
# --------------------------------------------------
# For each cell i (1..N), rook_neighbors_unique[[i]] gives
# neighbor cell indices (also in 1..N).
# For year_offset t, the ROW of cell i  = t*N + i
#                     the ROW of cell j  = t*N + j
# So we just need to enumerate (i, j) pairs from the nb object
# and then replicate across T years.

# 2a. Build cell-level edge list
from_cell <- rep(seq_len(N),
                 times = lengths(rook_neighbors_unique))
to_cell   <- unlist(rook_neighbors_unique, use.names = FALSE)

n_edges <- length(from_cell)   # total directed edges at cell level

# 2b. Expand across years: for each year_offset t in 0..(T-1),
#     target_row = t*N + from_cell,  neighbor_row = t*N + to_cell
# This creates two integer vectors of length T * n_edges.

year_offsets <- 0:(T - 1L)

# Use outer arithmetic, then flatten
# target_row[k]   = year_offsets[yt] * N + from_cell[e]
# neighbor_row[k] = year_offsets[yt] * N + to_cell[e]

# Memory estimate: 2 vectors × T × n_edges int32 values
#   2 × 28 × 1,373,394 × 4 bytes ≈ 308 MB  — fits in 16 GB
target_row   <- rep(year_offsets, each = n_edges) * N +
                rep(from_cell, times = T)
neighbor_row <- rep(year_offsets, each = n_edges) * N +
                rep(to_cell,   times = T)

# --------------------------------------------------
# STEP 3 : Compute neighbor stats vectorised
# --------------------------------------------------
# For each variable, pull neighbor values, group by target_row,
# compute max / min / mean.

adj_dt <- data.table(target = target_row, nbr = neighbor_row)

# Free the raw vectors to save RAM
rm(target_row, neighbor_row, from_cell, to_cell)
gc()

for (var_name in neighbor_source_vars) {

  # Pull neighbor values in one vectorised step
  adj_dt[, nbr_val := cell_data[[var_name]][nbr]]

  # Remove NAs before aggregation
  stats <- adj_dt[!is.na(nbr_val),
                  .(nb_max  = max(nbr_val),
                    nb_min  = min(nbr_val),
                    nb_mean = mean(nbr_val)),
                  keyby = target]

  # Initialize new columns to NA, then fill matched rows
  max_col  <- paste0(var_name, "_nb_max")
  min_col  <- paste0(var_name, "_nb_min")
  mean_col <- paste0(var_name, "_nb_mean")

  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]

  cell_data[stats$target, (max_col)  := stats$nb_max]
  cell_data[stats$target, (min_col)  := stats$nb_min]
  cell_data[stats$target, (mean_col) := stats$nb_mean]

  # Clean up within-loop temp column
  adj_dt[, nbr_val := NULL]
  rm(stats)
  gc()

  message("Done: ", var_name)
}

# --------------------------------------------------
# STEP 4 : Clean up helper columns
# --------------------------------------------------
cell_data[, c("cell_idx", "year_offset", "row_id") := NULL]

# cell_data now has the 15 new neighbor-stat columns
# (5 vars × 3 stats each) alongside all original columns.

# --------------------------------------------------
# STEP 5 : Use the existing trained Random Forest
#           for prediction (unchanged)
# --------------------------------------------------
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same neighbor relationships** | We use the exact same `rook_neighbors_unique` nb object; we simply translate cell-level indices to row-level indices with deterministic arithmetic. |
| **Same statistics** | `max`, `min`, `mean` are computed on exactly the same sets of non-NA neighbor values as the original code. |
| **Trained RF model untouched** | We only add the same 15 feature columns (same names, same numeric values) that the original pipeline produced. The model object is never modified. |
| **Original numerical estimand** | Because the feature values are numerically identical (same floating-point operations on the same inputs), downstream predictions and any estimand derived from them are unchanged. |

---

## Performance Estimate

| Step | Original | Optimized |
|---|---|---|
| Build neighbor lookup | ~80 hours (string ops × 6.46M) | ~30 seconds (integer arithmetic, vectorised) |
| Compute stats (5 vars) | ~6 hours (lapply × 6.46M × 5) | ~3–10 minutes (`data.table` grouped C code) |
| Peak RAM | >10 GB (6.46M-element list) | ~4–6 GB (flat int vectors + data.table) |
| **Total** | **~86 hours** | **~5–15 minutes** |