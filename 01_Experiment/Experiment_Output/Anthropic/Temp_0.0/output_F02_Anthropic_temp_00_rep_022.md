 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is twofold:

### 1. `build_neighbor_lookup` — O(n) `lapply` over ~6.46 million rows
Each iteration performs character coercion, `paste`, and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is hash-based but still carries overhead per call. Doing this 6.46 million times with string allocation and matching is extremely expensive. The resulting list of ~6.46M integer vectors also consumes significant memory.

### 2. `compute_neighbor_stats` — repeated per variable, pure R loop
For each of the 5 neighbor source variables, another `lapply` over 6.46M rows subsets values, removes NAs, and computes `max/min/mean`. This is called 5 times sequentially. The `do.call(rbind, ...)` on a 6.46M-element list is itself a known performance anti-pattern.

**Combined effect:** billions of R-level function calls, millions of temporary string allocations, and repeated list-to-matrix conversions produce the estimated 86+ hour runtime.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| String-key lookups in `build_neighbor_lookup` | Replace with integer arithmetic: encode `(id, year)` as a single integer key and use `data.table` hash joins or direct index math. Since years are contiguous (1992–2019, 28 years), we can compute row indices arithmetically: `row = (cell_index - 1) * 28 + (year - 1991)` if data is sorted by `(id, year)`. This eliminates all `paste`/string operations. |
| `lapply` over 6.46M rows in R | Vectorize the neighbor lookup by expanding the neighbor list into a flat edge table `(row_i, row_j)` and using grouped vectorized operations via `data.table`. |
| Per-variable `lapply` over 6.46M rows | Compute all 5 variables' neighbor stats in a single grouped `data.table` aggregation pass over the edge table. |
| `do.call(rbind, list_of_6.46M)` | Eliminated entirely; `data.table` returns a single result table. |
| Memory (16 GB) | The flat edge table has ~6.46M × avg_neighbors ≈ ~25–30M rows × 2 integer columns ≈ ~0.5 GB. The main data (~6.46M × 110 cols) is ~5–7 GB. Feasible within 16 GB if we avoid duplication. |

**Expected speedup:** from 86+ hours to roughly 5–20 minutes.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 0.  Convert to data.table (in-place if possible to save memory)
# ──────────────────────────────────────────────────────────────────────
setDT(cell_data)

# Ensure data is keyed/sorted by (id, year) — critical for index math
setorder(cell_data, id, year)

# Add a row index explicitly
cell_data[, row_idx := .I]

# ──────────────────────────────────────────────────────────────────────
# 1.  Build integer mappings (replaces build_neighbor_lookup entirely)
# ──────────────────────────────────────────────────────────────────────

# id_order is the vector of unique cell IDs matching rook_neighbors_unique
# Map each cell id -> its position in id_order (1-based "ref index")
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

# Map each (id, year) -> row_idx using data.table keyed join
#   Because data is sorted by (id, year) and years are contiguous 1992-2019,
#   we can use direct arithmetic.  But to be safe against missing cell-years
#   we build a small keyed lookup table.
id_year_to_row <- cell_data[, .(id, year, row_idx)]
setkey(id_year_to_row, id, year)

# ──────────────────────────────────────────────────────────────────────
# 2.  Expand neighbor list into a flat edge table  (row_i  →  row_j)
#     This is the key transformation: we do it ONCE, not per variable.
# ──────────────────────────────────────────────────────────────────────

# 2a. Build cell-level edge list from rook_neighbors_unique (spdep nb object)
#     Each element k of the nb list gives the ref-indices of neighbors of cell k.
message("Building cell-level edge list …")
n_cells <- length(id_order)
from_ref <- rep(seq_len(n_cells),
                times = lengths(rook_neighbors_unique))
to_ref   <- unlist(rook_neighbors_unique, use.names = FALSE)

# Convert ref indices → actual cell IDs
cell_edges <- data.table(
  from_id = id_order[from_ref],
  to_id   = id_order[to_ref]
)
rm(from_ref, to_ref)

# 2b. Expand to row-level edges by joining on every year.
#     For each (from_id, year) we need (to_id, year) — same year.
#     Strategy: cross-join cell_edges with the 28 years, then map to row_idx.
message("Expanding to row-level edge table …")

years <- sort(unique(cell_data$year))  # 1992:2019

# Expand: each cell edge × each year
edge_expanded <- cell_edges[, .(year = years), by = .(from_id, to_id)]
rm(cell_edges)

# Join to get row_idx for the "from" side
setkey(edge_expanded, from_id, year)
edge_expanded[id_year_to_row, row_i := i.row_idx, on = .(from_id = id, year)]

# Join to get row_idx for the "to" (neighbor) side
setkey(edge_expanded, to_id, year)
edge_expanded[id_year_to_row, row_j := i.row_idx, on = .(to_id = id, year)]

# Drop edges where either side is missing (cell-year not in data)
edge_expanded <- edge_expanded[!is.na(row_i) & !is.na(row_j),
                               .(row_i, row_j)]
setkey(edge_expanded, row_i)

message(sprintf("Edge table: %s rows", format(nrow(edge_expanded), big.mark = ",")))

rm(id_year_to_row)
gc()

# ──────────────────────────────────────────────────────────────────────
# 3.  Compute ALL neighbor stats in one vectorized pass
# ──────────────────────────────────────────────────────────────────────

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

message("Attaching neighbor values to edge table …")

# Pull only the columns we need from cell_data (saves memory in the join)
val_cols <- c("row_idx", neighbor_source_vars)
vals_dt  <- cell_data[, ..val_cols]
setkey(vals_dt, row_idx)

# Join neighbor values onto the edge table (row_j is the neighbor)
edge_with_vals <- vals_dt[edge_expanded, on = .(row_idx = row_j), nomatch = NA]
# edge_with_vals now has columns: row_idx (=row_j), <vars>, row_i
# Rename for clarity
setnames(edge_with_vals, "row_idx", "row_j")

rm(vals_dt, edge_expanded)
gc()

message("Computing grouped neighbor statistics …")

# Build aggregation expressions dynamically
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
    bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
    bquote(mean(.(as.name(v)), na.rm = TRUE))
  )
}), recursive = FALSE)

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

names(agg_exprs) <- agg_names

# Single grouped aggregation
neighbor_stats <- edge_with_vals[,
  eval(as.call(c(as.name("list"), agg_exprs))),
  by = row_i
]

rm(edge_with_vals)
gc()

# Replace Inf/-Inf (from max/min on all-NA groups) with NA
inf_cols <- setdiff(names(neighbor_stats), "row_i")
for (col in inf_cols) {
  set(neighbor_stats, which(is.infinite(neighbor_stats[[col]])), col, NA_real_)
}

# ──────────────────────────────────────────────────────────────────────
# 4.  Merge back into cell_data
# ──────────────────────────────────────────────────────────────────────

message("Merging neighbor features back into cell_data …")

# Remove any pre-existing neighbor columns to avoid duplication
old_cols <- intersect(names(cell_data), agg_names)
if (length(old_cols)) cell_data[, (old_cols) := NULL]

setkey(neighbor_stats, row_i)
setkey(cell_data, row_idx)

cell_data[neighbor_stats, (agg_names) := mget(paste0("i.", agg_names)),
          on = .(row_idx = row_i)]

# Rows with no neighbors will already be NA (unmatched in join)

# Clean up helper column
cell_data[, row_idx := NULL]

rm(neighbor_stats)
gc()

message("Done — neighbor features added.")
```

---

## Why This Preserves Correctness

| Requirement | How it is met |
|---|---|
| **Trained Random Forest model unchanged** | We only add/replace the same neighbor feature columns (`neighbor_max_*`, `neighbor_min_*`, `neighbor_mean_*`) that the original code produced. No model object is touched. |
| **Original numerical estimand preserved** | `max`, `min`, `mean` with `na.rm = TRUE` on the same neighbor sets produce identical values. The edge table is derived from the same `rook_neighbors_unique` nb object and the same year-matching logic. |
| **Same column names** | The naming convention `neighbor_{stat}_{var}` matches what `compute_and_add_neighbor_features` would have produced (adjust the prefix/suffix pattern if your wrapper used a different convention). |

---

## Performance Estimate

| Stage | Estimated time | Estimated peak memory |
|---|---|---|
| Edge table construction | ~1–2 min | ~2 GB (edge table ~30M rows × 4 int cols) |
| Join neighbor values | ~1–2 min | ~4 GB (edge table + 5 double cols) |
| Grouped aggregation | ~2–5 min | ~3 GB (result: 6.46M × 16 cols) |
| Merge back | ~30 sec | negligible additional |
| **Total** | **~5–10 min** | **~12–14 GB peak (fits in 16 GB)** |

If memory is still tight, the grouped aggregation can be chunked by variable (5 passes instead of 1), trading a small amount of speed for lower peak memory. But the single-pass approach should fit within 16 GB.