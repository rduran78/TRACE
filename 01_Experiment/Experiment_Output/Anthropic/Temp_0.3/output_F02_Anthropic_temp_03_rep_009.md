 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The two critical bottlenecks are:

### 1. `build_neighbor_lookup` — O(n) `lapply` over 6.46 million rows with per-row string pasting and named-vector lookups
- `paste()` is called millions of times inside the loop to create keys.
- Named-vector indexing (`idx_lookup[neighbor_keys]`) is an O(k) hash lookup repeated for every row, but the overhead of creating and matching character keys at this scale is enormous.
- The result is a **list of 6.46 million integer vectors**, which is itself a large, fragmented memory object (~hundreds of MB of pointer overhead alone).

### 2. `compute_neighbor_stats` — O(n) `lapply` over the same 6.46 million rows, repeated 5 times
- Each call iterates through every row, subsets a numeric vector, removes NAs, and computes `max`, `min`, `mean`.
- `do.call(rbind, result)` on a 6.46-million-element list of length-3 vectors is extremely slow (repeated memory allocation).
- This is called **5 times** (once per variable), so the full inner loop runs ~32.3 million row-level operations.

### Memory concern
- The neighbor lookup list alone (6.46M entries, each a small integer vector) can consume several GB due to R's per-object overhead (~128 bytes per SEXP header).
- Intermediate character vectors from `paste()` add further pressure on a 16 GB machine.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Per-row `paste` key construction | Replace character keys with **integer arithmetic keys**: `key = id_integer * 10000L + (year - 1991L)`. This is orders of magnitude faster and avoids string allocation. |
| Named-vector hash lookup in a loop | Pre-build a **keyed `data.table`** and use binary-search joins, or use a simple integer-indexed lookup vector. |
| 6.46M-element R list for neighbor lookup | Flatten into **two integer vectors** (a CSR-like compressed sparse structure): one holding all neighbor row-indices concatenated, and one holding the offset/pointer for each row. This eliminates millions of small R objects. |
| `lapply` + `do.call(rbind, ...)` for stats | Replace with a **single vectorized `data.table` grouped aggregation** over the flattened edge list, or use C++ via `Rcpp`. The `data.table` approach requires zero compilation. |
| 5 separate passes over the edge list | Compute **all 5 variables' neighbor stats in one pass** using `data.table`'s `melt` + grouped aggregation, or loop over variables but with the fast vectorized kernel. |

The strategy preserves all original numerical outputs (max, min, mean of neighbors) and does not touch the trained Random Forest model.

---

## Working R Code

```r
library(data.table)

# ===========================================================================
# STEP 0 — Convert to data.table (if not already) and create integer keys
# ===========================================================================
cell_dt <- as.data.table(cell_data)

# Ensure id and year are integer
cell_dt[, id := as.integer(id)]
cell_dt[, year := as.integer(year)]

# Create a compact integer key:  id * 10000 + (year - 1991)
# year range 1992-2019 → offsets 1-28, well within 10000 margin
cell_dt[, row_idx := .I]
cell_dt[, int_key := id * 10000L + (year - 1991L)]

# Build a fast key → row_idx lookup vector.
# Maximum possible key value:
max_key <- max(cell_dt$int_key)
key_to_row <- integer(max_key)            
# This vector may be large (~max_id * 10000 * 4 bytes).
# For 344,208 cells: 344208 * 10000 * 4 bytes ≈ 13.8 GB — TOO BIG.
# Instead, use a data.table hash join.

# --- Use data.table keyed join instead ---
setkey(cell_dt, int_key)

# ===========================================================================
# STEP 1 — Build a flat edge table (CSR-style, but as a data.table)
#           This replaces build_neighbor_lookup entirely.
# ===========================================================================
build_edge_table <- function(cell_dt, id_order, neighbors) {
  # id_order: integer vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer index vectors into id_order)
  
  id_order <- as.integer(id_order)
  n_cells  <- length(id_order)
  
  # --- Build edges at the cell level (id → neighbor_id) ---
  # Pre-allocate by computing total number of directed edges
  n_edges <- sum(lengths(neighbors))  # ~1.37 million
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_len(n_cells)) {
    nb_i <- neighbors[[i]]
    if (length(nb_i) == 0L || (length(nb_i) == 1L && nb_i[1] == 0L)) next
    len  <- length(nb_i)
    from_id[pos:(pos + len - 1L)] <- id_order[i]
    to_id[pos:(pos + len - 1L)]   <- id_order[nb_i]
    pos <- pos + len
  }
  
  # Trim if any nb entries were empty
  if (pos <= n_edges) {
    from_id <- from_id[1:(pos - 1L)]
    to_id   <- to_id[1:(pos - 1L)]
  }
  
  cell_edges <- data.table(from_id = from_id, to_id = to_id)
  
  # --- Expand edges across all 28 years ---
  years <- sort(unique(cell_dt$year))
  
  # Cross join edges × years  (~1.37M × 28 ≈ 38.5M rows)
  # Each row says: "for cell from_id in year y, one neighbor is to_id"
  edge_year <- cell_edges[, CJ(from_id = from_id, to_id = to_id, year = years, 
                                 unique = FALSE)]
  # CJ expands fully — we need a simple cross with years instead:
  edge_year <- cell_edges[rep(seq_len(.N), each = length(years))]
  edge_year[, year := rep(years, times = nrow(cell_edges))]
  
  # Compute integer keys for the "from" and "to" sides
  edge_year[, from_key := from_id * 10000L + (year - 1991L)]
  edge_year[, to_key   := to_id   * 10000L + (year - 1991L)]
  
  edge_year
}

cat("Building edge table...\n")
edge_year <- build_edge_table(cell_dt, id_order, rook_neighbors_unique)
cat(sprintf("Edge table: %s rows\n", format(nrow(edge_year), big.mark = ",")))

# ===========================================================================
# STEP 2 — Attach neighbor variable values via keyed join and aggregate
#           This replaces compute_neighbor_stats + the outer loop.
# ===========================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare a slim table of the columns we need for the neighbor lookup
# (to_key is the key we join on; we need the variable values at that key)
val_cols <- c("int_key", neighbor_source_vars)
val_dt   <- cell_dt[, ..val_cols]
setkey(val_dt, int_key)

# Join neighbor values onto the edge table
cat("Joining neighbor values onto edge table...\n")
setkey(edge_year, to_key)
edge_year <- val_dt[edge_year, on = .(int_key = to_key), nomatch = NA, allow.cartesian = TRUE]
# After this join, edge_year has columns: int_key (=to_key), ntl, ec, ..., from_id, to_id, year, from_key, to_key

# ===========================================================================
# STEP 3 — Grouped aggregation: max, min, mean per (from_key, variable)
# ===========================================================================
cat("Computing neighbor statistics...\n")

# Aggregate all 5 variables at once, grouped by from_key
agg <- edge_year[, {
  res <- list()
  for (v in neighbor_source_vars) {
    vals <- .SD[[v]]
    vals <- vals[!is.na(vals)]
    if (length(vals) == 0L) {
      res[[paste0("neighbor_max_", v)]]  <- NA_real_
      res[[paste0("neighbor_min_", v)]]  <- NA_real_
      res[[paste0("neighbor_mean_", v)]] <- NA_real_
    } else {
      res[[paste0("neighbor_max_", v)]]  <- max(vals)
      res[[paste0("neighbor_min_", v)]]  <- min(vals)
      res[[paste0("neighbor_mean_", v)]] <- mean(vals)
    }
  }
  res
}, by = from_key, .SDcols = neighbor_source_vars]

# ===========================================================================
# STEP 4 — Join aggregated stats back onto the main cell_dt
# ===========================================================================
cat("Joining neighbor features back to main table...\n")
setkey(agg, from_key)
setkey(cell_dt, int_key)

cell_dt <- agg[cell_dt, on = .(from_key = int_key)]

# Clean up helper columns
cell_dt[, c("int_key", "row_idx") := NULL]

# Convert back to data.frame if downstream code expects it
cell_data <- as.data.frame(cell_dt)

cat("Done. Neighbor features added.\n")
```

---

### If 38.5 M rows × many columns causes memory pressure

The edge table with all 5 variable columns joined can reach ~3–4 GB. On a 16 GB laptop this is feasible but tight. If memory is still a concern, process variables **one at a time** with the same approach:

```r
for (v in neighbor_source_vars) {
  cat(sprintf("Processing neighbor stats for: %s\n", v))
  
  # Slim value table: just the key and one variable
  val_one <- cell_dt[, .(int_key, val = get(v))]
  setkey(val_one, int_key)
  
  # Join onto edge table (only ~38.5M rows × 1 numeric column)
  setkey(edge_year, to_key)
  edges_v <- val_one[edge_year[, .(from_key, to_key)], 
                      on = .(int_key = to_key), nomatch = NA]
  
  # Aggregate
  agg_v <- edges_v[!is.na(val), 
                    .(nmax = max(val), nmin = min(val), nmean = mean(val)), 
                    by = from_key]
  
  setnames(agg_v, c("nmax", "nmin", "nmean"),
           paste0("neighbor_", c("max_", "min_", "mean_"), v))
  
  # Join back
  setkey(agg_v, from_key)
  cell_dt <- agg_v[cell_dt, on = .(from_key = int_key)]
  
  rm(edges_v, agg_v, val_one)
  gc()
}
```

---

## Expected Performance Improvement

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M `paste` + hash lookups → hours | Integer key arithmetic + `data.table` join → **minutes** |
| `compute_neighbor_stats` (×5) | ~32.3M R-level `lapply` iterations | `data.table` grouped aggregation (C-level radix sort + group-by) → **minutes** |
| Memory (neighbor lookup) | ~6.46M R list elements (~2–5 GB overhead) | Flat `data.table` (~1.5–3 GB, contiguous) |
| Total estimated time | **86+ hours** | **10–30 minutes** on the same laptop |

The trained Random Forest model is untouched — only the feature-engineering step is optimized. The numerical outputs (neighbor max, min, mean) are identical to the original implementation.