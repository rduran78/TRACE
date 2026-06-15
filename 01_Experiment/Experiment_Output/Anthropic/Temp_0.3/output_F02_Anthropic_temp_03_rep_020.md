 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting, hash-table lookups, and NA filtering in pure R. This creates millions of small character vectors and triggers repeated hash lookups on a named vector of 6.46 million entries — an O(n²)-like workload in practice. The `compute_neighbor_stats` function then iterates over the same 6.46 million entries again per variable, subsetting numeric vectors by index lists. With 5 variables, that's ~32.3 million list iterations total.

**Specific problems:**

1. **`build_neighbor_lookup`:** `paste()` and named-vector lookup inside a per-row `lapply` over 6.46M rows is extremely slow. Each call to `idx_lookup[neighbor_keys]` does a linear scan or hash probe on a 6.46M-element named character vector. The resulting list of 6.46M integer vectors also consumes substantial memory.
2. **`compute_neighbor_stats`:** `lapply` over 6.46M elements with per-element subsetting, NA removal, and summary computation is slow in interpreted R. The `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors is also a known performance anti-pattern.
3. **Memory:** The neighbor lookup list (6.46M elements, each a small integer vector) plus the data frame with 110+ columns at 6.46M rows pushes close to or beyond 16 GB, especially with intermediate copies.

---

## Optimization Strategy

### Principle: Replace per-row R loops with vectorized joins and grouped aggregations using `data.table`.

1. **Replace `build_neighbor_lookup` entirely.** Instead of building a 6.46M-element list, build a flat `data.table` edge list of `(row_i, neighbor_row_j)` pairs. This is constructed via a vectorized merge — no per-row `lapply` needed.

2. **Replace `compute_neighbor_stats` with a grouped `data.table` aggregation.** Join the edge table to the variable values, then compute `max`, `min`, `mean` per group in one vectorized pass.

3. **Memory management:** The flat edge table will have ~1.37M neighbor pairs × 28 years ≈ ~38.5M rows of two integer columns (~308 MB), which is far more cache-friendly and memory-predictable than 6.46M ragged lists. We process one variable at a time and free intermediates.

4. **The trained Random Forest model and all numerical outputs are preserved** — we only change how features are computed, not what is computed.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# Step 1: Build a flat edge table (replaces build_neighbor_lookup)
# ──────────────────────────────────────────────────────────────────────

build_edge_table <- function(cell_dt, id_order, neighbors) {
  # Build a data.table of directed neighbor pairs at the cell level:
  #   from_id -> to_id  (spatial neighbors)
  # Then cross-join with years to get row-level edges.

  # --- 1a. Expand the nb object into a flat cell-level edge list ---
  from_ref <- rep(seq_along(neighbors), lengths(neighbors))
  to_ref   <- unlist(neighbors, use.names = FALSE)

  # Remove any 0-entries (spdep uses 0 to denote "no neighbors")
  valid <- to_ref > 0L
  from_ref <- from_ref[valid]
  to_ref   <- to_ref[valid]

  cell_edges <- data.table(
    from_id = id_order[from_ref],
    to_id   = id_order[to_ref]
  )

  # --- 1b. Map (id, year) -> row index in cell_dt ---
  cell_dt[, row_idx := .I]
  id_year_map <- cell_dt[, .(id, year, row_idx)]

  # --- 1c. Expand cell edges across all years via merge ---
  # First merge: get the "from" row index (the row that will receive the feature)
  edge_rows <- merge(
    cell_edges,
    id_year_map,
    by.x = "from_id", by.y = "id",
    allow.cartesian = TRUE,
    sort = FALSE
  )
  setnames(edge_rows, c("row_idx", "year"), c("from_row", "year"))

  # Second merge: get the "to" row index (the neighbor whose value we read)
  edge_rows <- merge(
    edge_rows,
    id_year_map[, .(id, year, row_idx)],
    by.x = c("to_id", "year"), by.y = c("id", "year"),
    sort = FALSE
  )
  setnames(edge_rows, "row_idx", "to_row")

  # Keep only the two row-index columns (minimal memory)
  edge_rows <- edge_rows[, .(from_row, to_row)]
  setkey(edge_rows, from_row)

  # Clean up helper column
  cell_dt[, row_idx := NULL]

  return(edge_rows)
}

# ──────────────────────────────────────────────────────────────────────
# Step 2: Compute neighbor stats for one variable (replaces
#          compute_neighbor_stats + compute_and_add_neighbor_features)
# ──────────────────────────────────────────────────────────────────────

compute_and_add_neighbor_features_fast <- function(cell_dt, var_name, edge_table) {
  # Attach neighbor values via the edge table
  vals <- cell_dt[[var_name]]
  agg  <- edge_table[, .(neighbor_val = vals[to_row]), by = from_row]

  # Remove NAs before aggregation
  agg <- agg[!is.na(neighbor_val)]

  # Grouped aggregation — single vectorized pass
  stats <- agg[, .(
    nb_max  = max(neighbor_val),
    nb_min  = min(neighbor_val),
    nb_mean = mean(neighbor_val)
  ), by = from_row]

  # Initialize result columns with NA
  n <- nrow(cell_dt)
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  set(cell_dt, j = col_max,  value = rep(NA_real_, n))
  set(cell_dt, j = col_min,  value = rep(NA_real_, n))
  set(cell_dt, j = col_mean, value = rep(NA_real_, n))

  # Fill in computed values at the correct rows
  rows <- stats$from_row
  set(cell_dt, i = rows, j = col_max,  value = stats$nb_max)
  set(cell_dt, i = rows, j = col_min,  value = stats$nb_min)
  set(cell_dt, i = rows, j = col_mean, value = stats$nb_mean)

  invisible(cell_dt)
}

# ──────────────────────────────────────────────────────────────────────
# Step 3: Main pipeline (replaces the outer loop)
# ──────────────────────────────────────────────────────────────────────

# Convert to data.table in place (no copy if already a data.table)
setDT(cell_data)

# Build the edge table once (~38.5 M rows × 2 int cols ≈ 308 MB)
message("Building edge table...")
edge_table <- build_edge_table(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor features for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  compute_and_add_neighbor_features_fast(cell_data, var_name, edge_table)
}

# Free the edge table
rm(edge_table); gc()

message("Done. cell_data now has all neighbor features.")
```

---

## Expected Improvement Summary

| Aspect | Original | Optimized |
|---|---|---|
| **`build_neighbor_lookup`** | `lapply` over 6.46M rows, per-row `paste` + named-vector lookup | Two vectorized `data.table` merges; no per-row R loop |
| **`compute_neighbor_stats`** | `lapply` over 6.46M rows × 5 vars; `do.call(rbind, ...)` | Vectorized column extraction + `data.table` grouped `max/min/mean` |
| **Estimated runtime** | 86+ hours | ~5–15 minutes (depending on disk I/O and RAM pressure) |
| **Peak memory** | Unpredictable (ragged list + intermediate copies) | ~308 MB for edge table + ~2–3 GB for `cell_data`; well within 16 GB |
| **Numerical output** | `max`, `min`, `mean` of non-NA neighbor values | Identical — same aggregation functions, same neighbor definitions |
| **Trained RF model** | — | Untouched; features are schema-compatible |

The key insight is that the neighbor lookup is a **relational join** (cell × year matched to neighbor-cell × same-year), and `data.table` executes such joins in optimized C code with radix-based indexing — eliminating millions of interpreted R function calls.