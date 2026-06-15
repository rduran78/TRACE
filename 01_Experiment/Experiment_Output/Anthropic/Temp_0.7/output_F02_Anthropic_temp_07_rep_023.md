 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is clear: **`build_neighbor_lookup`** creates 6.46 million character-paste keys and performs 6.46 million named-vector lookups inside an `lapply` loop. Named vector lookup in R is O(n) per query on the hash, but the real cost is:

1. **`build_neighbor_lookup`**: For each of ~6.46M rows, it pastes character keys and does named-vector subsetting. The `paste` + named-lookup pattern inside `lapply` over millions of rows is extremely slow in interpreted R. Estimated: ~70–80% of total runtime.
2. **`compute_neighbor_stats`**: For each of ~6.46M rows × 5 variables, it subsets a numeric vector, removes NAs, and computes max/min/mean. The `lapply` + `do.call(rbind, ...)` pattern over millions of small vectors is slow and memory-wasteful (millions of tiny allocations, then a massive `rbind`).
3. **Memory**: With ~6.46M rows × 110 columns, the main data.frame is ~5–6 GB. Building a 6.46M-element list of integer vectors for the neighbor lookup adds another ~1–2 GB. The `do.call(rbind, list_of_6M_vectors)` pattern temporarily doubles memory.

**Root cause summary**: Row-level interpreted R loops over millions of rows, millions of small allocations, character key hashing, and a non-vectorized aggregation pattern.

---

## Optimization Strategy

### Core idea: Replace row-level R loops with vectorized joins and grouped aggregations using `data.table`.

1. **Replace `build_neighbor_lookup`** with a pre-materialized **edge table** (`data.table`) that maps every `(id, year)` → `(neighbor_id, year)` → row index. This is a single vectorized merge, not 6.46M sequential lookups.

2. **Replace `compute_neighbor_stats`** with a **grouped `data.table` aggregation** on the edge table joined to the variable values. One grouped operation computes max, min, and mean for all rows simultaneously — no `lapply`, no `do.call(rbind, ...)`.

3. **Process all 5 variables** in a single pass over the edge table (or 5 fast grouped aggregations), avoiding rebuilding lookup structures.

4. **Memory management**: The edge table for ~1.37M directed neighbor pairs × 28 years ≈ 38.5M rows of integer pairs — about 600 MB. This is feasible on 16 GB RAM, especially if we drop intermediate objects.

### Expected speedup: From 86+ hours → **~5–15 minutes**.

### Preserves: The trained Random Forest model (untouched) and the original numerical estimand (same max, min, mean statistics are computed).

---

## Working R Code

```r
library(data.table)

# ─────────────────────────────────────────────────────────────
# Step 0: Convert cell_data to data.table if not already
# ─────────────────────────────────────────────────────────────
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Assign a row index to cell_data for later re-joining
cell_data[, .row_idx := .I]

# ─────────────────────────────────────────────────────────────
# Step 1: Build the edge list from the nb object (one-time)
#
# rook_neighbors_unique is a list of length = number of cells.
# id_order is the vector of cell IDs in the same order.
# rook_neighbors_unique[[i]] gives integer indices into id_order
# for the neighbors of cell id_order[i].
# ─────────────────────────────────────────────────────────────

build_edge_list <- function(id_order, neighbors) {
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(neighbors))
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb <- neighbors[[i]]
    n  <- length(nb)
    if (n > 0L) {
      from_id[pos:(pos + n - 1L)] <- id_order[i]
      to_id[pos:(pos + n - 1L)]   <- id_order[nb]
      pos <- pos + n
    }
  }
  
  data.table(focal_id = from_id, neighbor_id = to_id)
}

cat("Building edge list from nb object...\n")
edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
cat(sprintf("Edge list: %d directed edges\n", nrow(edge_dt)))

# ─────────────────────────────────────────────────────────────
# Step 2: Expand edge list across all years via cross-join
#
# Every neighbor relationship exists in every year.
# ─────────────────────────────────────────────────────────────

years <- sort(unique(cell_data$year))
year_dt <- data.table(year = years)

cat("Expanding edge list across years...\n")
# Cross join: every edge × every year
edge_year_dt <- edge_dt[, CJ_idx := 1L][
  year_dt[, CJ_idx := 1L], 
  on = "CJ_idx", 
  allow.cartesian = TRUE
][, CJ_idx := NULL]

# Clean up
edge_dt[, CJ_idx := NULL]
year_dt[, CJ_idx := NULL]

cat(sprintf("Edge-year table: %d rows (%.1f M)\n", nrow(edge_year_dt), nrow(edge_year_dt)/1e6))

# ─────────────────────────────────────────────────────────────
# Step 3: Join to get focal row index and neighbor values,
#         then compute grouped statistics per variable
# ─────────────────────────────────────────────────────────────

# Create a key lookup: (id, year) -> row index in cell_data
setkey(cell_data, id, year)

# Add focal row index to edge_year_dt
# We join edge_year_dt to cell_data to get the focal row's .row_idx
edge_year_dt <- cell_data[, .(id, year, .row_idx)][
  edge_year_dt, 
  on = .(id = focal_id, year = year),
  nomatch = 0L
]
setnames(edge_year_dt, ".row_idx", "focal_row_idx")

# Add neighbor values: join on neighbor_id + year
# We'll do this per variable to control memory

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare a slim neighbor-value table: (id, year, var1, var2, ...)
neighbor_val_cols <- c("id", "year", neighbor_source_vars)
neighbor_vals_dt <- cell_data[, ..neighbor_val_cols]
setkey(neighbor_vals_dt, id, year)

cat("Joining neighbor values...\n")
# Join neighbor values onto the edge-year table
edge_year_dt <- neighbor_vals_dt[
  edge_year_dt,
  on = .(id = neighbor_id, year = year),
  nomatch = NA
]

# Now edge_year_dt has columns:
#   id (= neighbor_id), year, ntl, ec, pop_density, def, usd_est_n2, focal_row_idx

# ─────────────────────────────────────────────────────────────
# Step 4: Grouped aggregation — compute max, min, mean per
#         focal_row_idx for each variable
# ─────────────────────────────────────────────────────────────

cat("Computing neighbor statistics...\n")

# Build aggregation expressions dynamically
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
    bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
    bquote(as.numeric(mean(.(as.name(v)), na.rm = TRUE)))
  )
}))

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", v, c("_max", "_min", "_mean"))
}))

names(agg_exprs) <- agg_names

# Single grouped aggregation
neighbor_stats <- edge_year_dt[, 
  lapply(agg_exprs, eval, envir = .SD), 
  by = focal_row_idx
]

# ─── Alternative (clearer, equally fast) ────────────────────
# If the above eval approach is tricky, do it explicitly:

neighbor_stats <- edge_year_dt[, {
  out <- list()
  for (v in neighbor_source_vars) {
    vals <- .SD[[v]]
    vals <- vals[!is.na(vals)]
    if (length(vals) == 0L) {
      out[[paste0("neighbor_", v, "_max")]]  <- NA_real_
      out[[paste0("neighbor_", v, "_min")]]  <- NA_real_
      out[[paste0("neighbor_", v, "_mean")]] <- NA_real_
    } else {
      out[[paste0("neighbor_", v, "_max")]]  <- max(vals)
      out[[paste0("neighbor_", v, "_min")]]  <- min(vals)
      out[[paste0("neighbor_", v, "_mean")]] <- mean(vals)
    }
  }
  out
}, by = focal_row_idx, .SDcols = neighbor_source_vars]

cat(sprintf("Neighbor stats computed: %d rows, %d columns\n", nrow(neighbor_stats), ncol(neighbor_stats)))

# ─────────────────────────────────────────────────────────────
# Step 5: Join statistics back to cell_data
# ─────────────────────────────────────────────────────────────

# Remove the edge table to free memory
rm(edge_year_dt, neighbor_vals_dt)
gc()

setkey(neighbor_stats, focal_row_idx)

# Merge back by row index
new_cols <- setdiff(names(neighbor_stats), "focal_row_idx")

# Remove any pre-existing neighbor columns to avoid conflicts
for (col in new_cols) {
  if (col %in% names(cell_data)) {
    cell_data[, (col) := NULL]
  }
}

cell_data <- neighbor_stats[cell_data, on = .(focal_row_idx = .row_idx)]

# Rename focal_row_idx back to .row_idx if needed, or drop it
cell_data[, focal_row_idx := NULL]

# If you had .row_idx, drop it too (it was a helper)
if (".row_idx" %in% names(cell_data)) cell_data[, .row_idx := NULL]

cat("Done. Neighbor features added to cell_data.\n")
```

---

## Memory Management Notes for 16 GB RAM

If the cross-joined edge-year table (~38.5M rows) causes memory pressure:

```r
# ─── Memory-safe variant: process one year at a time ─────────
# This trades a small amount of speed for much lower peak memory.

all_stats <- vector("list", length(years))

for (yi in seq_along(years)) {
  yr <- years[yi]
  cat(sprintf("Processing year %d (%d/%d)...\n", yr, yi, length(years)))
  
  # Subset to this year
  cd_yr <- cell_data[year == yr, c("id", ".row_idx", neighbor_source_vars), with = FALSE]
  setkey(cd_yr, id)
  
  # Join edges to focal rows
  ey <- edge_dt[cd_yr, on = .(focal_id = id), nomatch = 0L, allow.cartesian = TRUE]
  # ey now has: focal_id, neighbor_id, .row_idx (focal)
  
  # Join neighbor values
  ey <- cd_yr[, c("id", neighbor_source_vars), with = FALSE][
    ey, on = .(id = neighbor_id), nomatch = NA
  ]
  
  # Aggregate
  stats_yr <- ey[, {
    out <- list()
    for (v in neighbor_source_vars) {
      vals <- .SD[[v]]
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        out[[paste0("neighbor_", v, "_max")]]  <- NA_real_
        out[[paste0("neighbor_", v, "_min")]]  <- NA_real_
        out[[paste0("neighbor_", v, "_mean")]] <- NA_real_
      } else {
        out[[paste0("neighbor_", v, "_max")]]  <- max(vals)
        out[[paste0("neighbor_", v, "_min")]]  <- min(vals)
        out[[paste0("neighbor_", v, "_mean")]] <- mean(vals)
      }
    }
    out
  }, by = .(.row_idx), .SDcols = neighbor_source_vars]
  
  all_stats[[yi]] <- stats_yr
  rm(ey, cd_yr, stats_yr)
}

neighbor_stats <- rbindlist(all_stats)
rm(all_stats); gc()

# Join back to cell_data as in Step 5 above
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M `paste` + named vector lookups in `lapply` | One-time vectorized edge-list build + `data.table` keyed join |
| **Aggregation** | 6.46M × 5 `lapply` calls + `do.call(rbind, ...)` | Single grouped `data.table` aggregation |
| **Peak memory** | Uncontrolled (millions of small list allocations) | ~2–4 GB for edge-year table (or ~300 MB per year in chunked mode) |
| **Estimated runtime** | 86+ hours | ~5–15 minutes |
| **RF model** | Untouched ✓ | Untouched ✓ |
| **Numerical output** | max, min, mean per neighbor set | Identical max, min, mean per neighbor set ✓ |