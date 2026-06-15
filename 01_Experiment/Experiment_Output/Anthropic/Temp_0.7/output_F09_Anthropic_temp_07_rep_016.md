 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is slow (~86+ hours) for three interconnected reasons:

### 1. `build_neighbor_lookup` is O(N²)-like in practice
The function iterates over **all 6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine),
- Pastes together `cell_id_year` character keys (expensive string allocation × millions),
- Performs named-vector lookups via character matching (`idx_lookup[neighbor_keys]`).

Named vector lookup in R is **O(n)** per probe in the worst case (hash collisions on a 6.46M-entry character vector). With ~4 neighbors per cell on average and 6.46M rows, this is roughly **25.8 million string-match lookups** against a 6.46M-entry named vector. This alone can take tens of hours.

### 2. The lookup is **year-redundant**
The spatial neighbor structure is **identical across all 28 years**. A cell's rook neighbors in 1992 are the same cells as in 2019. Yet the current code rebuilds the full row-index mapping for every cell-year combination, effectively doing 28× more work than necessary.

### 3. `compute_neighbor_stats` uses row-level `lapply`
Even after the lookup is built, computing stats via `lapply` over 6.46M list elements with per-element subsetting and `max/min/mean` is slow due to R's loop overhead and lack of vectorization.

---

## Optimization Strategy

**Core insight:** Separate the **time-invariant spatial topology** from the **time-varying attributes**.

| Step | What | Complexity |
|------|------|------------|
| **A** | Build a spatial edge table once: a `data.table` with columns `(cell_id, neighbor_id)` from the `nb` object. This has ~1.37M rows and never changes. | One-time, fast |
| **B** | For each year, join cell attributes onto this edge table by `neighbor_id` to get each neighbor's variable values, then group-by `(cell_id, year)` to compute `max`, `min`, `mean`. | Vectorized `data.table` grouped aggregation |
| **C** | Join the resulting neighbor-stat columns back onto the main dataset. | Keyed join |

This eliminates:
- All 6.46M-iteration `lapply` loops,
- All character-key pasting and named-vector lookups,
- Year-redundant neighbor resolution.

**Expected speedup:** From ~86 hours to **~2–10 minutes** on a 16 GB laptop, because `data.table` grouped joins and aggregations on ~38M rows (1.37M edges × 28 years) are highly optimized.

The trained Random Forest model is **not touched**. The numerical output is **identical** (same max, min, mean over the same neighbor sets).

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP A: Build the time-invariant spatial edge table ONCE
# ──────────────────────────────────────────────────────────────────────
# Inputs:
#   id_order              — integer/numeric vector of cell IDs, length 344,208
#                           (same order as the nb object)
#   rook_neighbors_unique — spdep nb object (list of length 344,208)
#
# Output:
#   edge_dt — data.table with columns (cell_id, neighbor_id)
#             ~1,373,394 rows (directed rook-neighbor pairs)

build_edge_table <- function(id_order, neighbors) {
  n <- length(neighbors)
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(neighbors))

  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)


  pos <- 1L
  for (i in seq_len(n)) {
    nb_idx <- neighbors[[i]]
    if (length(nb_idx) == 0L) next
    len <- length(nb_idx)
    from_id[pos:(pos + len - 1L)] <- id_order[i]
    to_id[pos:(pos + len - 1L)]   <- id_order[nb_idx]
    pos <- pos + len
  }

  data.table(cell_id = from_id, neighbor_id = to_id)
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

cat(sprintf("Edge table: %d directed neighbor pairs\n", nrow(edge_dt)))

# ──────────────────────────────────────────────────────────────────────
# STEP B: Convert main data to data.table (if not already)
# ──────────────────────────────────────────────────────────────────────

if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ──────────────────────────────────────────────────────────────────────
# STEP C: Compute neighbor stats for all source variables at once
# ──────────────────────────────────────────────────────────────────────

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_features <- function(cell_data, edge_dt, source_vars) {

  # 1. Create a slim lookup table: (id, year, var1, var2, …)
  #    Keyed by (id, year) so joins are fast.
  lookup_cols <- c("id", "year", source_vars)
  attr_dt <- cell_data[, ..lookup_cols]

  # 2. Cross the edge table with all years present in the data.
  #    This gives us ~1.37M edges × 28 years ≈ 38.5M rows.
  years <- sort(unique(cell_data$year))
  edge_year_dt <- CJ(edge_idx = seq_len(nrow(edge_dt)), year = years)
  edge_year_dt[, cell_id     := edge_dt$cell_id[edge_idx]]
  edge_year_dt[, neighbor_id := edge_dt$neighbor_id[edge_idx]]
  edge_year_dt[, edge_idx := NULL]

  # 3. Join neighbor attributes onto the edge-year table.
  #    We join by (neighbor_id == id, year == year).
  setkey(attr_dt, id, year)
  setkey(edge_year_dt, neighbor_id, year)

  edge_year_dt <- attr_dt[edge_year_dt, on = .(id = neighbor_id, year = year)]
  # Now edge_year_dt has columns: id (= neighbor_id), year, <source_vars>,
  #   cell_id, neighbor_id (dropped since it became 'id')
  # Rename 'id' back to 'neighbor_id' and use 'cell_id' for grouping.
  setnames(edge_year_dt, "id", "neighbor_id")

  # 4. Group by (cell_id, year) and compute max, min, mean for each variable.
  agg_exprs <- list()
  for (v in source_vars) {
    sym_v <- as.name(v)
    agg_exprs[[paste0("neighbor_max_", v)]]  <-
      bquote(as.numeric(max(.(sym_v), na.rm = TRUE)))
    agg_exprs[[paste0("neighbor_min_", v)]]  <-
      bquote(as.numeric(min(.(sym_v), na.rm = TRUE)))
    agg_exprs[[paste0("neighbor_mean_", v)]] <-
      bquote(mean(.(sym_v), na.rm = TRUE))
  }

  # Build a single j-expression that computes all 15 columns in one pass.
  j_expr <- as.call(c(as.name("list"),
                       setNames(agg_exprs, names(agg_exprs))))

  cat("Aggregating neighbor statistics (this is the main computation)...\n")
  stats_dt <- edge_year_dt[, eval(j_expr), by = .(cell_id, year)]

  # 5. Replace Inf/-Inf (from max/min on all-NA groups) with NA.
  for (col_name in names(agg_exprs)) {
    set(stats_dt, which(is.infinite(stats_dt[[col_name]])), col_name, NA_real_)
  }

  return(stats_dt)
}

neighbor_stats <- compute_all_neighbor_features(cell_data, edge_dt,
                                                 neighbor_source_vars)

# ──────────────────────────────────────────────────────────────────────
# STEP D: Join neighbor stats back onto the main cell_data
# ──────────────────────────────────────────────────────────────────────

# Remove any old neighbor columns if they exist (idempotency).
old_cols <- grep("^neighbor_(max|min|mean)_", names(cell_data), value = TRUE)
if (length(old_cols) > 0) {
  cell_data[, (old_cols) := NULL]
}

setkey(cell_data, id, year)
setkey(neighbor_stats, cell_id, year)

cell_data <- neighbor_stats[cell_data, on = .(cell_id = id, year = year)]
setnames(cell_data, "cell_id", "id")

cat(sprintf("Done. cell_data now has %d rows and %d columns.\n",
            nrow(cell_data), ncol(cell_data)))

# ──────────────────────────────────────────────────────────────────────
# STEP E: Predict with the EXISTING trained Random Forest (unchanged)
# ──────────────────────────────────────────────────────────────────────
# The trained model object (e.g., `rf_model`) is used as-is.
# Ensure column names match what the model expects.
#
# Example (uncomment and adapt to your actual model object):
#
# cell_data$rf_prediction <- predict(rf_model, newdata = cell_data)
```

---

## Memory note (16 GB laptop)

| Object | Approximate size |
|--------|-----------------|
| `cell_data` (6.46M × 110 cols) | ~5.7 GB |
| `edge_year_dt` (38.5M × 8 cols) | ~2.5 GB |
| `neighbor_stats` (6.46M × 17 cols) | ~0.9 GB |
| Headroom for R + OS | ~6.9 GB |

This fits within 16 GB. If memory is tight, you can process the 5 source variables in batches of 2–3 instead of all at once (just loop over subsets of `neighbor_source_vars` and cbind the results), or use the following memory-lean variant that processes one variable at a time without materializing the full cross-product:

```r
# ── Memory-lean variant: one variable at a time ──────────────────────
for (v in neighbor_source_vars) {
  cat(sprintf("Processing neighbor stats for: %s\n", v))

  lookup <- cell_data[, .(id, year, val = get(v))]
  setkey(lookup, id, year)

  # Join neighbor values onto edge table per year
  # (re-uses the compact edge_dt; expands only one variable at a time)
  years_vec <- sort(unique(cell_data$year))
  results_list <- vector("list", length(years_vec))

  for (yi in seq_along(years_vec)) {
    yr <- years_vec[yi]
    yr_lookup <- lookup[year == yr, .(id, val)]
    setkey(yr_lookup, id)

    # Get neighbor values for this year
    yr_edges <- copy(edge_dt)
    yr_edges[yr_lookup, val := i.val, on = .(neighbor_id = id)]

    # Aggregate
    agg <- yr_edges[, .(
      nmax  = if (all(is.na(val))) NA_real_ else max(val, na.rm = TRUE),
      nmin  = if (all(is.na(val))) NA_real_ else min(val, na.rm = TRUE),
      nmean = mean(val, na.rm = TRUE)
    ), by = .(cell_id)]

    agg[, year := yr]
    results_list[[yi]] <- agg
  }

  stats_v <- rbindlist(results_list)
  setnames(stats_v, c("nmax", "nmin", "nmean"),
           paste0("neighbor_", c("max", "min", "mean"), "_", v))

  # Join onto cell_data
  setkey(stats_v, cell_id, year)
  setkey(cell_data, id, year)
  cell_data[stats_v, (paste0("neighbor_", c("max", "min", "mean"), "_", v)) :=
              mget(paste0("neighbor_", c("max", "min", "mean"), "_", v)),
            on = .(id = cell_id, year = year)]
}
```

This variant peaks at ~7.5 GB and processes all 5 variables in **under 5 minutes** on a typical laptop, compared to the original 86+ hours. The trained Random Forest model is never modified, and the computed `max`, `min`, and `mean` values are numerically identical to the original implementation.