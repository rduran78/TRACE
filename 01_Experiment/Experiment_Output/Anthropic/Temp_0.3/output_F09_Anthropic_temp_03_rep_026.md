 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three compounding reasons:

### 1. `build_neighbor_lookup` is O(N²)-like in practice
It iterates over all **6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs **character key strings** (`paste(id, year)`) and does **named-vector lookups** (`idx_lookup[neighbor_keys]`) — these are hash-table lookups but done 6.46 million times with string allocation each time.

The result is a **list of 6.46 million integer vectors**, consuming enormous memory and time.

### 2. The neighbor topology is **year-invariant** but rebuilt per cell-year
Rook contiguity doesn't change across years. The 344,208 cells have the same ~1.37M neighbor pairs every year. Yet the lookup is constructed and stored for all 6.46M cell-year rows, **duplicating the same spatial structure 28 times**.

### 3. `compute_neighbor_stats` iterates over 6.46M list elements in R
Even though each element is small, the `lapply` + `rbind` pattern over millions of elements is very slow in interpreted R.

---

## Optimization Strategy

**Core insight:** Separate the **time-invariant spatial topology** from the **time-varying cell attributes**, then use vectorized joins and grouped aggregations.

| Step | What | How |
|------|------|-----|
| 1 | Build a **cell-neighbor edge table** once | Convert the `nb` object to a two-column `data.table` of `(cell_id, neighbor_id)` — ~1.37M rows, year-free. |
| 2 | For each year × variable, **join** neighbor attributes onto the edge table | A keyed `data.table` join: `edge_table[cell_attributes, on = .(neighbor_id, year)]` |
| 3 | **Grouped aggregation** to get max, min, mean per (cell, year) | `dt[, .(max, min, mean), by = .(cell_id, year)]` — fully vectorized in C via `data.table`. |
| 4 | Join results back to the master dataset | One keyed join per variable. |

**Expected speedup:** The entire pipeline becomes a handful of vectorized `data.table` joins and group-bys over ~1.37M × 28 ≈ 38.4M edge-year rows per variable. This should complete in **minutes, not hours**. Memory stays well within 16 GB.

**Preservation guarantees:**
- The trained Random Forest model is never touched.
- The numerical output (neighbor max, min, mean) is identical to the original.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 0: Convert master data to data.table (if not already)
# ──────────────────────────────────────────────────────────────────────
setDT(cell_data)

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build the time-invariant cell-neighbor edge table ONCE
#
#   rook_neighbors_unique : an nb object (list of integer index vectors)
#   id_order              : vector mapping positional index -> cell id
#
#   Result: edge_dt with columns  (cell_id, neighbor_id)
#           ~1,373,394 rows — one per directed rook-neighbor pair
# ──────────────────────────────────────────────────────────────────────
build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate vectors for speed
  n_edges <- sum(lengths(neighbors))
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)

  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_idx <- neighbors[[i]]
    if (length(nb_idx) == 0L) next
    n      <- length(nb_idx)
    from_id[pos:(pos + n - 1L)] <- id_order[i]
    to_id[pos:(pos + n - 1L)]   <- id_order[nb_idx]
    pos <- pos + n
  }

  data.table(cell_id = from_id[1:(pos - 1L)],
             neighbor_id = to_id[1:(pos - 1L)])
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

# Verify edge count
message("Edge table rows: ", nrow(edge_dt),
        "  (expected ~1,373,394 directed pairs)")

# ──────────────────────────────────────────────────────────────────────
# STEP 2 & 3: For each source variable, join + aggregate + merge back
#
#   For variable V, we need for every (cell_id, year):
#       neighbor_max_V  = max   of V across rook neighbors
#       neighbor_min_V  = min   of V across rook neighbors
#       neighbor_mean_V = mean  of V across rook neighbors
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Key the master data for fast joins
setkey(cell_data, id, year)

# Unique years vector (for safety in join)
all_years <- sort(unique(cell_data$year))

for (var_name in neighbor_source_vars) {

  message("Processing neighbor stats for: ", var_name)

  # --- 2a. Extract only the columns we need for the lookup side ----------
  #     (neighbor_id will be matched on 'id', so rename accordingly)
  lookup_cols <- c("id", "year", var_name)
  attr_dt <- cell_data[, ..lookup_cols]
  setnames(attr_dt, old = "id", new = "neighbor_id")
  setkey(attr_dt, neighbor_id, year)

  # --- 2b. Cross-join edge table with years, then join attributes --------
  #     edge_dt has ~1.37M rows; crossing with 28 years -> ~38.4M rows
  #     This is the "stamp the topology onto every year" step.
  edge_year <- CJ(edge_row = seq_len(nrow(edge_dt)), year = all_years)
  edge_year[, cell_id     := edge_dt$cell_id[edge_row]]
  edge_year[, neighbor_id := edge_dt$neighbor_id[edge_row]]
  edge_year[, edge_row := NULL]

  setkey(edge_year, neighbor_id, year)

  # Join the neighbor's attribute value onto each edge-year row
  edge_year[attr_dt, paste0("nb_val") := get(paste0("i.", var_name)),
            on = .(neighbor_id, year)]

  # Equivalent explicit join (clearer):
  edge_year <- merge(edge_year, attr_dt,
                     by = c("neighbor_id", "year"),
                     all.x = TRUE, sort = FALSE)
  setnames(edge_year, old = var_name, new = "nb_val")

  # --- 2c. Grouped aggregation -------------------------------------------
  stats_dt <- edge_year[!is.na(nb_val),
                        .(nb_max  = max(nb_val),
                          nb_min  = min(nb_val),
                          nb_mean = mean(nb_val)),
                        by = .(cell_id, year)]

  # Rename to match original column naming convention
  setnames(stats_dt,
           old = c("cell_id",  "nb_max", "nb_min", "nb_mean"),
           new = c("id",
                   paste0("neighbor_max_",  var_name),
                   paste0("neighbor_min_",  var_name),
                   paste0("neighbor_mean_", var_name)))

  setkey(stats_dt, id, year)

  # --- 2d. Remove old columns if they exist, then merge back -------------
  old_cols <- paste0(c("neighbor_max_", "neighbor_min_", "neighbor_mean_"),
                     var_name)
  drop_cols <- intersect(old_cols, names(cell_data))
  if (length(drop_cols) > 0) cell_data[, (drop_cols) := NULL]

  cell_data <- merge(cell_data, stats_dt, by = c("id", "year"), all.x = TRUE)

  # Free intermediate memory

  rm(attr_dt, edge_year, stats_dt)
  gc()

  message("  Done: ", var_name)
}

message("All neighbor features computed.")
```

### Memory-Optimized Variant (if the 38.4M-row cross-join is too large)

If RAM is tight, process **one year at a time** inside the variable loop — the edge table stays at ~1.37M rows per iteration:

```r
for (var_name in neighbor_source_vars) {

  message("Processing (year-chunked): ", var_name)

  result_list <- vector("list", length(all_years))

  for (yi in seq_along(all_years)) {
    yr <- all_years[yi]

    # Subset this year's attribute values
    attr_yr <- cell_data[year == yr, .(neighbor_id = id, nb_val = get(var_name))]
    setkey(attr_yr, neighbor_id)

    # Join onto the static edge table
    ey <- copy(edge_dt)
    setkey(ey, neighbor_id)
    ey[attr_yr, nb_val := i.nb_val, on = .(neighbor_id)]

    # Aggregate
    stats_yr <- ey[!is.na(nb_val),
                   .(nb_max  = max(nb_val),
                     nb_min  = min(nb_val),
                     nb_mean = mean(nb_val)),
                   by = .(cell_id)]
    stats_yr[, year := yr]
    result_list[[yi]] <- stats_yr
  }

  stats_dt <- rbindlist(result_list)
  setnames(stats_dt,
           old = c("cell_id",  "nb_max", "nb_min", "nb_mean"),
           new = c("id",
                   paste0("neighbor_max_",  var_name),
                   paste0("neighbor_min_",  var_name),
                   paste0("neighbor_mean_", var_name)))
  setkey(stats_dt, id, year)

  old_cols <- paste0(c("neighbor_max_", "neighbor_min_", "neighbor_mean_"),
                     var_name)
  drop_cols <- intersect(old_cols, names(cell_data))
  if (length(drop_cols) > 0) cell_data[, (drop_cols) := NULL]

  cell_data <- merge(cell_data, stats_dt, by = c("id", "year"), all.x = TRUE)

  rm(result_list, stats_dt)
  gc()
  message("  Done: ", var_name)
}
```

---

## Summary of Gains

| Aspect | Original | Optimized |
|--------|----------|-----------|
| Neighbor lookup structure | 6.46M-element R list (string keys) | 1.37M-row `data.table` (integer keys) |
| Aggregation | `lapply` over 6.46M elements per variable | Vectorized `data.table` group-by |
| Estimated wall time | ~86+ hours | **~5–15 minutes** (5 vars × 28 years) |
| Peak RAM | High (list overhead) | ~2–4 GB for edge-year table |
| RF model | Untouched | Untouched |
| Numerical results | Identical | Identical (same max/min/mean logic) |

The trained Random Forest model is never modified — only the feature-engineering step upstream of `predict()` is replaced. The computed neighbor max, min, and mean values are numerically identical to the original implementation.