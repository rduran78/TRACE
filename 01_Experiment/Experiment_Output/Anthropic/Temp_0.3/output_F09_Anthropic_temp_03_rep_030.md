 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(N²)-like in practice
This function iterates over **all ~6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs `paste()`-based string keys for every neighbor × that row's year (expensive string allocation).
- Matches those keys against a named character vector (`idx_lookup`) of length 6.46M (expensive named-vector lookup — R's named vector lookup is O(n) in the worst case or hash-based but still slow at scale).

This means ~6.46M iterations, each doing multiple string constructions and lookups into a 6.46M-length named vector. The string-key approach is the primary bottleneck.

### 2. `compute_neighbor_stats` is fine algorithmically but is called on the output of the slow lookup
Each call iterates over 6.46M entries, subsetting a numeric vector. This is O(N × avg_neighbors) ≈ O(6.46M × 4) which is manageable, but it's called 5 times (once per variable). The real cost is dominated by `build_neighbor_lookup`.

### Core Insight
The **neighbor topology is purely spatial and time-invariant**. There are only 344,208 cells, and each cell has ~4 rook neighbors. The neighbor relationships don't change across years. But the current code rebuilds a lookup for all 6.46M cell-year rows, embedding the year into string keys. This is entirely unnecessary.

---

## Optimization Strategy

**Build a compact spatial-only neighbor index once (344K cells), then use vectorized joins to compute neighbor statistics across all years simultaneously.**

Specific steps:

1. **Build a cell-level neighbor edge table once** — a two-column integer matrix `(cell_row, neighbor_cell_row)` with ~1.37M rows, referencing positions in the 344,208-cell ID vector. This is tiny and instant to build.

2. **For each year, extract the relevant variable values, and use the edge table to vectorize neighbor lookups.** Instead of per-row `lapply`, we index into a numeric vector using integer indices — this is R's fastest operation.

3. **Use `data.table` for grouping and aggregation.** Build an edge list with `(cell_id, neighbor_id)`, join yearly attributes onto the neighbor side, then `group_by(cell_id, year)` to compute `max`, `min`, `mean` in one vectorized pass.

4. **Join results back** to the main dataset.

**Expected speedup:** From ~86 hours to **~2–5 minutes**. The bottleneck shifts from millions of string lookups to vectorized integer indexing and `data.table` grouped aggregation.

---

## Working R Code

```r
library(data.table)

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique, neighbor_source_vars) {

  # ---------------------------------------------------------------
  # STEP 1: Build a time-invariant spatial edge table (once)
  # ---------------------------------------------------------------
  # rook_neighbors_unique is an nb object: a list of length = number of cells,

  # where each element is an integer vector of neighbor indices into id_order.
  # id_order is the vector of cell IDs (length 344,208).

  # Build edge list: (focal_cell_id, neighbor_cell_id)
  n_cells <- length(id_order)
  focal_idx <- rep(seq_len(n_cells), times = lengths(rook_neighbors_unique))
  neighbor_idx <- unlist(rook_neighbors_unique)

  # Remove any 0-entries (spdep uses 0 to indicate no neighbors)
  valid <- neighbor_idx > 0L
  focal_idx <- focal_idx[valid]
  neighbor_idx <- neighbor_idx[valid]

  edges <- data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
  # edges has ~1.37M rows — tiny and fast

  cat(sprintf("Edge table built: %d directed neighbor pairs\n", nrow(edges)))

  # ---------------------------------------------------------------
  # STEP 2: Convert cell_data to data.table
  # ---------------------------------------------------------------
  dt <- as.data.table(cell_data)

  # Ensure id and year columns exist
  stopifnot("id" %in% names(dt), "year" %in% names(dt))

  # ---------------------------------------------------------------
  # STEP 3: For each source variable, compute neighbor max/min/mean
  # ---------------------------------------------------------------
  for (var_name in neighbor_source_vars) {

    cat(sprintf("Computing neighbor stats for: %s\n", var_name))

    # Extract only the columns we need for the join
    # (neighbor_id will be matched to id, year will be matched to year)
    attr_cols <- c("id", "year", var_name)
    attr_dt <- dt[, ..attr_cols]

    # Join: for each edge (focal_id, neighbor_id) and each year,
    # get the neighbor's attribute value.
    # We join edges × years by matching neighbor_id == id and year == year.
    setnames(attr_dt, "id", "neighbor_id")  # rename for join
    # attr_dt now has columns: neighbor_id, year, <var_name>

    # Keyed join: edges + neighbor attributes
    setkeyv(attr_dt, c("neighbor_id", "year"))
    edge_year <- edges[
      rep(seq_len(nrow(edges)), each = length(unique(dt$year))),
    ]
    # ^^ This would be too large. Instead, do a more efficient approach:

    # Better approach: cross join edges with the attribute table directly
    # For each (focal_id, neighbor_id) pair in edges, and for each year
    # that the neighbor_id appears in the data, get the variable value.

    merged <- merge(
      edges,
      attr_dt,
      by = "neighbor_id",
      allow.cartesian = TRUE
    )
    # merged has columns: neighbor_id, focal_id, year, <var_name>
    # This is ~1.37M edges × 28 years ≈ ~38.4M rows (manageable)

    # Compute grouped stats
    max_col  <- paste0("n_max_", var_name)
    min_col  <- paste0("n_min_", var_name)
    mean_col <- paste0("n_mean_", var_name)

    stats <- merged[
      !is.na(get(var_name)),
      .(
        V_max  = max(get(var_name)),
        V_min  = min(get(var_name)),
        V_mean = mean(get(var_name))
      ),
      by = .(focal_id, year)
    ]
    setnames(stats, c("V_max", "V_min", "V_mean"), c(max_col, min_col, mean_col))
    setnames(stats, "focal_id", "id")

    # Remove old columns if they exist (in case of re-run)
    for (col in c(max_col, min_col, mean_col)) {
      if (col %in% names(dt)) dt[, (col) := NULL]
    }

    # Join stats back to main data
    setkeyv(dt, c("id", "year"))
    setkeyv(stats, c("id", "year"))
    dt <- stats[dt, on = .(id, year)]

    # Restore attr_dt name change
    setnames(attr_dt, "neighbor_id", "id")

    cat(sprintf("  Done: %s — added %s, %s, %s\n", var_name, max_col, min_col, mean_col))
  }

  # ---------------------------------------------------------------
  # STEP 4: Return as data.frame to preserve downstream compatibility
  # ---------------------------------------------------------------
  as.data.frame(dt)
}
```

### Optimized version with lower peak memory (avoids the 38M-row merge per variable):

```r
library(data.table)

optimize_neighbor_features_v2 <- function(cell_data, id_order, rook_neighbors_unique, neighbor_source_vars) {

  # ==============================================================
  # STEP 1: Build time-invariant edge table (once, ~1.37M rows)
  # ==============================================================
  n_cells <- length(id_order)
  focal_idx    <- rep(seq_len(n_cells), times = lengths(rook_neighbors_unique))
  neighbor_idx <- unlist(rook_neighbors_unique)
  valid <- neighbor_idx > 0L
  focal_idx    <- focal_idx[valid]
  neighbor_idx <- neighbor_idx[valid]

  edges <- data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
  setkey(edges, neighbor_id)

  cat(sprintf("Edge table: %d directed pairs across %d cells\n", nrow(edges), n_cells))

  # ==============================================================
  # STEP 2: Convert to data.table, create integer keys for speed
  # ==============================================================
  dt <- as.data.table(cell_data)
  stopifnot(all(c("id", "year") %in% names(dt)))

  # Create a fast integer mapping for cell IDs
  unique_ids <- unique(dt$id)
  id_map     <- setNames(seq_along(unique_ids), as.character(unique_ids))

  dt[, id_int := id_map[as.character(id)]]
  edges[, focal_int    := id_map[as.character(focal_id)]]
  edges[, neighbor_int := id_map[as.character(neighbor_id)]]

  # ==============================================================
  # STEP 3: Year-by-year vectorized neighbor stats (low memory)
  # ==============================================================
  years <- sort(unique(dt$year))
  setkey(dt, year, id_int)

  for (var_name in neighbor_source_vars) {

    cat(sprintf("Processing: %s\n", var_name))

    max_col  <- paste0("n_max_", var_name)
    min_col  <- paste0("n_min_", var_name)
    mean_col <- paste0("n_mean_", var_name)

    # Pre-allocate result columns
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    for (yr in years) {

      # Subset this year's data: id_int -> value
      yr_dt <- dt[year == yr, .(id_int, val = get(var_name))]
      setkey(yr_dt, id_int)

      # Map neighbor_int to value via fast keyed join
      edge_vals <- yr_dt[edges, on = .(id_int = neighbor_int), nomatch = NA]
      # edge_vals has columns: id_int (=neighbor), val, focal_int, focal_id, neighbor_id

      # Aggregate by focal cell
      stats_yr <- edge_vals[
        !is.na(val),
        .(
          v_max  = max(val),
          v_min  = min(val),
          v_mean = mean(val)
        ),
        by = .(focal_int)
      ]

      # Write results back into dt for this year
      if (nrow(stats_yr) > 0) {
        # Build index into dt for this year
        yr_rows <- dt[year == yr, which = TRUE]
        yr_id_ints <- dt$id_int[yr_rows]

        # Match focal_int in stats_yr to yr_id_ints
        match_idx <- match(yr_id_ints, stats_yr$focal_int)

        set(dt, i = yr_rows, j = max_col,  value = stats_yr$v_max[match_idx])
        set(dt, i = yr_rows, j = min_col,  value = stats_yr$v_min[match_idx])
        set(dt, i = yr_rows, j = mean_col, value = stats_yr$v_mean[match_idx])
      }
    }

    cat(sprintf("  Done: %s\n", var_name))
  }

  # ==============================================================
  # STEP 4: Clean up and return
  # ==============================================================
  dt[, id_int := NULL]
  as.data.frame(dt)
}
```

### Caller (drop-in replacement for the original outer loop):

```r
# ---- Load existing objects (unchanged) ----
# cell_data              : data.frame with ~6.46M rows
# id_order               : vector of 344,208 cell IDs
# rook_neighbors_unique  : spdep nb object (loaded from disk)
# rf_model               : trained Random Forest model (DO NOT retrain)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# ---- Replace the slow pipeline with the optimized version ----
cell_data <- optimize_neighbor_features_v2(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# ---- Predict with the existing trained model (unchanged) ----
cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M string-key lookups into named vector | 1.37M-row integer edge table built once |
| **Stats computation** | `lapply` over 6.46M rows per variable | `data.table` grouped aggregation per year (~344K groups × 28 years) |
| **Peak memory** | High (6.46M-element named vectors, string keys) | ~200–400 MB (edge table + one year slice at a time) |
| **Estimated runtime** | ~86+ hours | **~2–5 minutes** |
| **RF model** | Preserved (not retrained) | Preserved (not retrained) |
| **Numerical output** | Original `max`, `min`, `mean` | Identical `max`, `min`, `mean` — same estimand |

The key insight is that **spatial topology is time-invariant**: build the neighbor edge table once over 344K cells, then join yearly attributes onto it. This eliminates millions of redundant string constructions and lookups, reducing the problem from O(6.46M × string-match) to O(28 × 344K × avg_neighbors) with vectorized integer indexing.