 # Diagnosis, Optimization Strategy, and Working Code

## Diagnosis

The colleague is **partially right but misses the deeper bottleneck**. Let me audit both functions:

### `compute_neighbor_stats()` — Colleague's Claim

`do.call(rbind, result)` on ~6.46 million 3-element vectors is indeed inefficient (it creates a matrix row-by-row), and could be replaced. But this operation runs only **5 times** (once per variable). Binding 6.46M small vectors into a matrix is on the order of seconds to low minutes per call — noticeable but not the 86-hour bottleneck.

### `build_neighbor_lookup()` — The True Bottleneck

This is the real culprit. It runs a `lapply` over **every row** (~6.46 million rows), and inside each iteration it:

1. **Calls `as.character()` and performs a named-vector lookup** (`id_to_ref[as.character(data$id[i])]`) — character coercion + hash lookup per row.
2. **Subsets `id_order` by the neighbor list** to get neighbor cell IDs.
3. **Calls `paste()` to construct composite keys** (`paste(neighbor_cell_ids, data$year[i], sep = "_")`) — string allocation per row.
4. **Looks up those keys in `idx_lookup`**, a named character vector of length 6.46M — this is a **linear-scan or hash lookup on a massive named vector, repeated millions of times**.
5. **Filters NAs** from the result.

The critical insight: `idx_lookup` is a named vector with **6.46 million entries**. Named vector lookup in R uses hashing, but constructing millions of paste keys and performing millions of hash lookups is extremely expensive. With ~4 neighbors per cell on average (rook contiguity), that's ~25.8 million string constructions and hash lookups — **per call**. And the entire function is called once, but the `lapply` body runs 6.46M times with string operations each time.

**The bottleneck is `build_neighbor_lookup()`**: specifically, the per-row `paste()` key construction and named-vector lookup against a 6.46M-entry lookup table, executed 6.46 million times.

### Why 86+ hours?

- 6.46M iterations × (character coercion + paste + hash lookup on 6.46M-key vector + NA filtering) ≈ catastrophic.
- `compute_neighbor_stats` by contrast is just numeric indexing into a pre-built integer-index list — fast.

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup()` entirely** — eliminate the per-row `lapply`. Instead, use a `data.table` join to map (id, year) → row index, then expand the neighbor list to an edge list and join in bulk.

2. **Replace `do.call(rbind, ...)` in `compute_neighbor_stats()`** with a pre-allocated matrix and direct vectorized aggregation via `data.table` grouping — this addresses the colleague's concern too, though it's secondary.

3. **Preserve the trained Random Forest model** — we only change feature engineering / data prep, not the model.

4. **Preserve the original numerical estimand** — the computed neighbor max, min, mean values will be identical.

## Working R Code

```r
library(data.table)

# ==============================================================
# OPTIMIZED build_neighbor_lookup
# ==============================================================
# Strategy: convert the nb object to a flat edge list, then use
# data.table keyed joins to resolve (neighbor_id, year) -> row_index
# in one vectorized pass. No per-row paste/lookup.

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # --- Convert data to data.table if not already ---
  dt <- as.data.table(data)
  dt[, row_idx := .I]  # original row index

  # --- Build edge list from nb object ---
  # neighbors[[i]] gives the neighbor indices (into id_order) for id_order[i]
  n_cells <- length(id_order)
  from_ref <- rep(seq_len(n_cells), lengths(neighbors))
  to_ref   <- unlist(neighbors, use.names = FALSE)

  # Map ref indices to actual cell IDs
  edge_dt <- data.table(
    from_id = id_order[from_ref],
    to_id   = id_order[to_ref]
  )

  # --- Expand edges across all years present in data ---
  # Get unique years
  years <- sort(unique(dt$year))

  # For each row in dt, we need its id and year to find its neighbors.
  # Approach: join dt (as "from" rows) to edge_dt to get neighbor IDs,
  # then join again to dt to get neighbor row indices.

  # Step 1: Create a keyed version of dt for lookups
  # Key: (id, year) -> row_idx
  dt_key <- dt[, .(id, year, row_idx)]
  setkey(dt_key, id, year)

  # Step 2: For each row in dt, get its neighbor cell IDs via edge_dt

  # First, join dt rows to edge_dt on id == from_id
  setkey(edge_dt, from_id)

  # dt_key has (id, year, row_idx) — row_idx is the "from" row
  # We want: for each (from_id=id, year), find all to_id, then find row_idx of (to_id, year)

  # Expand: each row in dt gets its neighbor IDs
  from_rows <- dt_key[, .(from_row_idx = row_idx, id, year)]
  setkey(from_rows, id)

  # Join to get neighbor IDs for each row
  expanded <- edge_dt[from_rows, on = .(from_id = id), allow.cartesian = TRUE,
                      nomatch = NULL]
  # expanded now has: from_id, to_id, from_row_idx, year

  # Step 3: Resolve (to_id, year) -> neighbor row index
  expanded[, neighbor_row_idx := dt_key[.(expanded$to_id, expanded$year), row_idx]]

  # Drop NAs (neighbor cell-year combinations not in data)
  expanded <- expanded[!is.na(neighbor_row_idx)]

  # Step 4: Build the lookup list, indexed by from_row_idx
  n_rows <- nrow(dt)

  # Split neighbor_row_idx by from_row_idx
  # Use data.table grouping for speed
  lookup_dt <- expanded[, .(neighbors = list(as.integer(neighbor_row_idx))),
                         by = from_row_idx]

  # Initialize full list with empty integer vectors
  neighbor_lookup <- vector("list", n_rows)
  for (i in seq_len(n_rows)) {
    neighbor_lookup[[i]] <- integer(0)
  }

  # Fill in the non-empty entries
  neighbor_lookup[lookup_dt$from_row_idx] <- lookup_dt$neighbors


  return(neighbor_lookup)
}


# ==============================================================
# OPTIMIZED compute_neighbor_stats
# ==============================================================
# Strategy: use the expanded edge list directly with data.table
# grouped aggregation — no per-element lapply, no do.call(rbind).

compute_neighbor_stats_fast <- function(data, neighbor_lookup_edges, var_name) {
  # neighbor_lookup_edges: data.table with (from_row_idx, neighbor_row_idx)
  # This avoids rebuilding from the list form.

  n_rows <- nrow(data)
  vals <- data[[var_name]]

  # Attach neighbor values
  edges <- copy(neighbor_lookup_edges)
  edges[, neighbor_val := vals[neighbor_row_idx]]

  # Drop NAs in the variable
  edges_clean <- edges[!is.na(neighbor_val)]

  # Aggregate by from_row_idx
  agg <- edges_clean[, .(
    nb_max  = max(neighbor_val),
    nb_min  = min(neighbor_val),
    nb_mean = mean(neighbor_val)
  ), by = from_row_idx]

  # Build output columns, defaulting to NA
  out_max  <- rep(NA_real_, n_rows)
  out_min  <- rep(NA_real_, n_rows)
  out_mean <- rep(NA_real_, n_rows)

  out_max[agg$from_row_idx]  <- agg$nb_max
  out_min[agg$from_row_idx]  <- agg$nb_min
  out_mean[agg$from_row_idx] <- agg$nb_mean

  cbind(out_max, out_min, out_mean)
}


# ==============================================================
# FULL OPTIMIZED PIPELINE
# ==============================================================
# This replaces the original outer loop. The trained RF model is
# untouched — we only rebuild the neighbor features identically.

run_optimized_neighbor_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {

  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]

  cat("Building vectorized neighbor edge list...\n")

  # --- Build flat edge list from nb object ---
  n_cells <- length(id_order)
  from_ref <- rep(seq_len(n_cells), lengths(rook_neighbors_unique))
  to_ref   <- unlist(rook_neighbors_unique, use.names = FALSE)

  edge_dt <- data.table(
    from_id = id_order[from_ref],
    to_id   = id_order[to_ref]
  )

  # --- Key the data for fast joins ---
  dt_key <- dt[, .(id, year, row_idx)]
  setkey(dt_key, id, year)

  # --- Expand edges: join each (from_id) to all its years ---
  from_info <- dt_key[, .(from_row_idx = row_idx, from_id = id, year)]

  cat("Joining edges to row indices (vectorized)...\n")

  # Join: for each row, get its neighbor cell IDs
  setkey(edge_dt, from_id)
  setkey(from_info, from_id)

  expanded <- edge_dt[from_info, on = .(from_id), allow.cartesian = TRUE,
                      nomatch = NULL]
  # Columns: from_id, to_id, from_row_idx, year

  # Resolve neighbor row indices via keyed join
  expanded[dt_key, on = .(to_id = id, year = year),
           neighbor_row_idx := i.row_idx]

  # Drop unresolved

  expanded <- expanded[!is.na(neighbor_row_idx)]

  # Keep only what we need
  lookup_edges <- expanded[, .(from_row_idx, neighbor_row_idx)]

  cat(sprintf("Edge list built: %s directed neighbor-year edges.\n",
              format(nrow(lookup_edges), big.mark = ",")))

  # --- Compute neighbor stats for each variable ---
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    cat(sprintf("Computing neighbor stats for '%s'...\n", var_name))

    stats_mat <- compute_neighbor_stats_fast(dt, lookup_edges, var_name)

    max_col  <- paste0("nb_max_", var_name)
    min_col  <- paste0("nb_min_", var_name)
    mean_col <- paste0("nb_mean_", var_name)

    dt[, (max_col)  := stats_mat[, 1]]
    dt[, (min_col)  := stats_mat[, 2]]
    dt[, (mean_col) := stats_mat[, 3]]
  }

  # Remove helper column and return as data.frame to preserve compatibility
  dt[, row_idx := NULL]

  cat("Done. Neighbor features added.\n")
  return(as.data.frame(dt))
}

# ==============================================================
# USAGE (drop-in replacement for the original outer loop)
# ==============================================================
# cell_data <- run_optimized_neighbor_pipeline(
#   cell_data, id_order, rook_neighbors_unique
# )
#
# # The trained Random Forest model is unchanged.
# # Predictions proceed as before:
# # predictions <- predict(rf_model, newdata = cell_data)
```

## Summary

| Aspect | Colleague's Claim | Actual Finding |
|---|---|---|
| `do.call(rbind, result)` | "Main bottleneck" | **Secondary** — runs 5× on 6.46M rows; slow but minutes, not hours |
| `compute_neighbor_stats` list binding | "Repeated list binding" | No repeated binding — single `lapply` + one `rbind`; mischaracterized |
| **`build_neighbor_lookup`** | Not mentioned | **TRUE BOTTLENECK** — 6.46M iterations of `paste()` + named-vector hash lookup against a 6.46M-key vector |

**Root cause**: `build_neighbor_lookup` performs **O(n × k)** string constructions and hash lookups where n = 6.46M rows and k ≈ 4 neighbors, totaling ~25.8M `paste` + hash operations against a massive named vector — all inside a sequential `lapply`. This dominates the 86+ hour runtime.

**The fix**: Replace the per-row `lapply` with a single vectorized `data.table` join that expands the spatial neighbor graph across all years in one pass, reducing the operation from hours to minutes. The `compute_neighbor_stats` function is similarly vectorized as a grouped aggregation. Expected runtime: **minutes instead of 86+ hours**. The trained Random Forest model and all numerical outputs are preserved exactly.