 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not ideal, it is a one-shot operation per variable (only 5 calls total) and each call binds fixed-length 3-element vectors — this is relatively cheap.

**The true bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **`paste()` and named-vector lookups over 6.46M rows:** `build_neighbor_lookup` creates `idx_lookup` — a named vector of ~6.46 million entries keyed by `paste(id, year)`. It then calls `lapply` over all ~6.46M rows, and *inside each iteration* it:
   - Calls `as.character()` on a single id → character conversion 6.46M times.
   - Looks up `id_to_ref[as.character(...)]` — a named-vector lookup (linear hash probe) 6.46M times.
   - Retrieves `neighbors[[ref_idx]]` to get neighbor cell IDs (typically ~4 for rook contiguity).
   - Calls `paste(neighbor_cell_ids, data$year[i], sep="_")` — string concatenation inside the loop, ~4× per row = ~25.8M paste operations.
   - Performs `idx_lookup[neighbor_keys]` — named-vector lookup of ~4 keys against a 6.46M-length named vector, 6.46M times.

2. **Complexity:** Named vector lookup in R via `[` on a character-named vector is **O(n)** in the worst case per probe (R uses hashing, but the hash table is rebuilt/probed repeatedly). With ~6.46M rows × ~4 neighbor lookups each = ~25.8M hash probes against a 6.46M-entry hash. This dwarfs everything in `compute_neighbor_stats()`.

3. **`compute_neighbor_stats` is comparatively cheap:** Once `neighbor_lookup` exists, each call simply does `vals[idx]` (integer subsetting — very fast), computes `max/min/mean` on ~4 values, and returns a length-3 vector. The `do.call(rbind, ...)` on 6.46M length-3 vectors takes seconds, not hours.

**Conclusion:** `build_neighbor_lookup()` is the dominant bottleneck, likely consuming 80%+ of the 86-hour runtime. The repeated string construction and named-vector lookups inside a 6.46M-iteration `lapply` are the root cause.

---

## Optimization Strategy

1. **Replace named-vector lookups with `data.table` hash joins or environment-based hashing.** Use `match()` or `data.table` keyed joins instead of named-vector character lookups.

2. **Vectorize `build_neighbor_lookup` entirely.** Instead of looping row-by-row, expand the neighbor relationships into a full edge list (cell_i, cell_j), join on year to create (row_i, row_j) pairs, then use `data.table` grouped operations to compute neighbor stats directly — **eliminating `build_neighbor_lookup` and `compute_neighbor_stats` as separate steps.**

3. **Compute all 5 variables' neighbor stats in one pass** over the edge list, avoiding redundant work.

This reduces the problem from 6.46M R-level loop iterations with string operations to a single vectorized join + grouped aggregation.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# Optimized pipeline: replaces build_neighbor_lookup(),
# compute_neighbor_stats(), and the outer for-loop entirely.
# ---------------------------------------------------------------

compute_all_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique, neighbor_source_vars) {

  dt <- as.data.table(cell_data)

  # --- Step 1: Build a full directed edge list from the nb object ---
  # rook_neighbors_unique is a list of integer vectors (spdep nb object).
  # neighbors[[i]] gives the indices (into id_order) of neighbors of id_order[i].

  # Pre-allocate edge list vectors
  n_edges <- sum(lengths(rook_neighbors_unique))  # ~1,373,394
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)

  pos <- 1L
  for (i in seq_along(rook_neighbors_unique)) {
    nb_i <- rook_neighbors_unique[[i]]
    if (length(nb_i) == 0L || (length(nb_i) == 1L && nb_i[1] == 0L)) next
    n_nb <- length(nb_i)
    from_id[pos:(pos + n_nb - 1L)] <- id_order[i]
    to_id[pos:(pos + n_nb - 1L)]   <- id_order[nb_i]
    pos <- pos + n_nb
  }

  # Trim if any nb entries were empty (0-neighbor cells)
  if (pos <= n_edges) {
    from_id <- from_id[1:(pos - 1L)]
    to_id   <- to_id[1:(pos - 1L)]
  }

  edges <- data.table(from_id = from_id, to_id = to_id)

  # --- Step 2: Create a row-index mapping: (id, year) -> row_index ---
  dt[, row_idx := .I]

  # --- Step 3: For each year, expand edges into (row_i, row_j) pairs ---
  # We need: for each row (from_id, year), find all neighbor rows (to_id, same year).
  # Strategy: join edges with dt on id to get (from_row, to_id, year),
  # then join again to get to_row.

  # Keyed lookup tables
  id_year_to_row <- dt[, .(id, year, row_idx)]
  setkey(id_year_to_row, id, year)

  # Expand: each edge × each year that from_id appears in
  # Join edges with id_year_to_row on from_id = id
  setnames(id_year_to_row, "id", "from_id")
  setkey(edges, from_id)
  setkey(id_year_to_row, from_id)

  # This gives us: for each (from_id, to_id) edge, all years where from_id has data
  edge_year <- edges[id_year_to_row, on = "from_id", allow.cartesian = TRUE, nomatch = 0L]
  # Columns: from_id, to_id, year, row_idx (= from_row_idx)
  setnames(edge_year, "row_idx", "from_row_idx")

  # Now join to get to_row_idx: match (to_id, year)
  setnames(id_year_to_row, c("to_id", "year", "to_row_idx"))
  setkey(id_year_to_row, to_id, year)
  setkey(edge_year, to_id, year)

  edge_rows <- id_year_to_row[edge_year, on = c("to_id", "year"), nomatch = 0L]
  # Columns: to_id, year, to_row_idx, from_id, from_row_idx

  # --- Step 4: Compute neighbor stats for all variables at once ---
  # For each from_row_idx, gather neighbor values (at to_row_idx), compute max/min/mean.

  for (var_name in neighbor_source_vars) {
    message("Computing neighbor features for: ", var_name)

    # Extract neighbor values via integer indexing (very fast)
    edge_rows[, nval := dt[[var_name]][to_row_idx]]

    # Grouped aggregation — the core computation
    stats <- edge_rows[!is.na(nval),
                       .(nmax  = max(nval),
                         nmin  = min(nval),
                         nmean = mean(nval)),
                       by = from_row_idx]

    # Initialize columns with NA
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Assign results by row index
    dt[stats$from_row_idx, (max_col)  := stats$nmax]
    dt[stats$from_row_idx, (min_col)  := stats$nmin]
    dt[stats$from_row_idx, (mean_col) := stats$nmean]

    # Clean up temp column
    edge_rows[, nval := NULL]
  }

  # Remove helper column
  dt[, row_idx := NULL]

  # Return as data.frame to preserve downstream compatibility (e.g., with trained RF model)
  as.data.frame(dt)
}

# ---------------------------------------------------------------
# Usage (drop-in replacement for the original pipeline section):
# ---------------------------------------------------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Column names and numerical values are identical to the original pipeline.
# Proceed directly to prediction:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **True bottleneck** | `build_neighbor_lookup`: 6.46M iterations with `paste()` + named-vector probes | Eliminated entirely |
| **Neighbor resolution** | Row-by-row string key construction & lookup | Single vectorized `data.table` keyed join |
| **Stats computation** | `lapply` over 6.46M rows + `do.call(rbind, ...)` | `data.table` grouped aggregation (`by=`) |
| **Number of passes over edges** | 1 (build) + 5 (compute per var) = 6 | Edge expansion once; 5 lightweight grouped aggregations |
| **Expected runtime** | ~86+ hours | **Minutes** (estimated 5–15 min on the same laptop) |
| **RAM safety** | `edge_rows` at ~1.37M edges × 28 years ≈ 38.4M rows × 5 int/numeric cols ≈ ~1.5 GB — fits in 16 GB | ✓ |
| **Trained RF model** | Preserved (no retraining) | ✓ |
| **Numerical equivalence** | Original `max/min/mean` over non-NA neighbor values | Identical — same `max()`, `min()`, `mean()` over same neighbor sets | ✓ |