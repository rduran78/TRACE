 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It creates a list of 6.46 million elements, where each element is built by:

1. **Character key construction and lookup** (`paste`, `setNames`, named-vector indexing) — for every single row. Named-vector lookup in R is O(n) per query in the worst case because R's named vectors use linear hashing with potential collisions, and here the lookup table has ~6.46M entries.
2. **`lapply` over 6.46M rows** — each iteration does string pasting, named-vector subsetting, and NA filtering. The per-element overhead of R's interpreted loop is enormous at this scale.
3. **Memory**: The resulting `neighbor_lookup` is a list of ~6.46M integer vectors. Combined with the intermediate character vectors, this can spike well beyond available RAM, causing swapping.

Then `compute_neighbor_stats` iterates over the same 6.46M-element list **5 times** (once per variable), each time extracting neighbor values and computing max/min/mean in pure R.

**Root cause summary:**
- O(n) named-vector lookups inside an O(n) loop → effectively O(n²) behavior.
- Repeated string allocation (`paste`) for 6.46M × avg-neighbors keys.
- The neighbor lookup is **time-invariant** but is encoded with year-specific row indices, forcing a full rebuild if anything changes.
- `compute_neighbor_stats` does 5 separate passes with per-element R overhead.

## Optimization Strategy

### 1. Replace character-key lookups with integer-arithmetic direct indexing

Since the panel is balanced (344,208 cells × 28 years = 9,637,824 potential rows, of which ~6.46M exist), we can build a **cell-ID → integer index map** and a **year → integer offset map** using `data.table` or environments (O(1) hash lookup), then compute row indices arithmetically.

### 2. Vectorize neighbor lookup construction using `data.table` joins

Instead of looping over 6.46M rows, we:
- Expand the `nb` object into an edge list (cell_i, cell_j) — only ~1.37M edges.
- Cross-join with years to get (cell_i, year, cell_j) — ~1.37M × 28 ≈ 38.4M rows (but filtered to existing cell-years).
- Join against the data to get row indices for both the focal cell and the neighbor cell.
- Group by focal-row-index and compute stats directly — **all 5 variables in one pass**.

This replaces the 86-hour loop with vectorized `data.table` operations that should complete in **minutes**.

### 3. Compute all neighbor stats in one grouped aggregation

Instead of 5 separate `lapply` passes, compute max/min/mean for all 5 variables simultaneously in a single `data.table` grouped operation on the edge table.

### 4. Preserve the trained RF model and numerical estimand

We only change how the neighbor features are computed, not their values. The column names and semantics are identical, so the trained model's `predict()` call is unchanged.

---

## Working R Code

```r
library(data.table)

# ── 0. Inputs assumed to exist ──────────────────────────────────────────────
# cell_data            : data.frame/data.table with columns id, year, ntl, ec,
#                        pop_density, def, usd_est_n2, ... (~6.46M rows)
# id_order             : integer/character vector of cell IDs matching the nb object
# rook_neighbors_unique: spdep nb object (list of integer index vectors)
# rf_model             : trained Random Forest model (untouched)

# ── 1. Convert cell_data to data.table (in-place if possible) ───────────────
setDT(cell_data)

# ── 2. Build edge list from the nb object ───────────────────────────────────
#    This is ~1.37M directed edges; very fast.
edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {

  nb_i <- rook_neighbors_unique[[i]]
  # spdep nb: integer(0) means no neighbors; 0L is also possible

  nb_i <- nb_i[nb_i > 0L]
  if (length(nb_i) == 0L) return(NULL)
  data.table(focal_cell = id_order[i], neighbor_cell = id_order[nb_i])
}))

cat("Edge list rows:", nrow(edges), "\n")

# ── 3. Add row-index to cell_data ──────────────────────────────────────────
cell_data[, .row_idx := .I]

# ── 4. Build a keyed lookup: (id, year) → row index ────────────────────────
id_year_key <- cell_data[, .(id, year, .row_idx)]
setkey(id_year_key, id, year)

# ── 5. Get the unique years present ────────────────────────────────────────
all_years <- sort(unique(cell_data$year))

# ── 6. Cross-join edges × years, then filter to existing cell-years ────────
#    To avoid a 38M-row cross join in one shot (memory), we process in
#    year-chunks. Each chunk is ~1.37M rows — trivially small.

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-extract the columns we need for neighbor stats into a matrix for speed
val_mat <- as.matrix(cell_data[, ..neighbor_source_vars])

# Prepare result columns (pre-allocate with NA)
stat_names <- c("max", "min", "mean")
new_col_names <- as.vector(outer(
  neighbor_source_vars, stat_names,
  function(v, s) paste0("neighbor_", s, "_", v)
))

# Pre-allocate result matrix: nrow(cell_data) × 15
result_mat <- matrix(NA_real_, nrow = nrow(cell_data), ncol = length(new_col_names))
colnames(result_mat) <- new_col_names

cat("Processing", length(all_years), "years...\n")

for (yr in all_years) {
  # Rows in this year
  yr_rows <- id_year_key[year == yr]  # columns: id, year, .row_idx
  setkey(yr_rows, id)

  # Join edges to get focal row index
  #   edges has (focal_cell, neighbor_cell)
  #   yr_rows has (id, year, .row_idx)
  focal_join <- yr_rows[edges, on = .(id = focal_cell), nomatch = 0L,
                        .(focal_row = .row_idx,
                          neighbor_cell = i.neighbor_cell)]

  # Join again to get neighbor row index
  setkey(focal_join, neighbor_cell)
  full_join <- yr_rows[focal_join, on = .(id = neighbor_cell), nomatch = 0L,
                       .(focal_row = i.focal_row,
                         neighbor_row = .row_idx)]

  if (nrow(full_join) == 0L) next

  # Extract neighbor values: matrix subsetting is very fast
  nb_vals <- val_mat[full_join$neighbor_row, , drop = FALSE]

  # Build a data.table for grouped aggregation
  agg_dt <- data.table(focal_row = full_join$focal_row, nb_vals)

  # Grouped aggregation: max, min, mean for each variable, grouped by focal_row
  # Build the aggregation expression dynamically
  agg_exprs <- lapply(neighbor_source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  })
  agg_exprs <- unlist(agg_exprs, recursive = FALSE)
  names(agg_exprs) <- new_col_names

  agg_call <- as.call(c(as.name("list"), agg_exprs))

  agg_result <- agg_dt[, eval(agg_call), by = focal_row]

  # Write results into the pre-allocated matrix
  result_mat[agg_result$focal_row, ] <- as.matrix(
    agg_result[, ..new_col_names]
  )

  cat("  Year", yr, "done —", nrow(agg_result), "focal cells with neighbors\n")
}

# ── 7. Handle Inf/-Inf from max/min on all-NA groups (shouldn't happen, but safe)
result_mat[is.infinite(result_mat)] <- NA_real_

# ── 8. Bind result columns to cell_data ─────────────────────────────────────
for (j in seq_along(new_col_names)) {
  set(cell_data, j = new_col_names[j], value = result_mat[, j])
}

# Clean up helper column
cell_data[, .row_idx := NULL]

cat("Done. Added", length(new_col_names), "neighbor-stat columns.\n")

# ── 9. Predict with the existing RF model (unchanged) ──────────────────────
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same neighbor relationships** | The edge list is built from the identical `rook_neighbors_unique` nb object and `id_order`. |
| **Same statistics** | `max`, `min`, `mean` with `na.rm = TRUE` on exactly the same neighbor sets, filtered to existing cell-years via inner join (`nomatch = 0L`). |
| **Same column names** | The naming convention `neighbor_{stat}_{var}` matches the original `compute_and_add_neighbor_features` output. (Adjust the `paste0` template if your original used a different convention.) |
| **RF model untouched** | No retraining; the model object is never modified. |

## Expected Performance

| Step | Estimated Time |
|---|---|
| Edge list construction (1.37M edges) | ~2 seconds |
| Per-year join + aggregation (28 iterations × ~1.37M edges) | ~1–3 minutes total |
| Column binding | < 1 second |
| **Total** | **~2–5 minutes** (vs. 86+ hours) |

Peak memory: ~2–3 GB (the 6.46M × 110 data.table + the edge list + one year's aggregation buffer), well within 16 GB.