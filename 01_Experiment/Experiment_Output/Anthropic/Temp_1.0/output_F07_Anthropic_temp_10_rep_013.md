 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is in `build_neighbor_lookup`. It creates **~6.46 million individual list elements**, each constructed inside an `lapply` call that performs per-row character coercion, string pasting, and named-vector lookups. Specifically:

1. **`build_neighbor_lookup` is O(N·k) with enormous constant factors.** For each of the 6.46M rows, it converts IDs to characters, pastes strings, and does named-vector indexing (`idx_lookup[neighbor_keys]`). Named-vector lookup in R is hash-based but carries substantial per-call overhead when done millions of times in an interpreted loop.

2. **The lookup is built row-by-row in pure R.** `lapply` over 6.46M rows with string operations inside is catastrophically slow. The estimated 86+ hours is almost entirely attributable to this function.

3. **`compute_neighbor_stats` is also row-by-row** but is cheaper per iteration (just numeric subsetting). Still, iterating 6.46M times × 5 variables = 32.3M iterations of interpreted R is unnecessarily slow.

4. **Memory layout is list-of-vectors**, which has high overhead for millions of small vectors.

The core insight: the neighbor structure is **time-invariant** (the same spatial neighbors apply to every year). We should exploit this by doing a **merge/join** operation at the spatial level, then vectorizing the statistics computation across all cell-years using `data.table`.

---

## Optimization Strategy

1. **Replace the per-row lookup with a `data.table` equi-join.** Build an edge table (cell_id → neighbor_id) from the `nb` object, then join `cell_data` to itself on `(neighbor_id, year)` to retrieve neighbor values. This is a single vectorized join — no interpreted loop at all.

2. **Compute grouped statistics with `data.table` aggregation.** After the join, group by `(id, year)` and compute `max`, `min`, `mean` in one pass per variable. `data.table`'s GForce will optimize these to C-level operations.

3. **Loop only over the 5 source variables**, not over rows. Each variable requires one join + one grouped aggregation — about 5 passes total.

4. **Memory management:** The edge table has ~1.37M edges. The joined table (edges × years) has ~1.37M × 28 ≈ 38.5M rows, each with just (id, year, value). At ~3 columns of 8 bytes each, that's ~0.9 GB — well within 16 GB RAM. We process one variable at a time and discard intermediates.

**Expected speedup:** From 86+ hours to **~2–10 minutes** (roughly 500–2500× faster).

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# Step 1: Build a spatial edge table from the nb object (once)
# ---------------------------------------------------------------
build_edge_table <- function(id_order, nb_obj) {
  # nb_obj is a list of integer vectors (indices into id_order)
  # Expand into a two-column data.table: (id, neighbor_id)
  from_idx <- rep(seq_along(nb_obj), lengths(nb_obj))
  to_idx   <- unlist(nb_obj)
  
  # Remove zero-neighbor entries (spdep uses integer(0) for islands)
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]
  
  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# edge_dt has ~1,373,394 rows: one per directed rook-neighbor pair

# ---------------------------------------------------------------
# Step 2: Convert cell_data to data.table (if not already)
# ---------------------------------------------------------------
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure keyed for fast joins
setkey(cell_data, id, year)

# ---------------------------------------------------------------
# Step 3: For each source variable, compute neighbor stats via join
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_neighbor_features_fast <- function(cell_data, edge_dt, var_name) {
  # Subset to only the columns we need for the join
  # cell_data must have columns: id, year, <var_name>
  val_dt <- cell_data[, .(id, year, value = get(var_name))]
  setkey(val_dt, id, year)
  
  # Join edges with values: for each (id, year), find neighbor values
  # edge_dt has (id, neighbor_id); we join neighbor_id -> val_dt$id
  # to get the neighbor's value in the same year
  joined <- merge(
    edge_dt,
    val_dt,
    by.x = "neighbor_id",
    by.y = "id",
    allow.cartesian = TRUE  # each neighbor appears in 28 years
  )
  # joined now has columns: neighbor_id, id, year, value
  # where 'value' is the NEIGHBOR's value of var_name in that year
  # and 'id' is the focal cell
  
  # Compute grouped stats: group by (id, year)
  stats <- joined[
    !is.na(value),
    .(
      nmax  = max(value),
      nmin  = min(value),
      nmean = mean(value)
    ),
    keyby = .(id, year)
  ]
  
  # Name the output columns
  suffix <- var_name
  setnames(stats, c("nmax", "nmin", "nmean"),
           paste0(c("nmax_", "nmin_", "nmean_"), suffix))
  
  return(stats)
}

# ---------------------------------------------------------------
# Step 4: Loop over variables, merge results back into cell_data
# ---------------------------------------------------------------
for (var_name in neighbor_source_vars) {
  cat("Computing neighbor stats for:", var_name, "\n")
  
  # Remove old columns if re-running (idempotency)
  old_cols <- paste0(c("nmax_", "nmin_", "nmean_"), var_name)
  old_cols_present <- intersect(old_cols, names(cell_data))
  if (length(old_cols_present) > 0) {
    cell_data[, (old_cols_present) := NULL]
  }
  
  stats_dt <- compute_neighbor_features_fast(cell_data, edge_dt, var_name)
  
  # Left join back to cell_data (preserves rows with no neighbors as NA)
  cell_data <- merge(cell_data, stats_dt, by = c("id", "year"), all.x = TRUE)
  
  rm(stats_dt)
  gc()
}

# Verify
cat("Final dimensions:", nrow(cell_data), "x", ncol(cell_data), "\n")
cat("Neighbor feature columns added:\n")
print(grep("^(nmax_|nmin_|nmean_)", names(cell_data), value = TRUE))

# ---------------------------------------------------------------
# cell_data now has the same neighbor max/min/mean columns as the
# original implementation, computed identically (same numerical
# estimand), ready for prediction with the existing trained
# Random Forest model — no retraining required.
# ---------------------------------------------------------------
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same neighbor relationships** | `edge_dt` is built directly from the same `rook_neighbors_unique` nb object and `id_order` — identical directed pairs. |
| **Same statistics** | `max`, `min`, `mean` are computed on exactly the same set of non-NA neighbor values, grouped by `(id, year)`. |
| **Same NA behavior** | Cells with no neighbors (or all-NA neighbors) get `NA` via the `all.x = TRUE` left join, matching the original `c(NA, NA, NA)` return. |
| **Trained RF model untouched** | Only the feature-engineering pipeline is replaced. Column names and values are identical, so the existing model's `predict()` call works unchanged. |

---

## Expected Performance

| Step | Time estimate (laptop, 16 GB) |
|---|---|
| `build_edge_table` | < 1 second |
| One variable: join + aggregate | ~20–60 seconds |
| All 5 variables | **~2–5 minutes total** |
| **Original implementation** | **86+ hours** |

The speedup comes from replacing ~32 million interpreted R loop iterations with 5 vectorized `data.table` merge + group-by operations executed in C.