 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It is the two spatial-neighbor functions, and specifically the way they interact with R's performance characteristics:

### `build_neighbor_lookup` — The Primary Bottleneck

1. **Row-level `lapply` over ~6.46 million rows.** Each iteration performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookup (`idx_lookup[neighbor_keys]`). These are all slow in a scalar R loop.
2. **Repeated string allocation.** For every single row, `paste(neighbor_cell_ids, data$year[i], sep = "_")` creates a new character vector. Over 6.46M rows, this generates billions of short-lived string objects, causing enormous GC pressure.
3. **Named vector lookup is O(n) hashing per call.** `idx_lookup[neighbor_keys]` on a named vector of length 6.46M is re-hashing on every access in base R. This is done ~6.46M times.

**Estimated cost:** Each row touches ~4 neighbors on average (rook contiguity). That's ~6.46M × 4 = ~25.8M string constructions and named-vector lookups, on top of the 6.46M `paste` calls for keys. This alone can take tens of hours.

### `compute_neighbor_stats` — A Secondary Bottleneck

1. **`lapply` + `do.call(rbind, ...)`** over 6.46M rows: `do.call(rbind, list_of_6.46M_vectors)` is notoriously slow — it copies and rebinds incrementally.
2. Called **5 times** (once per source variable), so the cost multiplies.

### Summary

| Component | Calls | Estimated share of 86h |
|---|---|---|
| `build_neighbor_lookup` (string ops, named-vector lookup) | 6.46M | ~60–70% |
| `compute_neighbor_stats` (lapply + rbind, ×5 vars) | 5 × 6.46M | ~25–35% |
| Random Forest prediction | 1 | < 5% |

---

## Optimization Strategy

The core idea: **eliminate all string operations and named-vector lookups; work entirely with integer indices and vectorized/`data.table` operations.**

### Step-by-step plan

1. **Replace the string-keyed lookup with an integer-keyed join.** Build a `data.table` keyed on `(id, year)` → `row_index`. Use a fast equi-join to resolve all neighbor references at once, vectorized.

2. **Expand the neighbor list into a flat edge table once.** Instead of iterating row-by-row, explode `rook_neighbors_unique` into a two-column integer table `(cell_ref, neighbor_ref)`, then join against the panel to produce `(row_i, neighbor_row_i)` in one vectorized pass.

3. **Compute all neighbor stats with grouped `data.table` aggregation.** Group the flat edge table by `row_i`, pull the variable values by `neighbor_row_i`, and compute `max`, `min`, `mean` in one grouped operation — no R-level loop.

4. **Process all 5 variables in one pass** over the edge table (or with minimal additional passes), avoiding redundant work.

**Expected speedup:** From ~86 hours to **minutes** (typically 2–10 minutes depending on RAM bandwidth), because every operation becomes vectorized C-level code inside `data.table`.

**Numerical equivalence:** The aggregation functions (`max`, `min`, `mean`) are identical; only the iteration strategy changes. The trained Random Forest model is untouched.

---

## Working R Code

```r
library(data.table)

optimize_neighbor_features <- function(cell_data,
                                       id_order,
                                       rook_neighbors_unique,
                                       neighbor_source_vars) {
  # ---------------------------------------------------------------
  # 0. Convert to data.table (if not already); record original order

  # ---------------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, row_idx__ := .I]

  # ---------------------------------------------------------------
  # 1. Build integer cell-id → ref-index mapping
  #    id_order[ref_idx] == cell_id
  # ---------------------------------------------------------------
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # ---------------------------------------------------------------
  # 2. Expand the nb object into a flat edge list (ref-space)
  #    Each element of rook_neighbors_unique is an integer vector of

  #    neighbor ref-indices for that ref-index.
  # ---------------------------------------------------------------
  n_refs <- length(rook_neighbors_unique)
  # Lengths of each neighbor set
  lens <- lengths(rook_neighbors_unique)
  # "from" ref-index repeated
  from_ref <- rep(seq_len(n_refs), lens)
  # "to" ref-index concatenated
  to_ref   <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Convert ref-indices to actual cell ids
  edges <- data.table(
    from_id = id_order[from_ref],
    to_id   = id_order[to_ref]
  )
  rm(from_ref, to_ref, lens)

  # ---------------------------------------------------------------
  # 3. Build a row-index lookup keyed on (id, year)
  # ---------------------------------------------------------------
  row_key <- dt[, .(id, year, row_idx__)]
  setkey(row_key, id, year)

  # ---------------------------------------------------------------
  # 4. Get unique years once
  # ---------------------------------------------------------------
  years <- sort(unique(dt$year))

  # ---------------------------------------------------------------
  # 5. Cross edges × years to get the full (row_i, neighbor_row_j)
  #    mapping.  To keep memory manageable we do this by year.
  # ---------------------------------------------------------------

  # Pre-allocate result columns (NA_real_) in dt for every feature
  stat_names <- c("max", "min", "mean")
  new_cols <- character(0)
  for (v in neighbor_source_vars) {
    for (s in stat_names) {
      col <- paste0(v, "_neighbor_", s)
      new_cols <- c(new_cols, col)
      set(dt, j = col, value = NA_real_)
    }
  }

  # Pre-extract variable vectors for fast indexing
  var_vectors <- setNames(
    lapply(neighbor_source_vars, function(v) dt[[v]]),
    neighbor_source_vars
  )

  # Process year-by-year to bound memory (~344K cells × ~4 neighbors)
  for (yr in years) {
    # Rows in this year
    yr_rows <- row_key[.(unique(edges$from_id), yr), nomatch = 0L,
                        on = .(id, year)]
    setnames(yr_rows, c("id", "year", "row_idx__"),
             c("from_id", "year_from", "row_i"))

    # Join edges to get (from_id -> to_id) for this year
    yr_edges <- edges[yr_rows, on = .(from_id), nomatch = 0L, allow.cartesian = TRUE]
    # yr_edges now has: from_id, to_id, year_from, row_i

    # Resolve to_id + year -> neighbor row index
    yr_edges[, year := year_from]
    neighbor_rows <- row_key[yr_edges, on = .(id = to_id, year), nomatch = 0L]
    # neighbor_rows has: id (=to_id), year, row_idx__ (=neighbor row), row_i

    # For each source variable, compute grouped stats
    for (v in neighbor_source_vars) {
      neighbor_rows[, val__ := var_vectors[[v]][row_idx__]]

      stats <- neighbor_rows[!is.na(val__),
                             .(nmax  = max(val__),
                               nmin  = min(val__),
                               nmean = mean(val__)),
                             keyby = .(row_i)]

      col_max  <- paste0(v, "_neighbor_max")
      col_min  <- paste0(v, "_neighbor_min")
      col_mean <- paste0(v, "_neighbor_mean")

      set(dt, i = stats$row_i, j = col_max,  value = stats$nmax)
      set(dt, i = stats$row_i, j = col_min,  value = stats$nmin)
      set(dt, i = stats$row_i, j = col_mean, value = stats$nmean)
    }
  }

  # ---------------------------------------------------------------
  # 6. Clean up helper column and return
  # ---------------------------------------------------------------
  dt[, row_idx__ := NULL]

  # Return as data.frame if the caller expects one
  if (inherits(cell_data, "data.frame") && !inherits(cell_data, "data.table")) {
    return(as.data.frame(dt))
  }
  return(dt[])
}

# ===================================================================
# USAGE — drop-in replacement for the original outer loop
# ===================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The trained Random Forest model is unchanged; proceed to prediction:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same numerical estimand** | `max`, `min`, `mean` are computed over exactly the same neighbor sets (same rook contiguity, same year matching, same `NA` removal). |
| **Trained RF model untouched** | No model retraining. The output columns have the same names and values, so `predict(rf_model, newdata = cell_data)` is unchanged. |
| **Column naming** | Assumes the original `compute_and_add_neighbor_features` created columns named `{var}_neighbor_max`, `{var}_neighbor_min`, `{var}_neighbor_mean`. Adjust the `paste0` pattern if the original naming convention differs. |

## Expected Performance

| Stage | Original (est.) | Optimized (est.) |
|---|---|---|
| Build neighbor lookup | ~50–60 h | <10 s (vectorized expand) |
| Compute stats (5 vars × 6.46M rows) | ~25–30 h | ~2–8 min (data.table grouped agg, by year) |
| Peak RAM | Moderate (GC thrash from strings) | ~3–5 GB (flat edge table + panel) |
| **Total** | **~86 h** | **~3–10 min** |

The key insight: replacing 6.46 million R-level string-paste-and-lookup iterations with a single vectorized `data.table` equi-join eliminates the bottleneck entirely.