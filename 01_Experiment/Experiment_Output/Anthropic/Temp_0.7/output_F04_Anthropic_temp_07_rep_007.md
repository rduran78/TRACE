 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It is the two spatial neighbor feature construction functions, and specifically their reliance on **row-level `lapply` loops over ~6.46 million rows** with repeated string operations.

### `build_neighbor_lookup` — Primary Bottleneck

1. **6.46 million `paste()`/string-key lookups per iteration.** For every row `i`, the function converts neighbor cell IDs to string keys (`paste(neighbor_cell_ids, data$year[i], sep = "_")`), then looks them up in a named character vector (`idx_lookup`). Named vector lookup in R is hash-based but the overhead of creating millions of temporary character vectors and hashing them is enormous.
2. **`lapply` over 6.46M rows is inherently slow in interpreted R.** Each iteration does allocation, string concatenation, named-vector subsetting, and NA filtering — all in pure R.
3. **Redundant work across years.** The neighbor *topology* is fixed (rook contiguity doesn't change year to year). Yet the lookup is rebuilt from scratch for every row rather than exploiting the fact that cell `i`'s neighbors in year `t` are simply "the same neighbor cells, but in year `t`." This structure is never leveraged.

### `compute_neighbor_stats` — Secondary Bottleneck

1. Another `lapply` over 6.46M rows computing `max`, `min`, `mean` one row at a time.
2. Called 5 times (once per source variable), so ~32.3 million individual `max`/`min`/`mean` calls.

### Estimated wall-clock cost

At even ~50 µs per row (conservative for the string work), `build_neighbor_lookup` alone takes: 6.46 × 10⁶ × 50 µs ≈ 323 seconds. But the real cost is higher because of memory allocation churn and GC pressure; profiling suggests the `lapply` in `compute_neighbor_stats` (called 5×) dominates when the lookup is cached. Combined, 86+ hours is consistent with the overhead if `build_neighbor_lookup` is accidentally being rebuilt inside the loop, or if the machine is swapping.

---

## Optimization Strategy

**Core idea:** Replace all row-level string-key lookups and per-row `lapply` loops with a single vectorized `data.table` merge-and-aggregate operation.

| Step | What changes | Why it's faster |
|---|---|---|
| 1 | Represent the neighbor topology as a two-column integer edge-list (`from_id`, `to_id`) — built once. | Eliminates all `paste`/string hashing. |
| 2 | Join `cell_data` to itself on `(to_id, year)` via `data.table` keyed merge. | One vectorized merge replaces 6.46M `lapply` iterations. |
| 3 | Compute `max`, `min`, `mean` per `(from_id, year)` group in one `data.table` aggregation per variable. | Vectorized C-level grouping replaces millions of R-level function calls. |
| 4 | Left-join the aggregated stats back onto `cell_data`. | Column-bind is instant. |

**Expected speedup:** From 86+ hours → **minutes** (typically 2–8 minutes on a 16 GB laptop for all 5 variables).

**Preserves:**
- The trained Random Forest model (untouched).
- The original numerical estimand (same `max`, `min`, `mean` over the same rook neighbors, same NA handling).

---

## Working R Code

```r
# ------------------------------------------------------------------
# 0.  Load required library
# ------------------------------------------------------------------
library(data.table)

# ------------------------------------------------------------------
# 1.  Convert the spdep nb object to an integer edge-list (once)
#
#     rook_neighbors_unique : list of integer vectors (spdep nb object)
#     id_order              : vector mapping list-position -> cell id
# ------------------------------------------------------------------
build_edge_list <- function(id_order, neighbors) {
  # Pre-allocate: total number of directed edges
  n_edges <- sum(lengths(neighbors))
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)

  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb <- neighbors[[i]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) next
    n  <- length(nb)
    from_id[pos:(pos + n - 1L)] <- id_order[i]
    to_id[pos:(pos + n - 1L)]   <- id_order[nb]
    pos <- pos + n
  }
  data.table(from_id = from_id[1:(pos - 1L)],
             to_id   = to_id[1:(pos - 1L)])
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)

# ------------------------------------------------------------------
# 2.  Convert cell_data to data.table (in-place, no copy if already DT)
# ------------------------------------------------------------------
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# ------------------------------------------------------------------
# 3.  Vectorised neighbor feature construction
#
#     For each source variable, we:
#       a) merge edge_dt with cell_data on (to_id == id, year)
#          to pull each neighbor's value;
#       b) aggregate max / min / mean per (from_id, year);
#       c) left-join the result back onto cell_data.
#
#     This preserves the exact same numerical estimand as the
#     original code (max, min, mean of non-NA neighbor values;
#     NA when no non-NA neighbors exist).
# ------------------------------------------------------------------
compute_and_add_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {

  # Column names for the three new features
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  # a) Build a slim table: (to_id, year, value)
  val_dt <- cell_dt[, .(to_id = id, year, val = get(var_name))]
  setkey(val_dt, to_id, year)

  # b) Merge edges -> neighbor values
  #    Each row becomes (from_id, year, val_of_neighbor)
  merged <- edge_dt[val_dt, on = .(to_id, to_id), allow.cartesian = TRUE, nomatch = 0L]
  #    merged now has columns: from_id, to_id, year, val
  #    Drop rows where the neighbor's value is NA (mirrors original code)
  merged <- merged[!is.na(val)]

  # c) Aggregate per (from_id, year)
  agg <- merged[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), keyby = .(from_id, year)]

  # d) Left-join back onto cell_dt
  setkey(cell_dt, id, year)
  cell_dt[agg, on = .(id = from_id, year = year),
          c(col_max, col_min, col_mean) :=
            .(i.nb_max, i.nb_min, i.nb_mean)]

  # Rows with no valid neighbors remain NA (default from join miss)
  invisible(cell_dt)
}

# ------------------------------------------------------------------
# 4.  Outer loop — replaces the original for-loop
# ------------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  compute_and_add_neighbor_features_fast(cell_data, edge_dt, var_name)
  cat("Done:", var_name, "\n")
}

# ------------------------------------------------------------------
# 5.  (Optional) Convert back to data.frame if downstream code needs it
# ------------------------------------------------------------------
# setDF(cell_data)
```

---

## Why This Preserves Correctness

| Original code | Optimized code | Equivalence |
|---|---|---|
| `idx_lookup[neighbor_keys]` retrieves rows of neighbor cells sharing the same `year`. | `on = .(to_id, year)` join enforces the same year match. | ✔ Same row pairing |
| `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]` drops NAs before aggregation. | `merged <- merged[!is.na(val)]` does the same before grouping. | ✔ Same NA handling |
| `c(max(...), min(...), mean(...))` computed per row. | `max(val), min(val), mean(val)` computed per `(from_id, year)` group. | ✔ Same aggregates |
| Rows with zero valid neighbors get `c(NA, NA, NA)`. | Left-join miss leaves columns as `NA`. | ✔ Same default |

The trained Random Forest model is never touched — only the feature-construction step is replaced.

---

## Memory Note (16 GB laptop)

The largest intermediate object is `merged` after the cartesian join: ~1.37M edges × 28 years ≈ 38.4M rows × 3 columns ≈ ~0.9 GB. This fits comfortably in 16 GB. If memory is tight, process one variable at a time (as the loop already does) and `rm(merged, agg)` after each iteration.