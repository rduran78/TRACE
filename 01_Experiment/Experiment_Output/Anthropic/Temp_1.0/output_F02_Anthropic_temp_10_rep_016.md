 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The performance bottleneck has two root causes:

### 1. `build_neighbor_lookup` — O(n) `lapply` over 6.46 million rows with per-row string pasting and named-vector lookups

- **String key construction**: `paste(id, year, sep="_")` is called millions of times inside the loop body (once per neighbor per row), producing enormous intermediate character vectors.
- **Named-vector lookup (`idx_lookup[neighbor_keys]`)**: Named vector indexing in R uses hashing, but creating and resolving ~6.46M × avg-neighbors character keys is extremely slow and memory-hungry.
- **`lapply` over 6.46M elements**: Each closure allocates small vectors; the cumulative overhead is massive.

### 2. `compute_neighbor_stats` — repeated `lapply` over 6.46M ragged index lists

- Called once per variable (×5), each time iterating 6.46M list elements.
- `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors is a known R anti-pattern (quadratic copy risk in older R; still slow).

### Memory pressure

- The `neighbor_lookup` list alone: 6.46M list elements, each a small integer vector → hundreds of MB of list overhead (each list element has a 56-byte SEXP header on 64-bit R, so ~360 MB just in headers, plus the integer data).
- Holding all of `cell_data` (~6.46M × 110 columns) simultaneously adds several more GB.

**Net effect**: the combination of character-key gymnastics, per-row R-level iteration, and ragged list structures yields the estimated 86+ hour runtime.

---

## Optimization Strategy

| Technique | What it fixes | Expected speedup |
|---|---|---|
| **Replace the ragged list with a flat edge table (data.table)** | Eliminates per-row `lapply`, string keys, and named-vector lookups | 50–200× for `build_neighbor_lookup` |
| **Vectorised join + grouped aggregation (`data.table`)** | Replaces `lapply` in `compute_neighbor_stats` with a single keyed join and `by=` aggregation | 20–100× per variable |
| **Process all 5 variables in one pass** | Avoids 5 separate full-table scans | ~5× for the stats phase |
| **Integer keys throughout** | Avoid all `paste`/character work; use composite integer keys or multi-column keys | Large constant-factor improvement |
| **In-place `:=` assignment** | Avoid repeated shallow-copy of the growing `cell_data` frame | Reduces memory spikes |

The strategy preserves every numerical output (max, min, mean of rook neighbors per variable per cell-year) and never touches the trained Random Forest model.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 0.  Ensure cell_data is a data.table with an integer row-id column
# ──────────────────────────────────────────────────────────────────────
if (!is.data.table(cell_data)) setDT(cell_data)

# Ensure 'id' and 'year' are integer (avoids implicit coercion later)
cell_data[, id   := as.integer(id)]
cell_data[, year := as.integer(year)]

# ──────────────────────────────────────────────────────────────────────
# 1.  Build a flat edge table from the nb object (once)
#
#     rook_neighbors_unique is a list of length N_cells (344,208).
#     id_order is the vector that maps list-position → cell id.
#     We explode it into a two-column data.table: (id, neighbor_id).
# ──────────────────────────────────────────────────────────────────────
build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate vectors
  n_edges <- sum(lengths(neighbors))
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)

  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_i <- neighbors[[i]]
    if (length(nb_i) == 0L || (length(nb_i) == 1L && nb_i[1] == 0L)) next
    n_i <- length(nb_i)
    from_id[pos:(pos + n_i - 1L)] <- id_order[i]
    to_id[pos:(pos + n_i - 1L)]   <- id_order[nb_i]
    pos <- pos + n_i
  }
  # Trim if any nb entries were empty / zero-neighbor islands
  data.table(id = from_id[1:(pos - 1L)],
             neighbor_id = to_id[1:(pos - 1L)])
}

cat("Building edge table …\n")
edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# edge_dt now has ~1.37 M rows (directed edges), all integer columns.

# ──────────────────────────────────────────────────────────────────────
# 2.  Vectorised neighbor-stat computation
#
#     For every (id, year) pair and every variable, we need:
#       neighbor_max, neighbor_min, neighbor_mean
#     across all rook neighbors present in the data for that year.
#
#     Approach:
#       a) Join edge_dt onto cell_data to get (id, year, neighbor_id).
#       b) Join again to pull the neighbor's variable values.
#       c) Group-by (id, year) and compute max/min/mean.
#     All of this is one pass per variable (or batched).
# ──────────────────────────────────────────────────────────────────────

# Key cell_data for fast joins
setkey(cell_data, id, year)

# We need a lookup from (neighbor_id, year) → variable values.
# Build a small reference table with only the columns we need.
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Subset only what's needed for the neighbor lookup
ref_cols <- c("id", "year", neighbor_source_vars)
ref_dt   <- cell_data[, ..ref_cols]
setnames(ref_dt, "id", "neighbor_id")
setkey(ref_dt, neighbor_id, year)

# ──────────────────────────────────────────────────────────────────────
# 3.  For each variable, join → aggregate → assign back in-place
# ──────────────────────────────────────────────────────────────────────
cat("Computing neighbor features …\n")

for (var in neighbor_source_vars) {

  cat("  →", var, "\n")

  # Columns we need from the reference table for this variable
  ref_sub <- ref_dt[, .(neighbor_id, year, val = get(var))]
  setkey(ref_sub, neighbor_id, year)

  # Step A: expand cell_data rows by their neighbors
  #   Start from cell_data's (id, year), join to edge_dt to get neighbor_id,
  #   then join to ref_sub to get the neighbor's value.
  #
  #   To avoid materialising the full 6.46M × avg_neighbors table in memory
  #   we process year-by-year (28 chunks ≈ 230K × 4 neighbors each).

  stats_list <- vector("list", length(unique(cell_data$year)))
  years <- sort(unique(cell_data$year))

  for (yi in seq_along(years)) {
    yr <- years[yi]

    # Rows for this year: their ids
    ids_yr <- cell_data[year == yr, .(id)]

    # Attach neighbor ids via edge table
    #   ids_yr  join  edge_dt  on id  →  (id, neighbor_id)
    expanded <- edge_dt[ids_yr, on = "id", nomatch = NULL, allow.cartesian = TRUE]
    # expanded has columns: id, neighbor_id

    # Attach neighbor values
    expanded[, year := yr]
    expanded <- ref_sub[expanded, on = .(neighbor_id, year), nomatch = NA]
    # Now expanded has: neighbor_id, year, val, id

    # Aggregate
    stats_yr <- expanded[!is.na(val),
                         .(nb_max  = max(val),
                           nb_min  = min(val),
                           nb_mean = mean(val)),
                         by = .(id)]
    stats_yr[, year := yr]

    stats_list[[yi]] <- stats_yr
  }

  stats_all <- rbindlist(stats_list)
  setkey(stats_all, id, year)

  # Assign columns back into cell_data in-place
  max_col  <- paste0("neighbor_max_",  var)
  min_col  <- paste0("neighbor_min_",  var)
  mean_col <- paste0("neighbor_mean_", var)

  # Initialize with NA
  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]

  # Update join — only touches rows that have neighbor data
  cell_data[stats_all,
            `:=`(
              (max_col)  = i.nb_max,
              (min_col)  = i.nb_min,
              (mean_col) = i.nb_mean
            ),
            on = .(id, year)]

  # Free intermediate objects
  rm(ref_sub, stats_list, stats_all, expanded, stats_yr, ids_yr)
  gc()
}

cat("Done. Neighbor features added to cell_data.\n")
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max()`, `min()`, `mean()` are applied to exactly the same set of non-NA rook-neighbor values per (id, year) as the original code. The join path `cell_data → edge_dt → ref_dt` reproduces the same directed neighbor mapping that `build_neighbor_lookup` constructed via `id_to_ref` and `idx_lookup`. |
| **Trained Random Forest** | The model object is never referenced or modified. Only `cell_data` gains 15 new columns (3 stats × 5 variables) whose names match the originals. |
| **Column names** | Naming convention `neighbor_max_<var>`, `neighbor_min_<var>`, `neighbor_mean_<var>` matches the output of the presumed `compute_and_add_neighbor_features` wrapper. Adjust the prefix if your wrapper uses a different convention. |

---

## Expected Performance

| Phase | Original | Optimised | Notes |
|---|---|---|---|
| `build_neighbor_lookup` (→ `build_edge_table`) | ~hours (6.46M `paste` + named lookups) | **< 5 seconds** | Simple integer pre-allocation loop; runs once |
| `compute_neighbor_stats` per variable | ~15+ hours (6.46M `lapply`) | **~1–3 minutes** | `data.table` keyed join + `by=` aggregation, processed in 28 year-chunks to cap memory |
| **Total for 5 variables** | **86+ hours** | **~5–15 minutes** | Depends on disk I/O and available RAM |
| **Peak RAM** | High (ragged list + growing copies) | **~4–6 GB** | Year-chunking keeps the expanded join table small; `:=` avoids copies |

The year-by-year loop (28 iterations) is the key memory-control device: each chunk expands ~230K cells × ~4 neighbors = ~920K rows, trivially small for `data.table`. If your machine has headroom you can remove the year loop and process all years at once for a further ~2× speedup.