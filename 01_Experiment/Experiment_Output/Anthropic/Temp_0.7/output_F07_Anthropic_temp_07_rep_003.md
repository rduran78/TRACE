 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It constructs a list of 6.46 million elements, where each element is built by:

1. Looking up the focal cell's neighbor IDs via `id_to_ref` and `neighbors[[ref_idx]]` — this is fast.
2. Pasting together `neighbor_cell_ids` and `data$year[i]` into character keys, then matching them into `idx_lookup` — **this is extremely slow**.

The `paste(…, sep="_")` call inside a per-row `lapply` over ~6.46 million rows, each touching ~4 neighbors on average (~1,373,394 directed edges / 344,208 cells ≈ 4), means roughly **25+ million `paste` and hash-lookup operations executed one-at-a-time in interpreted R**. The named-vector lookup `idx_lookup[neighbor_keys]` is also O(n) per probe in the worst case for large named vectors.

Then `compute_neighbor_stats` iterates over the 6.46M-element list again, once **per variable** (×5 variables). Each call extracts neighbor values, removes NAs, and computes max/min/mean. This is another ~32 million interpreted iterations total.

**Summary of costs:**

| Step | Calls | Estimated wall-clock |
|---|---|---|
| `build_neighbor_lookup` (paste + named-vector match, 6.46M rows) | 1 | ~40–60 hours |
| `compute_neighbor_stats` (lapply, 6.46M × 5 vars) | 5 | ~25–30 hours |
| **Total** | | **~65–90 hours** |

---

## Optimization Strategy

### Key insight: Separate the spatial graph from the temporal dimension

Every cell has the **same** neighbors in every year. So the neighbor lookup can be decomposed into:

1. **A spatial edge list** (344K cells × ~4 neighbors = ~1.37M edges) — tiny.
2. **A year index** — 28 distinct years.

We never need to build a 6.46M-element list. Instead:

### Step 1: Build a sparse adjacency structure once (spatial only)

Convert `rook_neighbors_unique` (an `nb` object) into a two-column edge list (`from_id`, `to_id`) of cell IDs. This is ~1.37M rows — trivially small.

### Step 2: Vectorized join to expand edges across years

Use `data.table` to join the edge list against the panel on `(neighbor_id, year)`, pulling in neighbor values. This is a single equi-join of ~1.37M × 28 ≈ ~38.4M rows — fast and memory-safe (~2–3 GB peak).

### Step 3: Grouped aggregation

Group by `(focal_row, variable)` and compute `max`, `min`, `mean` in one vectorized `data.table` operation.

### Expected performance

| Step | Method | Estimated time |
|---|---|---|
| Build edge list from nb | Vectorized | < 1 sec |
| Expand edges × years + join | `data.table` equi-join | ~10–30 sec |
| Grouped max/min/mean × 5 vars | `data.table` grouped agg | ~20–60 sec |
| **Total** | | **< 2 minutes** |

This is a **~3,000× speedup** while preserving the exact same numerical output (max, min, mean of non-NA rook-neighbor values per cell-year).

---

## Working R Code

```r
# =============================================================================
# Fast neighbor-stats computation — exact drop-in replacement
# =============================================================================
# Requirements: data.table
# Preserves: trained Random Forest model (untouched), original numerical output
# =============================================================================

library(data.table)

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {
  # ------------------------------------------------------------------
  # 0.  Convert cell_data to data.table (by reference if already is one)
  # ------------------------------------------------------------------
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  # Ensure a deterministic row identifier so we can write results back
  cell_data[, .row_idx := .I]

  # ------------------------------------------------------------------
  # 1.  Build spatial edge list from the nb object  (~1.37 M rows)
  #     nb object: list of integer vectors; indices into id_order
  # ------------------------------------------------------------------
  from_ref <- rep(
    seq_along(rook_neighbors_unique),
    lengths(rook_neighbors_unique)
  )
  to_ref <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Remove the spdep convention where 0L means "no neighbors"
  valid <- to_ref != 0L
  from_ref <- from_ref[valid]
  to_ref   <- to_ref[valid]

  edges <- data.table(
    focal_id    = id_order[from_ref],
    neighbor_id = id_order[to_ref]
  )

  # ------------------------------------------------------------------
  # 2.  For each source variable, join, aggregate, and attach columns
  # ------------------------------------------------------------------
  # We need a keyed version of cell_data for fast joins
  # Key columns: id (cell identifier) and year
  id_col  <- "id"
  yr_col  <- "year"

  # Minimal neighbor table: just the neighbor's id, year, and value
  # We join edges to cell_data twice:
  #   - once to get the focal cell's row index (so we know where to write)
  #   - once to get the neighbor cell's value

  # Pre-build a lookup: (id, year) -> .row_idx   [for focal side]
  focal_key <- cell_data[, .(id, year, .row_idx)]
  setkey(focal_key, id, year)

  for (var_name in neighbor_source_vars) {
    message("Processing neighbor stats for: ", var_name)

    # Neighbor-side lookup: (id, year) -> value
    neighbor_vals <- cell_data[, .SD, .SDcols = c(id_col, yr_col, var_name)]
    setnames(neighbor_vals, c("neighbor_id", "year", "nval"))
    setkey(neighbor_vals, neighbor_id, year)

    # Expand edges across all years present for the focal cell
    # focal_key gives us every (focal_id, year) with its row index
    # edges gives us every (focal_id -> neighbor_id)
    # We need: (focal_id, year, neighbor_id) then look up neighbor value

    # Step A: join focal_key with edges on focal_id
    #   Result: (focal_id, year, .row_idx, neighbor_id)
    setkey(edges, focal_id)
    expanded <- edges[focal_key, on = .(focal_id = id),
                      .(focal_id, year, .row_idx, neighbor_id),
                      allow.cartesian = TRUE, nomatch = NULL]
    # This is ~1.37M edges × 28 years ≈ 38.4M rows

    # Step B: join with neighbor_vals to get the actual value
    setkey(expanded, neighbor_id, year)
    expanded <- neighbor_vals[expanded, on = .(neighbor_id, year),
                              .(focal_id, year, .row_idx, nval = x.nval),
                              nomatch = NA]

    # Step C: aggregate — drop NAs, compute max/min/mean per focal row
    stats <- expanded[!is.na(nval),
                      .(nmax  = max(nval),
                        nmin  = min(nval),
                        nmean = mean(nval)),
                      by = .row_idx]

    # Step D: write results back into cell_data
    max_col  <- paste0("neighbor_max_",  var_name)
    min_col  <- paste0("neighbor_min_",  var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    # Initialize with NA
    cell_data[, (max_col)  := NA_real_]
    cell_data[, (min_col)  := NA_real_]
    cell_data[, (mean_col) := NA_real_]

    # Fill matched rows
    cell_data[stats, on = .(.row_idx),
              `:=`(
                (max_col)  = i.nmax,
                (min_col)  = i.nmin,
                (mean_col) = i.nmean
              )]

    # Clean up to free memory within the loop
    rm(neighbor_vals, expanded, stats)
    gc(verbose = FALSE)
  }

  # Clean up helper column
  cell_data[, .row_idx := NULL]

  return(cell_data)
}


# =============================================================================
# Usage — drop-in replacement for the original outer loop
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# cell_data now contains the same 15 new columns as the original code:
#   neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl,
#   neighbor_max_ec,  neighbor_min_ec,  neighbor_mean_ec,
#   ... etc.
#
# The trained Random Forest model is untouched — just use cell_data
# as the input to predict() as before.
```

---

## Why This Preserves Correctness

| Property | Original | Replacement |
|---|---|---|
| Neighbor definition | Rook contiguity from `rook_neighbors_unique` nb object | Same nb object, same edges |
| Temporal matching | Neighbors matched within same year | Joined on `(neighbor_id, year)` — identical |
| NA handling | `neighbor_vals[!is.na(neighbor_vals)]`; returns `NA` if all missing | `expanded[!is.na(nval), …]`; unmatched rows stay `NA` — identical |
| Aggregation functions | `max`, `min`, `mean` | Same functions |
| Output columns | `neighbor_{max,min,mean}_{var}` | Same names and semantics |
| RF model | Not retrained | Not retrained |

The numerical estimand is identical because the same set of non-NA neighbor values is aggregated with the same functions. The only change is **how** those values are gathered — vectorized joins instead of interpreted loops.

---

## Memory Profile (16 GB laptop)

| Object | Approximate size |
|---|---|
| `cell_data` (6.46M × 110 cols) | ~5.5 GB |
| `edges` (1.37M × 2 int cols) | ~11 MB |
| `expanded` (peak: ~38.4M × 4 cols) | ~1.2 GB |
| `stats` (≤ 6.46M × 4 cols) | ~200 MB |
| Overhead / gc headroom | ~2 GB |
| **Peak total** | **~9 GB** ✓ |

Processing each variable sequentially (with `rm` + `gc` inside the loop) keeps peak memory well within 16 GB.