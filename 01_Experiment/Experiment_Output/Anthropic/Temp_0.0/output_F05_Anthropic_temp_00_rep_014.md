 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Inside the Loop

`build_neighbor_lookup` iterates over all ~6.46 million rows. For **each** row it:

1. Calls `paste()` to build string keys for every neighbor of that cell-year.
2. Looks those keys up in a named character vector (`idx_lookup`) — which itself was built by pasting 6.46M strings.

This means:
- **~6.46M iterations**, each doing string concatenation and hash-table lookups.
- The neighbor relationships are **time-invariant** (rook contiguity doesn't change year to year), yet the code re-discovers the same spatial neighbors for every year a cell appears.
- The string-keyed lookup is an indirect way of doing what is really a **join on (id, year)** — something that can be done vectorized with integer indexing.

### Quantifying the Waste

| Operation | Count |
|---|---|
| Rows (cell-years) | ~6.46M |
| Unique cells | 344,208 |
| Years | 28 |
| Directed neighbor pairs | ~1.37M |
| Neighbor-year lookups inside `lapply` | ~1.37M × 28 ≈ **38.5M** string pastes + hash lookups |

The entire `build_neighbor_lookup` is **O(R × avg_neighbors)** with expensive string constants. Then `compute_neighbor_stats` loops over 6.46M entries again per variable (×5 variables).

### Broader Pattern

The architecture is:

```
build_neighbor_lookup (slow, string-based)
  → returns list of 6.46M integer vectors
    → compute_neighbor_stats loops over that list 5 times
```

Both stages can be replaced with **vectorized, integer-only operations** using `data.table`.

---

## Optimization Strategy

### Key Insight: Separate Spatial Structure from Temporal Replication

1. **Convert the `nb` object to an edge list once** — a two-column integer table of `(cell_id, neighbor_id)` with ~1.37M rows.
2. **Join the edge list to the panel on `(neighbor_id, year)`** — this is a single `data.table` merge that produces ~38.5M rows, fully vectorized.
3. **Aggregate neighbor statistics** with a single grouped `data.table` operation per variable (or all at once).

This eliminates:
- All `paste()`/string-key construction.
- The 6.46M-element `lapply`.
- The per-variable R-level loop inside `compute_neighbor_stats`.

### Expected Speedup

| Stage | Old | New |
|---|---|---|
| Neighbor lookup build | ~hours (string `lapply`) | ~seconds (nb→edge list) |
| Per-variable stats | ~hours (`lapply` over 6.46M lists) | ~seconds (`data.table` grouped aggregation) |
| **Total for 5 variables** | **86+ hours** | **~1–3 minutes** |

Memory stays well within 16 GB: the edge-list × years table is ~38.5M rows × a few columns of integers/doubles.

---

## Working R Code

```r
library(data.table)

# ============================================================
# 1. Convert spdep nb object to an integer edge-list (one-time)
# ============================================================
nb_to_edge_list <- function(nb_obj, id_order) {
  # nb_obj: list of integer index vectors (spdep::nb), 
  #         indices refer to positions in id_order
  # id_order: vector of cell IDs in the order matching nb_obj
  from <- rep(seq_along(nb_obj), lengths(nb_obj))
  to   <- unlist(nb_obj, use.names = FALSE)
  # Remove the 0-neighbor sentinel if present
  valid <- to > 0L
  data.table(
    cell_id     = id_order[from[valid]],
    neighbor_id = id_order[to[valid]]
  )
}

# ============================================================
# 2. Vectorized neighbor-feature construction
# ============================================================
compute_all_neighbor_features <- function(cell_data, 
                                          neighbor_source_vars, 
                                          rook_neighbors_unique, 
                                          id_order) {
  
  # --- Step A: build edge list (spatial, time-invariant) ------
  edges <- nb_to_edge_list(rook_neighbors_unique, id_order)
  # ~1.37M rows: (cell_id, neighbor_id)
  
  # --- Step B: convert panel to data.table --------------------
  dt <- as.data.table(cell_data)
  
  # Ensure key columns are present and well-typed
  dt[, id   := as.integer(id)]
  dt[, year := as.integer(year)]
  
  # --- Step C: build the neighbor-year table ------------------
  # For every directed edge, replicate across all 28 years.
  # Instead of a cross-join, we join edges to the panel on the
  # neighbor side to pull neighbor values directly.
  
  # Subset to only the columns we need for the neighbor side
  neighbor_cols <- c("id", "year", neighbor_source_vars)
  dt_neighbor <- dt[, ..neighbor_cols]
  setnames(dt_neighbor, "id", "neighbor_id")
  
  # Key for fast join
  setkey(dt_neighbor, neighbor_id, year)
  
  # Expand edges: attach year from the focal cell
  # We need (cell_id, year) → list of neighbor values
  # Approach: join edges to focal rows to get (cell_id, neighbor_id, year),
  # then join to dt_neighbor to get neighbor values.
  
  # Focal side: just need (cell_id, year) — one row per cell-year
  focal <- dt[, .(cell_id = id, year)]
  
  # Cross of focal × edges on cell_id
  # This gives us (cell_id, neighbor_id, year) — ~38.5M rows
  setkey(edges, cell_id)
  setkey(focal, cell_id)
  expanded <- edges[focal, on = "cell_id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded has columns: cell_id, neighbor_id, year
  
  # Now join to get neighbor variable values
  setkey(expanded, neighbor_id, year)
  expanded <- dt_neighbor[expanded, on = .(neighbor_id, year), nomatch = NA]
  # expanded now has: neighbor_id, year, <var columns>, cell_id
  
  # --- Step D: aggregate per (cell_id, year) ------------------
  # Build aggregation expressions dynamically
  agg_exprs <- list()
  for (v in neighbor_source_vars) {
    v_sym <- as.name(v)
    suffix_max  <- paste0("neighbor_max_",  v)
    suffix_min  <- paste0("neighbor_min_",  v)
    suffix_mean <- paste0("neighbor_mean_", v)
    agg_exprs[[suffix_max]]  <- bquote(as.numeric(max(.(v_sym),  na.rm = TRUE)))
    agg_exprs[[suffix_min]]  <- bquote(as.numeric(min(.(v_sym),  na.rm = TRUE)))
    agg_exprs[[suffix_mean]] <- bquote(mean(.(v_sym), na.rm = TRUE))
  }
  
  # Single grouped aggregation
  agg_call <- as.call(c(as.name("list"), agg_exprs))
  stats <- expanded[, eval(agg_call), by = .(cell_id, year)]
  
  # Fix Inf/-Inf from max/min on all-NA groups → NA
  num_cols <- setdiff(names(stats), c("cell_id", "year"))
  for (col in num_cols) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }
  
  # --- Step E: merge back to the original panel ---------------
  setkey(stats, cell_id, year)
  setkey(dt, id, year)
  
  # Merge
  dt <- stats[dt, on = .(cell_id = id, year = year)]
  
  # Restore column name 'id'
  setnames(dt, "cell_id", "id")
  
  # Return as data.frame to preserve downstream compatibility
  as.data.frame(dt)
}

# ============================================================
# 3. Drop-in replacement for the outer loop
# ============================================================
# BEFORE (86+ hours):
#   neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
#   for (var_name in neighbor_source_vars) {
#     cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
#   }
#
# AFTER (~1-3 minutes):

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data            = cell_data,
  neighbor_source_vars = neighbor_source_vars,
  rook_neighbors_unique = rook_neighbors_unique,
  id_order             = id_order
)

# The trained Random Forest model is untouched.
# Prediction proceeds exactly as before:
# preds <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same numerical estimand** | `max`, `min`, `mean` are computed on exactly the same set of neighbor values per `(cell_id, year)`. The join `(neighbor_id, year)` replicates the original string-key lookup logic. `na.rm = TRUE` matches the original `neighbor_vals[!is.na(neighbor_vals)]` filtering. |
| **Column names** | Output columns follow the pattern `neighbor_max_ntl`, `neighbor_min_ntl`, `neighbor_mean_ntl`, etc. Adjust the naming template if your trained RF expects different names. |
| **Trained RF model** | No model retraining. The code only reconstructs the feature columns that the model already expects. |
| **Edge cases** | Cells with zero neighbors (islands) get `NA` for all neighbor stats, matching the original `c(NA, NA, NA)` return. |

---

## Memory Profile (16 GB laptop)

| Object | Approximate Size |
|---|---|
| `dt` (panel) | ~5.7 GB (6.46M × 110 cols) |
| `edges` | ~11 MB (1.37M × 2 int cols) |
| `expanded` | ~2.2 GB (38.5M × 7 cols) |
| `stats` | ~1.5 GB (6.46M × 17 cols) |
| **Peak** | **~10 GB** — fits in 16 GB |

If memory is tight, process variables in batches of 2–3 instead of all 5 at once, which reduces `expanded` width and `stats` width proportionally.