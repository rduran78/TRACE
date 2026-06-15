 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) per-row `lapply` with string operations

`build_neighbor_lookup` calls `lapply` over all ~6.46 million rows. For each row it:

1. **Pastes** neighbor cell IDs with the current year to form string keys — ~4 string allocations per neighbor × ~4 neighbors (rook) × 6.46M rows ≈ **103 million `paste` calls**.
2. **Looks up** those keys in a named character vector of length 6.46M — each lookup is an O(1) hash probe, but the hash table is built over **character keys of length 6.46M**, and R's named-vector lookup has substantial per-call overhead.

But more importantly, the entire `neighbor_lookup` structure (a list of 6.46M integer vectors) is **invariant across the 5 source variables**. It's correctly built once and reused — so that part is fine. The real bottleneck is the *construction* of `neighbor_lookup` itself via the per-row `lapply`.

### Why it takes 86+ hours

| Step | Complexity | Estimated cost |
|------|-----------|----------------|
| `paste` to build `idx_lookup` (once) | O(N) | ~seconds |
| Per-row `paste` + hash lookup inside `lapply` | O(N × avg_neighbors) with high constant factor | **~hours** |
| `compute_neighbor_stats` (5 vars) | O(N × avg_neighbors) but pure integer indexing | ~minutes |

The dominant cost is `build_neighbor_lookup`. The per-row string construction and named-vector lookup in an `lapply` over 6.46M rows is the killer. R's `lapply` here cannot be vectorized because each row has a *variable-length* neighbor set.

### The Deeper Insight: The neighbor graph is year-invariant

Every cell has the same rook neighbors in every year. The `nb` object defines a **spatial** adjacency that doesn't change over time. The lookup is simply: "for row `i` (cell `c` in year `t`), find all rows where cell ∈ neighbors(c) AND year = `t`." This is a **structural join** that can be computed entirely with vectorized integer arithmetic — no strings needed.

---

## Optimization Strategy

**Eliminate all string operations. Replace the per-row `lapply` with a fully vectorized merge/join approach:**

1. **Build a cell-to-row-index mapping** using integer arithmetic: create a matrix or data.table keyed by `(cell_id, year)` → `row_index`.
2. **Expand the neighbor list** into an edge table: `(cell_from, cell_to)` — ~1.37M directed edges.
3. **Cross-join** edges with years to get `(cell_from, year, cell_to)` — ~1.37M × 28 = ~38.5M rows.
4. **Join** to map `(cell_from, year)` → `source_row` and `(cell_to, year)` → `neighbor_row`.
5. **Compute grouped statistics** (max, min, mean) per `source_row` using `data.table` grouped aggregation — fully vectorized C-level code.

This replaces billions of R-level operations with a handful of vectorized joins and group-bys.

**Estimated time: minutes instead of days.**

---

## Working R Code

```r
library(data.table)

#' Vectorized neighbor feature construction.
#' Replaces build_neighbor_lookup + compute_neighbor_stats entirely.
#'
#' @param cell_data        data.frame or data.table with columns: id, year, and all var_names
#' @param id_order          integer vector of cell IDs in the order matching rook_neighbors_unique
#' @param rook_neighbors    spdep nb object (list of integer index vectors into id_order)
#' @param neighbor_source_vars character vector of variable names to compute neighbor stats for
#' @return data.table with original columns plus neighbor feature columns appended
compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors,
                                          neighbor_source_vars) {

  dt <- as.data.table(cell_data)

  # --- Step 1: Assign a row index to every row ---
  dt[, .row_idx := .I]

  # --- Step 2: Build edge list from the nb object ---
  # rook_neighbors[[k]] contains integer indices into id_order
  # So cell id_order[k] has neighbors id_order[ rook_neighbors[[k]] ]
  edges <- rbindlist(lapply(seq_along(rook_neighbors), function(k) {
    nb_idx <- rook_neighbors[[k]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
      return(NULL)
    }
    data.table(cell_from = id_order[k], cell_to = id_order[nb_idx])
  }))
  # edges is ~1.37M rows: (cell_from, cell_to)

  cat(sprintf("Edge table: %s directed edges\n", format(nrow(edges), big.mark = ",")))

  # --- Step 3: Build a keyed lookup from (id, year) -> row_idx ---
  row_lookup <- dt[, .(id, year, .row_idx)]
  setkey(row_lookup, id, year)

  # --- Step 4: Get unique years ---
  years <- sort(unique(dt$year))

  # --- Step 5: Cross-join edges × years ---
  # This gives us ~38.5M rows: (cell_from, cell_to, year)
  # Memory: ~38.5M × 3 int columns ≈ 462 MB — fits in 16 GB
  edge_year <- CJ_edges_years(edges, years)

  cat(sprintf("Edge-year table: %s rows\n", format(nrow(edge_year), big.mark = ",")))

  # --- Step 6: Map cell_from,year -> source row index ---
  setkey(edge_year, cell_from, year)
  edge_year[row_lookup, source_row := i..row_idx, on = .(cell_from = id, year = year)]

  # --- Step 7: Map cell_to,year -> neighbor row index ---
  setkey(edge_year, cell_to, year)
  edge_year[row_lookup, neighbor_row := i..row_idx, on = .(cell_to = id, year = year)]

  # --- Step 8: Drop any unmatched (boundary cells with neighbors outside data) ---
  edge_year <- edge_year[!is.na(source_row) & !is.na(neighbor_row)]

  cat(sprintf("Matched edge-year rows: %s\n", format(nrow(edge_year), big.mark = ",")))

  # --- Step 9: For each source variable, pull neighbor values and aggregate ---
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Computing neighbor stats for: %s\n", var_name))

    # Pull the neighbor's value for this variable
    edge_year[, nval := dt[[var_name]][neighbor_row]]

    # Grouped aggregation: max, min, mean per source_row (excluding NAs)
    agg <- edge_year[!is.na(nval),
                     .(nb_max  = max(nval),
                       nb_min  = min(nval),
                       nb_mean = mean(nval)),
                     by = source_row]

    # Create full-length result columns (NA for cells with no valid neighbors)
    max_col  <- rep(NA_real_, nrow(dt))
    min_col  <- rep(NA_real_, nrow(dt))
    mean_col <- rep(NA_real_, nrow(dt))

    max_col[agg$source_row]  <- agg$nb_max
    min_col[agg$source_row]  <- agg$nb_min
    mean_col[agg$source_row] <- agg$nb_mean

    # Assign to dt with the same naming convention as the original code
    # Adjust column names to match whatever compute_and_add_neighbor_features produced
    set(dt, j = paste0(var_name, "_neighbor_max"),  value = max_col)
    set(dt, j = paste0(var_name, "_neighbor_min"),  value = min_col)
    set(dt, j = paste0(var_name, "_neighbor_mean"), value = mean_col)
  }

  # Clean up temporary column
  edge_year[, nval := NULL]
  dt[, .row_idx := NULL]

  return(dt)
}


#' Helper: Cross-join edges with years without full CJ explosion in memory at once.
#' For ~1.37M edges × 28 years this is fine to do directly.
CJ_edges_years <- function(edges, years) {
  # Replicate each edge for every year
  n_edges <- nrow(edges)
  n_years <- length(years)

  edge_year <- data.table(
    cell_from = rep(edges$cell_from, times = n_years),
    cell_to   = rep(edges$cell_to,   times = n_years),
    year      = rep(years, each = n_edges)
  )
  return(edge_year)
}
```

### Usage — drop-in replacement for the original outer loop:

```r
# ---- BEFORE (86+ hours) ----
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }

# ---- AFTER (minutes) ----
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors       = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# cell_data is now a data.table with the 15 new neighbor columns appended.
# Convert back to data.frame if needed downstream:
# cell_data <- as.data.frame(cell_data)
```

### Column Name Alignment

If the original `compute_and_add_neighbor_features` used different naming conventions (e.g., `ntl_max_neighbor` instead of `ntl_neighbor_max`), adjust these three lines in the loop:

```r
set(dt, j = paste0(var_name, "_neighbor_max"),  value = max_col)
set(dt, j = paste0(var_name, "_neighbor_min"),  value = min_col)
set(dt, j = paste0(var_name, "_neighbor_mean"), value = mean_col)
```

to match whatever column names the trained Random Forest model expects. The **numerical values are identical** — same max, min, mean over the same neighbor sets — so the trained model needs no retraining.

---

## Performance Analysis

| Metric | Original | Optimized |
|--------|----------|-----------|
| String paste calls | ~103M+ | **0** |
| R-level loop iterations | ~6.46M | **0** (vectorized) |
| Hash lookups on 6.46M-length named vector | ~26M | **0** (keyed join) |
| `data.table` grouped aggregation | N/A | 5 passes over ~38.5M rows |
| **Estimated wall time** | **86+ hours** | **5–15 minutes** |
| Peak memory (edge_year table) | ~varies | ~1.2 GB (fits in 16 GB) |

### Why this is correct

- The `nb` object indexes into `id_order`. Edge `(id_order[k], id_order[j])` means cell `k` and cell `j` are rook neighbors.
- The join on `(id, year)` maps each spatial edge to the correct temporal row — exactly what the original string-key approach did.
- `max`, `min`, `mean` with `NA` removal matches the original `compute_neighbor_stats` logic.
- Cells with zero valid neighbors get `NA` for all three statistics — matching the original `c(NA, NA, NA)` return.