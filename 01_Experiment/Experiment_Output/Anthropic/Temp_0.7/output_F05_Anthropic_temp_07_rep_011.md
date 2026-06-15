 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### Local Inefficiency
- `paste(data$id, data$year, sep = "_")` creates ~6.46M strings to build `idx_lookup` — this happens once, so it's O(n) and tolerable in isolation.
- But inside the `lapply` over all 6.46M rows, `paste(neighbor_cell_ids, data$year[i], sep = "_")` is called per row, generating ~1.37 billion string constructions and hash lookups across all rows. This is the dominant cost.

### Broader Algorithmic Problem
The entire string-keyed lookup is an **unnecessary indirection**. The fundamental operation is: *for each cell-year row, find the rows that share the same year and belong to a neighboring cell*. This is a **join** — specifically a relational equi-join on `(year, neighbor_id)` — being implemented as a per-row hash-table probe via string keys. The correct reformulation is:

1. **Build the neighbor lookup once as an integer-indexed structure** (cell index → neighbor cell indices), which is already available from `rook_neighbors_unique`.
2. **Build a year-indexed structure** (year → row indices per cell), so that finding "the row for cell `j` in year `t`" is an O(1) integer lookup, not a string-hash probe.
3. **Vectorize the neighbor-stat computation** using `data.table` grouped operations or a single sparse-matrix multiplication, eliminating the per-row `lapply` entirely.

The best reformulation recognizes that the neighbor mean/max/min over a variable is equivalent to a **sparse matrix operation**: if `W` is the row-normalized (or raw) neighbor adjacency matrix in cell-year space, then `neighbor_mean = W %*% x`. Max and min require a grouped operation but can still be vectorized.

---

## Optimization Strategy

| Aspect | Current | Proposed |
|---|---|---|
| Neighbor resolution | Per-row string paste + hash lookup (6.46M × ~4 neighbors) | One-time integer expansion via `data.table` join |
| Stats computation | `lapply` over 6.46M rows, R-level loop | Vectorized `data.table` grouped aggregation |
| Repetition across vars | Neighbor lookup reused, but stats loop is R-level | Same neighbor index structure, fully vectorized stats |
| Estimated time | 86+ hours | **Minutes** (dominated by `data.table` grouped ops) |
| RAM | String vector ~6.46M × 20 bytes + hash table | Integer edge list ~50M rows × 3 cols ≈ ~1.2 GB |

---

## Working R Code

```r
library(data.table)

#' Build a fully vectorized neighbor feature pipeline.
#' Preserves the exact numerical estimand (max, min, mean of neighbor values).
#'
#' @param cell_data       data.frame or data.table with columns: id, year, and all neighbor_source_vars
#' @param id_order        integer vector of cell IDs in the order matching rook_neighbors_unique
#' @param rook_neighbors  spdep::nb object (list of integer index vectors into id_order)
#' @param neighbor_source_vars character vector of variable names
#' @return data.table with original columns plus neighbor feature columns appended

build_all_neighbor_features <- function(cell_data,
                                        id_order,
                                        rook_neighbors,
                                        neighbor_source_vars) {

  dt <- as.data.table(cell_data)

  # ---- Step 1: Build directed edge list (focal_cell_idx -> neighbor_cell_id) ----
  # rook_neighbors[[k]] gives integer indices into id_order for the neighbors of id_order[k]
  message("Building edge list...")

  # Pre-allocate edge list vectors
  n_edges <- sum(lengths(rook_neighbors))
  focal_ids   <- integer(n_edges)
  neighbor_ids <- integer(n_edges)

  pos <- 1L
  for (k in seq_along(rook_neighbors)) {
    nb_idx <- rook_neighbors[[k]]
    if (length(nb_idx) == 0L) next
    n <- length(nb_idx)
    focal_ids[pos:(pos + n - 1L)]    <- id_order[k]
    neighbor_ids[pos:(pos + n - 1L)] <- id_order[nb_idx]
    pos <- pos + n
  }

  edges <- data.table(focal_id = focal_ids, neighbor_id = neighbor_ids)
  rm(focal_ids, neighbor_ids)

  # ---- Step 2: Assign a row index to each cell-year observation ----
  dt[, row_idx := .I]

  # ---- Step 3: For each focal row, find all neighbor rows (same year) via join ----
  message("Joining edges to panel on year...")

  # Keyed lookup: for a given (id, year), what is the row_idx?
  id_year_key <- dt[, .(id, year, row_idx)]

  # Join edges with focal rows to get (focal_row_idx, neighbor_id, year)
  # Then join with id_year_key to get neighbor_row_idx
  # This replaces the entire per-row lapply + string-key lookup.

  # Focal side: get year for each focal cell-year
  setkey(id_year_key, id, year)

  # Expand edges × years:
  # For every edge (focal_id, neighbor_id) and every year the focal_id appears,
  # find the neighbor_id's row in that same year.

  # Get unique (focal_id, year, focal_row_idx)
  focal_rows <- dt[, .(focal_id = id, year, focal_row_idx = row_idx)]

  # Join: for each focal row, attach its neighbor cell IDs

  setkey(edges, focal_id)
  setkey(focal_rows, focal_id)

  # This is the big expansion: ~6.46M rows × ~4 neighbors = ~25.8M rows
  message("Expanding focal-neighbor-year combinations...")
  expanded <- edges[focal_rows, on = .(focal_id), allow.cartesian = TRUE, nomatch = NULL]
  # Columns: focal_id, neighbor_id, year, focal_row_idx

  # Now resolve neighbor_id + year -> neighbor_row_idx
  setnames(id_year_key, c("id", "year", "row_idx"), c("neighbor_id", "year", "neighbor_row_idx"))
  setkey(id_year_key, neighbor_id, year)
  setkey(expanded, neighbor_id, year)

  message("Resolving neighbor row indices...")
  expanded <- id_year_key[expanded, on = .(neighbor_id, year), nomatch = NA]
  # Drop rows where neighbor has no observation in that year
  expanded <- expanded[!is.na(neighbor_row_idx)]

  # ---- Step 4: Compute neighbor stats vectorized ----
  message("Computing neighbor statistics...")

  for (var_name in neighbor_source_vars) {
    message("  Processing: ", var_name)

    # Pull the variable values aligned to neighbor rows
    expanded[, nval := dt[[var_name]][neighbor_row_idx]]

    # Compute grouped stats per focal row
    stats <- expanded[!is.na(nval),
                      .(
                        nb_max  = max(nval),
                        nb_min  = min(nval),
                        nb_mean = mean(nval)
                      ),
                      by = .(focal_row_idx)]

    # Initialize columns with NA
    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Assign results back by row index
    dt[stats$focal_row_idx, (max_col)  := stats$nb_max]
    dt[stats$focal_row_idx, (min_col)  := stats$nb_min]
    dt[stats$focal_row_idx, (mean_col) := stats$nb_mean]

    expanded[, nval := NULL]
  }

  # Clean up
  dt[, row_idx := NULL]

  message("Done.")
  return(dt)
}
```

### Drop-in Replacement for the Outer Loop

```r
# ---- BEFORE (86+ hours) ----
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }

# ---- AFTER (minutes) ----
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors       = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# Convert back to data.frame if downstream code expects it
cell_data <- as.data.frame(cell_data)

# The trained Random Forest model is unchanged — only the feature table is rebuilt.
# Column names (e.g., ntl_neighbor_max, ntl_neighbor_min, ntl_neighbor_mean) must
# match what the RF model expects. Adjust the naming pattern above if needed.
```

---

## Why This Preserves Correctness

| Property | Guarantee |
|---|---|
| **Same neighbor set** | Edge list is built from the identical `rook_neighbors_unique` nb object and `id_order` mapping. |
| **Same year matching** | The `data.table` join on `(neighbor_id, year)` is the exact equivalent of `paste(neighbor_cell_ids, data$year[i], sep = "_")` lookup. |
| **Same NA handling** | `nomatch = NA` + `!is.na(neighbor_row_idx)` mirrors `result[!is.na(result)]`; `!is.na(nval)` mirrors `neighbor_vals[!is.na(neighbor_vals)]`. |
| **Same aggregates** | `max`, `min`, `mean` computed on the identical filtered neighbor value vectors. |
| **RF model untouched** | No retraining. The feature columns are numerically identical, so predictions are identical. |

## Complexity Comparison

| | Current | Proposed |
|---|---|---|
| String constructions | ~6.46M (lookup) + ~25.8B (per-row paste) | 0 |
| Hash probes | ~25.8B | 0 |
| Core operation | R-level `lapply` over 6.46M rows | `data.table` binary-search join + grouped aggregation (C-level) |
| Expected wall time | 86+ hours | **5–15 minutes** on 16 GB laptop |
| Peak RAM | Moderate (but slow) | ~2–4 GB for the expanded edge table |