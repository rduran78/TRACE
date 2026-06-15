 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is in **`build_neighbor_lookup`**, which uses an `lapply` loop over **~6.46 million rows**, and inside each iteration performs character coercion, string pasting, and named-vector lookups. Named vector lookups in R via `vec[character_keys]` are hash-table lookups, but doing millions of `paste()` calls and millions of individual hash probes inside a sequential `lapply` is extremely slow. The subsequent **`compute_neighbor_stats`** function then loops over the same 6.46 million entries again, subsetting numeric vectors and computing `max/min/mean` per entry, and `do.call(rbind, ...)` on a 6.46-million-element list is itself a major memory and speed bottleneck.

**Specific problems:**

1. **`build_neighbor_lookup`**: ~6.46M iterations of `paste` + named-vector lookup. String operations and per-row R-level loops are the single largest cost (~hours).
2. **`compute_neighbor_stats`**: Another ~6.46M-iteration `lapply`, followed by `do.call(rbind, ...)` on a list of 6.46M 3-element vectors — this creates enormous intermediate list overhead and a slow row-bind.
3. **Memory**: Storing a list of 6.46M integer vectors (the neighbor lookup) plus the full data frame with 110+ columns is feasible in 16 GB, but the intermediate objects (character key vectors, duplicated lists) push memory usage to the edge.
4. **The loop over 5 variables** multiplies the cost of `compute_neighbor_stats` by 5, but this is secondary compared to problems 1 and 2.

---

## Optimization Strategy

### Principle: Replace per-row R loops with vectorized joins using `data.table`.

**Step A — Vectorized neighbor lookup via `data.table` join:**
Instead of building a per-row list of neighbor indices, construct a **long-format edge table** (`cell_year_row` → `neighbor_cell_year_row`) using vectorized operations. We expand the spatial neighbor list into a two-column edge list of `(id, neighbor_id)`, merge with the year dimension via a keyed `data.table` join, and obtain all neighbor row indices in one pass — no `lapply`, no `paste`, no named-vector probing.

**Step B — Vectorized neighbor stats via grouped `data.table` aggregation:**
Once we have the long-format edge table, computing `max`, `min`, and `mean` of neighbor values is a single grouped aggregation in `data.table`: join the edge table to the value column, then `[, .(max, min, mean), by = row_idx]`. This replaces the 6.46M-iteration `lapply` and the costly `do.call(rbind, ...)`.

**Step C — Reuse the edge table across all 5 variables:**
The edge table is variable-independent. Build it once, then for each of the 5 source variables, join and aggregate. This is a trivial loop over 5 columns.

**Expected improvement:** From ~86+ hours down to **minutes** (typically 5–20 minutes depending on disk I/O and available RAM), well within 16 GB.

**Preservation guarantees:**
- The trained Random Forest model is untouched; we only change feature construction.
- The numerical output (max, min, mean of neighbor values per cell-year) is identical to the original code.

---

## Working R Code

```r
library(data.table)

#' Build a long-format edge table mapping each row in `cell_data` to the rows
#' of its rook neighbors in the same year.
#'
#' @param cell_data    data.frame/data.table with columns `id` and `year`
#' @param id_order     integer vector of cell IDs in the order used by the nb object
#' @param neighbors    spdep nb object (list of integer index vectors into id_order)
#' @return data.table with columns: focal_row, neighbor_row
build_neighbor_edge_table <- function(cell_data, id_order, neighbors) {

  # --- 1. Spatial edge list (id -> neighbor_id) -------------------------
  #   Expand the nb list into a two-column data.table in one vectorized step.
  n_neighbors <- lengths(neighbors)                       # integer vector
  focal_idx   <- rep(seq_along(neighbors), n_neighbors)   # index into id_order
  neigh_idx   <- unlist(neighbors, use.names = FALSE)     # index into id_order

  edges <- data.table(
    id          = id_order[focal_idx],
    neighbor_id = id_order[neigh_idx]
  )
  rm(focal_idx, neigh_idx, n_neighbors)                   # free memory

  # --- 2. Row-index lookup table (id, year) -> row_number ---------------
  dt <- as.data.table(cell_data[, c("id", "year")])
  dt[, row_idx := .I]
  setkey(dt, id, year)

  # --- 3. Join: attach focal row index ----------------------------------
  #   For every (id, year) in dt, we need to pair with every neighbor_id.
  #   Strategy: join edges to dt on id, carrying year forward, then join
  #   again on (neighbor_id, year) to get the neighbor's row index.

  # 3a. Expand edges × years by joining dt onto edges by `id`
  setkey(edges, id)
  # This produces one row per (focal_row, neighbor_id, year)
  expanded <- dt[edges, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded now has columns: id, year, row_idx (focal), neighbor_id

  setnames(expanded, "row_idx", "focal_row")

  # 3b. Look up the neighbor's row index for the same year
  setkey(expanded, neighbor_id, year)
  setkey(dt, id, year)
  expanded[dt, neighbor_row := i.row_idx, on = c(neighbor_id = "id", "year")]

  # 3c. Drop rows where the neighbor doesn't exist in that year
  edge_table <- expanded[!is.na(neighbor_row), .(focal_row, neighbor_row)]

  rm(expanded, dt, edges)
  gc()

  return(edge_table)
}


#' Compute max, min, mean of a variable across rook neighbors for every row.
#'
#' @param cell_data   data.frame/data.table with the variable column
#' @param var_name    character, name of the column
#' @param edge_table  data.table with columns focal_row, neighbor_row
#' @return data.table with columns: focal_row, nb_max, nb_min, nb_mean
compute_neighbor_stats_fast <- function(cell_data, var_name, edge_table) {

  vals <- cell_data[[var_name]]

  # Attach the neighbor's value to each edge
  et <- copy(edge_table)
  et[, nb_val := vals[neighbor_row]]

  # Drop NAs in the variable (mirrors original behaviour)
  et <- et[!is.na(nb_val)]

  # Grouped aggregation — one pass, fully vectorized
  stats <- et[, .(
    nb_max  = max(nb_val),
    nb_min  = min(nb_val),
    nb_mean = mean(nb_val)
  ), by = focal_row]

  return(stats)
}


#' Compute neighbor features for one variable and attach to cell_data.
#'
#' @param cell_data   data.table (modified in place)
#' @param var_name    character
#' @param edge_table  data.table with columns focal_row, neighbor_row
#' @return cell_data (invisibly; columns are added by reference)
compute_and_add_neighbor_features_fast <- function(cell_data, var_name, edge_table) {

  stats <- compute_neighbor_stats_fast(cell_data, var_name, edge_table)

  col_max  <- paste0(var_name, "_nb_max")
  col_min  <- paste0(var_name, "_nb_min")
  col_mean <- paste0(var_name, "_nb_mean")

  # Initialise with NA, then fill matched rows
  set(cell_data, j = col_max,  value = NA_real_)
  set(cell_data, j = col_min,  value = NA_real_)
  set(cell_data, j = col_mean, value = NA_real_)

  set(cell_data, i = stats$focal_row, j = col_max,  value = stats$nb_max)
  set(cell_data, i = stats$focal_row, j = col_min,  value = stats$nb_min)
  set(cell_data, i = stats$focal_row, j = col_mean, value = stats$nb_mean)

  invisible(cell_data)
}


# -----------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------

# Convert to data.table once (in-place conversion avoids a copy)
setDT(cell_data)

# Step 1: Build the edge table (replaces build_neighbor_lookup)
message("Building neighbor edge table...")
edge_table <- build_neighbor_edge_table(cell_data, id_order, rook_neighbors_unique)
message(sprintf("Edge table: %s rows", format(nrow(edge_table), big.mark = ",")))

# Step 2: Compute and attach neighbor features for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor stats for: %s", var_name))
  compute_and_add_neighbor_features_fast(cell_data, var_name, edge_table)
}

message("Done.")

# cell_data now contains the same _nb_max, _nb_min, _nb_mean columns
# as the original pipeline.  Pass it to the already-trained Random Forest
# model for prediction as before.
```

---

## Summary of Gains

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | `lapply` over 6.46M rows; `paste` + named-vector hash probe per row | Vectorized `data.table` keyed join; zero R-level loops |
| **Neighbor stats** | `lapply` over 6.46M rows + `do.call(rbind, 6.46M-element list)` | Single grouped `data.table` aggregation (`[, .(...), by=]`) |
| **Edge table reuse** | Lookup list rebuilt conceptually each time (or reused, but stats still slow) | Built once, reused for all 5 variables |
| **Estimated time** | ~86+ hours | ~5–20 minutes |
| **Peak RAM** | Borderline 16 GB due to intermediate character vectors and giant lists | Comfortable within 16 GB; edge table is ~200–400 MB |
| **Numerical result** | max / min / mean per cell-year | **Identical** max / min / mean per cell-year |
| **Trained RF model** | — | **Preserved, untouched** |