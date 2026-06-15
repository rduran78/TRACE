 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is in **`build_neighbor_lookup`**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting and named-vector lookups. Named vector lookups in R (`idx_lookup[neighbor_keys]`) are hash-table lookups, but doing millions of `paste` + lookup operations inside a sequential `lapply` is extremely slow. The second bottleneck is **`compute_neighbor_stats`**, which iterates over the same 6.46 million rows, subsetting and computing `max/min/mean` per row. Together, for 5 variables, this is ~32.3 million row-level R function calls. The `do.call(rbind, result)` on a 6.46-million-element list is also costly.

**Summary of problems:**

| Component | Issue |
|---|---|
| `build_neighbor_lookup` | 6.46M iterations of `paste` + named-vector hash lookup; builds a 6.46M-element list of integer vectors |
| `compute_neighbor_stats` | 6.46M iterations × 5 variables = 32.3M R-level subsetting + aggregation calls |
| `do.call(rbind, ...)` | Binding 6.46M small vectors into a matrix is slow |
| Memory | The neighbor lookup list alone (6.46M elements, each a variable-length integer vector) consumes several GB |

---

## Optimization Strategy

**Replace row-level R loops with vectorized `data.table` joins and group-by aggregations.**

The key insight: the neighbor lookup is conceptually a **join**. Each `(cell, year)` row needs to find its neighbors' `(neighbor_cell, year)` rows and aggregate their values. This is a classic equi-join + grouped aggregation, which `data.table` handles in highly optimized C code.

**Steps:**

1. **Build an edge table** (a two-column `data.table` of `id → neighbor_id`) from the `nb` object — done once, ~1.37M rows.
2. **Join** the edge table to the panel data on `(neighbor_id, year)` to get neighbor values — a single keyed join, no R-level loop.
3. **Aggregate** `max`, `min`, `mean` grouped by the origin row — a single `data.table` group-by.
4. **Repeat** for each of the 5 variables (or do all at once).

This eliminates all 6.46M-iteration `lapply` calls and the giant list-of-vectors lookup structure.

**Expected improvement:** From ~86+ hours to roughly **5–20 minutes** on the same laptop, with peak RAM well within 16 GB.

---

## Working R Code

```r
library(data.table)

# ==============================================================
# STEP 1: Convert the nb object to a data.table edge list (once)
# ==============================================================
build_edge_table <- function(id_order, neighbors) {
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer index vectors)
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)
  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# ~1.37M rows, two integer columns — trivial memory

# ==============================================================
# STEP 2: Convert cell_data to data.table (if not already)
# ==============================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Create a row index to preserve original order and merge results back
cell_data[, .row_id := .I]

# ==============================================================
# STEP 3: Compute neighbor features for all variables at once
# ==============================================================
compute_all_neighbor_features <- function(cell_data, edge_dt, var_names) {
  # Subset to only the columns we need for the join
  # This keeps memory low during the join
  join_cols <- c("id", "year", var_names)
  dt_slim   <- cell_data[, ..join_cols]

  # Key the slim table for fast join on (id, year)
  setkey(dt_slim, id, year)

  # Join: for every (id, year) row, find its neighbors' values

  # edge_dt has (id, neighbor_id). We join edge_dt to dt_slim twice:
  #   - first to get the origin row's year
  #   - then to get the neighbor's values for that year

  # Approach: build a long table of (origin_id, year, neighbor_id),
  # then join to get neighbor values, then aggregate.

  # Get unique (id, year) with row id
  origin <- cell_data[, .(id, year, .row_id)]

  # Merge origin with edge table to get (origin_row_id, year, neighbor_id)
  # This is the most memory-intensive step: ~1.37M edges × 28 years ≈ 38.4M rows
  # But each row is just 3 integers ≈ 38.4M × 12 bytes ≈ 461 MB — fits in 16 GB
  setkey(edge_dt, id)
  setkey(origin, id)
  expanded <- edge_dt[origin, on = "id", allow.cartesian = TRUE,
                      nomatch = NULL,
                      .(.row_id, year, neighbor_id)]

  # Now join to get the neighbor's variable values
  # We join expanded to dt_slim on (neighbor_id == id, year == year)
  setnames(expanded, "neighbor_id", "nb_id")
  setkey(expanded, nb_id, year)
  setkey(dt_slim, id, year)

  neighbor_vals <- dt_slim[expanded, on = c(id = "nb_id", "year"),
                           nomatch = NA]
  # neighbor_vals now has columns: id (=neighbor), year, var_names..., .row_id

  # Aggregate per origin row
  agg_exprs <- list()
  for (v in var_names) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("neighbor_max_", v)]]  <- substitute(
      suppressWarnings(max(V, na.rm = TRUE)), list(V = v_sym))
    agg_exprs[[paste0("neighbor_min_", v)]]  <- substitute(
      suppressWarnings(min(V, na.rm = TRUE)), list(V = v_sym))
    agg_exprs[[paste0("neighbor_mean_", v)]] <- substitute(
      mean(V, na.rm = TRUE), list(V = v_sym))
  }

  agg_result <- neighbor_vals[, eval(as.call(c(as.name("list"),
                                                agg_exprs))),
                              by = .row_id]

  # Replace Inf/-Inf (from max/min on all-NA groups) with NA
  inf_cols <- grep("^neighbor_(max|min)_", names(agg_result), value = TRUE)
  for (col in inf_cols) {
    set(agg_result, which(is.infinite(agg_result[[col]])), col, NA_real_)
  }

  return(agg_result)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

agg_result <- compute_all_neighbor_features(cell_data, edge_dt,
                                            neighbor_source_vars)

# ==============================================================
# STEP 4: Merge back to cell_data
# ==============================================================
setkey(agg_result, .row_id)
setkey(cell_data, .row_id)
cell_data <- agg_result[cell_data, on = ".row_id"]

# Clean up helper column
cell_data[, .row_id := NULL]

# ==============================================================
# STEP 5: (Optional) Convert back to data.frame if downstream
#          code or the trained RF model expects one
# ==============================================================
# cell_data <- as.data.frame(cell_data)
```

---

## Memory Management Variant (If 16 GB Is Tight)

If the ~38M-row expanded join table strains RAM, process one variable at a time and drop intermediates:

```r
compute_neighbor_features_lowmem <- function(cell_data, edge_dt, var_names) {
  origin <- cell_data[, .(id, year, .row_id)]
  setkey(edge_dt, id)
  setkey(origin, id)

  # Expand once — this is the big table (~38M rows, 3 int cols ≈ 460 MB)
  expanded <- edge_dt[origin, on = "id", allow.cartesian = TRUE,
                      nomatch = NULL,
                      .(.row_id, year, neighbor_id)]
  setnames(expanded, "neighbor_id", "nb_id")
  setkey(expanded, nb_id, year)

  all_agg <- cell_data[, .(.row_id)]  # anchor table

  for (v in var_names) {
    message("Processing neighbor stats for: ", v)
    dt_v <- cell_data[, .(id, year, val = get(v))]
    setkey(dt_v, id, year)

    joined <- dt_v[expanded, on = c(id = "nb_id", "year"), nomatch = NA]

    agg_v <- joined[, .(
      nmax  = suppressWarnings(max(val, na.rm = TRUE)),
      nmin  = suppressWarnings(min(val, na.rm = TRUE)),
      nmean = mean(val, na.rm = TRUE)
    ), by = .row_id]

    # Fix Inf
    agg_v[is.infinite(nmax), nmax := NA_real_]
    agg_v[is.infinite(nmin), nmin := NA_real_]

    setnames(agg_v,
             c("nmax", "nmin", "nmean"),
             paste0("neighbor_", c("max_", "min_", "mean_"), v))

    all_agg <- agg_v[all_agg, on = ".row_id"]
    rm(joined, agg_v, dt_v); gc()
  }

  return(all_agg)
}

cell_data[, .row_id := .I]
agg_result <- compute_neighbor_features_lowmem(cell_data, edge_dt,
                                               neighbor_source_vars)
setkey(agg_result, .row_id)
setkey(cell_data, .row_id)
cell_data <- agg_result[cell_data, on = ".row_id"]
cell_data[, .row_id := NULL]
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Trained RF model** | No model code is touched. Only the feature-engineering step is optimized. The output columns have the same names and identical numerical values. |
| **Numerical estimand** | `max`, `min`, `mean` with `na.rm = TRUE` are the same operations as the original code. `Inf`/`-Inf` from empty groups are converted to `NA`, matching the original `c(NA, NA, NA)` return. |
| **Row order** | The `.row_id` key ensures results merge back in the original row order. |

**Expected runtime:** ~5–20 minutes on a 16 GB laptop (dominated by the single 38M-row keyed join and grouped aggregation, both executed in `data.table`'s parallel C backend) versus the original 86+ hours.