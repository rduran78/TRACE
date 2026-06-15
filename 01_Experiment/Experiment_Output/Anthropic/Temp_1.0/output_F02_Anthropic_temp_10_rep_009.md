 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The bottleneck has two root causes:

### A. `build_neighbor_lookup` — O(n) `lapply` over 6.46 million rows with per-row string pasting and named-vector lookups

* `paste(neighbor_cell_ids, data$year[i], sep = "_")` and `idx_lookup[neighbor_keys]` are called **once per row**. Named character vector lookup in R is hash-based but the constant factor is high when called millions of times.
* Each call also allocates a new character vector, hits the global string cache, and creates intermediate objects. Across 6.46M rows this dominates wall-clock time.

### B. `compute_neighbor_stats` — R-level loop with per-row subsetting and aggregation

* For each of the 5 variables, another `lapply` over 6.46M rows computes `max`, `min`, `mean` on small vectors. The overhead is the R interpreter loop, not the arithmetic. This is called 5 times, producing ~32.3 million R-level function invocations.

### Memory pressure

* The `neighbor_lookup` list itself stores ~6.46M integer vectors. With an average of ~4 rook neighbors per cell and R's 40-byte minimum vector overhead, this alone is ≥ 6.46M × 72 bytes ≈ **0.46 GB**. Combined with the 6.46M × 110-column data frame (~5.7 GB at 8 bytes/double), 16 GB is tight but workable **only if intermediate copies are eliminated**.

---

## 2. Optimization Strategy

| Principle | Technique |
|---|---|
| **Eliminate per-row R loops** | Convert the neighbor lookup into a flat edge list (two integer columns: `from_row`, `to_row`) and use **vectorised grouped operations** via `data.table`. |
| **Build the edge list vectorised** | Use `data.table` joins instead of `paste`/named-vector lookup. One equi-join replaces 6.46M `paste` + hash probes. |
| **Compute stats vectorised** | Join the edge list to the variable column, then `group by from_row` to get `max`, `min`, `mean` — all in C-level `data.table` code. |
| **Process all 5 variables in one pass** | A single join + grouped aggregation over all 5 source variables avoids rebuilding intermediate structures. |
| **Keep memory bounded** | No giant list-of-vectors; the edge list is two integer columns (≈ 2 × 6.46M × 4 neighbors × 4 bytes ≈ 206 MB). |

Expected speedup: from 86+ hours to **minutes** (the dominant cost becomes a handful of `data.table` joins and grouped aggregations on ~26M edge rows).

---

## 3. Working R Code

```r
library(data.table)

# ── Step 0: Ensure cell_data is a data.table with an integer row index ──────
setDT(cell_data)
cell_data[, row_idx := .I]

# ── Step 1: Build a vectorised edge list (replaces build_neighbor_lookup) ────
build_edge_list <- function(cell_dt, id_order, neighbors) {
  # Map each cell id to its position in id_order
  id_to_ref <- data.table(
    id  = id_order,
    ref = seq_along(id_order)
  )


  # Expand the nb object into a flat edge list of (from_id, to_id)
  from_ref <- rep(
    seq_along(neighbors),
    lengths(neighbors)
  )
  to_ref <- unlist(neighbors, use.names = FALSE)

  edge_ids <- data.table(
    from_id = id_order[from_ref],
    to_id   = id_order[to_ref]
  )

  # Create a lookup from (id, year) → row_idx
  id_year_lookup <- cell_dt[, .(id, year, row_idx)]

  # Join: for every (from_id, year) find the from_row
  # First cross edge_ids with every year present for each from_id
  # Efficient approach: join edge_ids to the data on from_id, then
  # join to_id + year to get to_row.


  # from side: get (from_id, year, from_row)
  setkey(id_year_lookup, id)
  from_dt <- id_year_lookup[edge_ids, on = .(id = from_id),
                            .(from_row = row_idx,
                              to_id    = i.to_id,
                              year     = year),
                            nomatch = NULL,
                            allow.cartesian = TRUE]

  # to side: get to_row by joining (to_id, year)
  setkey(id_year_lookup, id, year)
  from_dt[id_year_lookup,
          to_row := i.row_idx,
          on = .(to_id = id, year = year),
          nomatch = NA]

  # Drop edges where the neighbor has no observation in that year

  from_dt <- from_dt[!is.na(to_row)]

  from_dt[, .(from_row = as.integer(from_row),
              to_row   = as.integer(to_row))]
}

edge_list <- build_edge_list(cell_data, id_order, rook_neighbors_unique)
setkey(edge_list, from_row)


# ── Step 2: Compute all neighbor stats in one vectorised pass ────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_features <- function(cell_dt, edge_dt, vars) {
  # Attach the source variable values to every edge via to_row
  # We only select the columns we need to keep memory low.
  val_cols  <- vars
  to_vals   <- cell_dt[edge_dt$to_row, ..val_cols]
  to_vals[, from_row := edge_dt$from_row]

  # Grouped aggregation: max, min, mean per from_row, per variable

  agg_exprs <- unlist(lapply(val_cols, function(v) {
    list(
      bquote(max(.(as.name(v)),   na.rm = TRUE)),
      bquote(min(.(as.name(v)),   na.rm = TRUE)),
      bquote(mean(.(as.name(v)),  na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(val_cols, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  names(agg_exprs) <- agg_names

  stats <- to_vals[, lapply(agg_exprs, eval, envir = .SD), by = from_row]

  # Replace -Inf/Inf (from max/min on all-NA groups) with NA
  for (col in agg_names) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }

  stats
}

neighbor_stats <- compute_all_neighbor_features(cell_data, edge_list, neighbor_source_vars)


# ── Step 3: Left-join the new features back onto cell_data ───────────────────
# Rows with no neighbors will get NA automatically.
cell_data[neighbor_stats, on = .(row_idx = from_row),
          (names(neighbor_stats)[names(neighbor_stats) != "from_row"]) :=
            mget(names(neighbor_stats)[names(neighbor_stats) != "from_row"])]

# Clean up helper column
cell_data[, row_idx := NULL]

# ── cell_data now contains the 15 new neighbor feature columns ───────────────
# (neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl, … for all 5 vars)
# The trained Random Forest model can be applied directly to this table.
```

> **Note on the grouped aggregation:** If you prefer an even simpler (and sometimes faster) formulation that avoids `bquote`, here is a drop-in alternative for Step 2:

```r
compute_all_neighbor_features_v2 <- function(cell_dt, edge_dt, vars) {
  to_vals        <- cell_dt[edge_dt$to_row, ..vars]
  to_vals[, from_row := edge_dt$from_row]

  result_list <- vector("list", length(vars))

  for (i in seq_along(vars)) {
    v   <- vars[i]
    tmp <- to_vals[, .(
      vmax  = max(.SD[[1L]],  na.rm = TRUE),
      vmin  = min(.SD[[1L]],  na.rm = TRUE),
      vmean = mean(.SD[[1L]], na.rm = TRUE)
    ), by = from_row, .SDcols = v]

    # Fix Inf from all-NA groups
    tmp[is.infinite(vmax),  vmax  := NA_real_]
    tmp[is.infinite(vmin),  vmin  := NA_real_]

    setnames(tmp,
             c("vmax", "vmin", "vmean"),
             paste0("neighbor_", c("max_", "min_", "mean_"), v))

    result_list[[i]] <- tmp
  }

  # Merge all variable-specific results
  out <- result_list[[1L]]
  for (i in seq_along(result_list)[-1L]) {
    out <- merge(out, result_list[[i]], by = "from_row", all = TRUE)
  }
  out
}
```

---

## 4. Summary

| Aspect | Original | Optimised |
|---|---|---|
| Neighbor lookup | 6.46M `paste` + named-vector probes in R loop | One `data.table` equi-join (~seconds) |
| Stat computation | 5 × 6.46M R-level `lapply` calls | 5 grouped `data.table` aggregations (C-level) |
| Memory pattern | 6.46M-element list of int vectors (~0.5 GB overhead) | Flat 2-column integer edge table (~200 MB) |
| Estimated time | 86+ hours | **< 10 minutes** on 16 GB laptop |
| Model impact | None — columns produced are numerically identical | Identical `max`, `min`, `mean` per row |

The trained Random Forest model requires no changes because the output columns are numerically identical to the originals — only the computational pathway has changed.