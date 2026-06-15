 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### Local Inefficiency
- `paste(data$id, data$year, sep = "_")` creates ~6.46 million strings to build `idx_lookup`. This is done once, so it's O(n) — acceptable in isolation.
- But inside the `lapply` over all 6.46M rows, `paste(neighbor_cell_ids, data$year[i], sep = "_")` is called per row, generating ~1.37 billion string constructions and hash lookups total (avg ~4 neighbors × 6.46M rows × string paste + named-vector lookup).

### Broader Algorithmic Problem
The **entire design is row-wise**: for each of 6.46M rows, it discovers neighbors via string keys. This is fundamentally an **equi-join** problem (match on `year` and neighbor `id`), which can be solved in one vectorized pass using `data.table` keyed joins — eliminating all per-row string work, all `lapply` overhead, and all named-vector hash lookups.

Furthermore, `compute_neighbor_stats` iterates over the lookup list again per variable (5 times), each time doing per-row subsetting. This can be replaced by a single grouped aggregation.

**Estimated complexity comparison:**

| Step | Current | Proposed |
|---|---|---|
| Build neighbor lookup | O(n × k) string ops in R loop | Eliminated |
| Compute stats (per var) | O(n) list subset | O(n) vectorized grouped op |
| Total string constructions | ~1.37B | 0 |
| Total R-level iterations | ~6.46M × lapply + 6.46M × 5 vars | 0 (vectorized) |

## Optimization Strategy

1. **Expand the neighbor list into an edge table** (`data.table` with columns `id`, `neighbor_id`) — done once, ~1.37M rows.
2. **Join the edge table to the panel** on `id` + `year` to get each row's neighbor row indices — one keyed join, fully vectorized.
3. **Join again** to pull neighbor values — one keyed join per variable (or all at once).
4. **Grouped aggregation** (`[, .(max, min, mean), by = .(id, year)]`) replaces all per-row `lapply` work.
5. **Merge results back** to the main panel.

This preserves the exact numerical estimand (max, min, mean of non-NA rook-neighbor values per cell-year) and does not touch the trained Random Forest model.

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 0.  Inputs assumed to exist:
#       cell_data            — data.frame or data.table, ~6.46M rows
#       id_order             — integer/numeric vector of cell IDs (length 344,208)
#       rook_neighbors_unique — spdep nb object (list of length 344,208)
#       neighbor_source_vars  — c("ntl","ec","pop_density","def","usd_est_n2")
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# 1.  Build a directed edge table from the nb object  (once, ~1.37M rows)
# ──────────────────────────────────────────────────────────────────────
build_edge_table <- function(id_order, nb_obj) {
  # nb_obj[[i]] contains integer indices into id_order for the i-th cell
  # We need (focal_id, neighbor_id) pairs
  n <- length(nb_obj)
  from_idx <- rep(seq_len(n), lengths(nb_obj))
  to_idx   <- unlist(nb_obj)

  # Remove the spdep "0 = no neighbors" sentinel if present
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edges <- build_edge_table(id_order, rook_neighbors_unique)
cat("Edge table rows:", nrow(edges), "\n")

# ──────────────────────────────────────────────────────────────────────
# 2.  Convert panel to data.table and key it
# ──────────────────────────────────────────────────────────────────────
if (!is.data.table(cell_data)) cell_data <- as.data.table(cell_data)
setkey(cell_data, id, year)

# ──────────────────────────────────────────────────────────────────────
# 3.  For each source variable, compute neighbor max / min / mean
#     via a single keyed join + grouped aggregation, then merge back.
# ──────────────────────────────────────────────────────────────────────
compute_and_add_all_neighbor_features <- function(dt, edges, vars) {


  # Unique (id, year) pairs — one per panel row
  # We expand each row by its neighbors using the edge table.

  # Step A: attach year to edges by joining on focal id
  #   For every (id, year) row in dt, replicate its edges.
  #   This is the "big" join: ~6.46M × avg_degree ≈ 25.8M rows
  #   but it is vectorized and data.table handles it efficiently.

  # We only need id, year, and the source vars from dt for the neighbor side.
  # For the focal side we only need id and year.

  # Focal side: get unique (id, year) — but every row is unique by (id, year)
  # so we can use dt directly.

  # Create the join table: focal (id, year) × neighbor_id
  # edges has (id, neighbor_id).  We need to cross with years.
  # Efficient approach: join dt[, .(id, year)] to edges on id.

  focal <- dt[, .(id, year)]
  setkey(edges, id)
  # This expands each (id, year) by its neighbor count
  expanded <- edges[focal, on = "id", allow.cartesian = TRUE, nomatch = NULL]
  # expanded has columns: id, neighbor_id, year
  # "id" is the focal cell, "neighbor_id" is the neighbor, "year" is the focal year

  cat("Expanded join table rows:", nrow(expanded), "\n")

  # Step B: pull neighbor values by joining on (neighbor_id, year) → dt
  # We need the source variable columns from the neighbor rows.
  # Rename for the join:
  setnames(expanded, "neighbor_id", "n_id")

  # Prepare a lookup keyed on (id, year) with only needed columns
  lookup_cols <- c("id", "year", vars)
  nbr_vals <- dt[, ..lookup_cols]
  setnames(nbr_vals, "id", "n_id")
  setkey(nbr_vals, n_id, year)

  # Join to get neighbor variable values
  setkey(expanded, n_id, year)
  expanded <- nbr_vals[expanded, on = c("n_id", "year"), nomatch = NA]
  # Now expanded has: n_id, year, <vars>, id  (id = focal cell)

  # Step C: grouped aggregation — max, min, mean per (id, year) per variable
  # Build aggregation expressions dynamically
  agg_exprs <- unlist(lapply(vars, function(v) {
    vn <- as.name(v)
    list(
      bquote(as.numeric(max(.(vn), na.rm = TRUE))),
      bquote(as.numeric(min(.(vn), na.rm = TRUE))),
      bquote(as.numeric(mean(.(vn), na.rm = TRUE)))
    )
  }))

  agg_names <- unlist(lapply(vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  names(agg_exprs) <- agg_names

  # Evaluate
  agg_call <- as.call(c(as.name("list"), agg_exprs))
  stats <- expanded[, eval(agg_call), by = .(id, year)]

  # Replace Inf / -Inf (from max/min of zero-length after na.rm) with NA
  for (col in agg_names) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }

  cat("Stats table rows:", nrow(stats), "\n")

  # Step D: merge back to the main panel
  setkey(stats, id, year)
  setkey(dt, id, year)

  # Remove old neighbor columns if they exist (idempotent re-runs)
  old_cols <- intersect(agg_names, names(dt))
  if (length(old_cols)) dt[, (old_cols) := NULL]

  dt <- stats[dt, on = c("id", "year")]

  return(dt)
}

# ──────────────────────────────────────────────────────────────────────
# 4.  Run it
# ──────────────────────────────────────────────────────────────────────
cell_data <- compute_and_add_all_neighbor_features(
  cell_data, edges, neighbor_source_vars
)

# ──────────────────────────────────────────────────────────────────────
# 5.  Verify column names match what the trained RF model expects.
#     The naming convention above produces:
#       neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl, ...
#     If your trained model expects a different naming scheme (e.g.,
#     ntl_neighbor_max), rename here:
# ──────────────────────────────────────────────────────────────────────
# Example renaming (uncomment and adapt if needed):
# setnames(cell_data, "neighbor_max_ntl", "ntl_neighbor_max")

cat("Done. cell_data now has", ncol(cell_data), "columns and",
    nrow(cell_data), "rows.\n")
```

## Memory-Constrained Variant

If the expanded join table (~25.8M rows × several columns) strains 16 GB RAM, process variables one at a time:

```r
compute_neighbor_features_lowmem <- function(dt, edges, vars) {
  focal <- dt[, .(id, year)]
  setkey(edges, id)
  expanded_base <- edges[focal, on = "id", allow.cartesian = TRUE, nomatch = NULL]
  setnames(expanded_base, "neighbor_id", "n_id")
  setkey(expanded_base, n_id, year)

  for (v in vars) {
    cat("Processing variable:", v, "\n")

    # Minimal lookup: only (n_id, year, value)
    nbr <- dt[, .(n_id = id, year, val = get(v))]
    setkey(nbr, n_id, year)

    # Join
    tmp <- nbr[expanded_base, on = c("n_id", "year"), nomatch = NA]

    # Aggregate
    agg <- tmp[, {
      vv <- val[!is.na(val)]
      if (length(vv) == 0L) list(nb_max = NA_real_, nb_min = NA_real_, nb_mean = NA_real_)
      else list(nb_max = max(vv), nb_min = min(vv), nb_mean = mean(vv))
    }, by = .(id, year)]

    new_names <- paste0("neighbor_", c("max_", "min_", "mean_"), v)
    setnames(agg, c("nb_max", "nb_min", "nb_mean"), new_names)

    # Merge back
    old_cols <- intersect(new_names, names(dt))
    if (length(old_cols)) dt[, (old_cols) := NULL]

    setkey(agg, id, year)
    setkey(dt, id, year)
    dt <- agg[dt, on = c("id", "year")]

    rm(nbr, tmp, agg); gc()
  }

  return(dt)
}

# Usage:
cell_data <- compute_neighbor_features_lowmem(cell_data, edges, neighbor_source_vars)
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| String constructions | ~1.37 billion | 0 |
| R-level loop iterations | ~38.8M (6.46M build + 6.46M × 5 stats) | 0 |
| Core algorithm | Per-row string hash lookup | Vectorized keyed equi-join + grouped agg |
| Estimated runtime | 86+ hours | **Minutes** (data.table join + group-by) |
| Numerical result | max/min/mean of non-NA neighbor vals | **Identical** |
| RF model impact | N/A | **None — untouched** |
| Peak RAM (low-mem variant) | ~similar | ~4–6 GB for largest intermediate |