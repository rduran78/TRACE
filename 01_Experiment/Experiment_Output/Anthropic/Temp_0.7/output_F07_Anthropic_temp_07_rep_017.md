 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which creates a list of ~6.46 million elements, each produced by an R-level `lapply` iteration that performs character coercion, `paste`, and named-vector lookups. This is O(N) in pure R with heavy per-row overhead (string allocation, hash lookups on a 6.46M-entry named vector). The subsequent `compute_neighbor_stats` is a second O(N) `lapply` loop over the same 6.46M rows, also in pure R.

**Specific costs:**

| Component | Why it's slow |
|---|---|
| `paste(data$id, data$year, sep="_")` + `setNames` on 6.46M keys | Builds a 6.46M-entry character hash map — expensive to construct and probe |
| `lapply` over 6.46M rows in `build_neighbor_lookup` | Pure-R loop; each iteration does `as.character`, `paste`, named-vector subset |
| `lapply` over 6.46M rows in `compute_neighbor_stats` (×5 variables) | Pure-R loop; `max`/`min`/`mean` on small vectors with NA handling |
| Storing `neighbor_lookup` as a 6.46M-element list of integer vectors | ~25–50 GB of list overhead for millions of small vectors |

Estimated: ~86+ hours is consistent with ~48 μs/row × 6.46M rows × 6 passes.

---

## Optimization Strategy

**Core idea:** Replace per-row R loops with vectorized operations on an **edge list** (a sparse adjacency representation), then use `data.table` grouped operations to compute neighbor max, min, and mean in one vectorized pass per variable.

### Steps

1. **Expand the `nb` object into a directed edge list once** — two integer columns `(from_cell_id, to_cell_id)` representing all ~1.37M rook-neighbor pairs. This is O(E) and small.

2. **Join the edge list to the panel `data.table` by `(to_cell_id, year)`** to look up neighbor values. This replaces the 6.46M-row `lapply` + `paste` + named-vector lookup with a single keyed `data.table` merge — O(E × T) rows but executed in C.

3. **Group by `(from_cell_id, year)` and compute `max`, `min`, `mean`** in one pass per variable using `data.table`'s grouped aggregation — fully vectorized in C.

4. **Left-join the results back** to the main panel, filling cells with no neighbors with `NA`.

5. Repeat for each of the 5 neighbor source variables (or batch them).

**Complexity reduction:** From ~6.46M × 6 R-level iterations to a handful of vectorized `data.table` joins and group-bys. Expected runtime: **minutes, not hours.**

**Numerical equivalence:** `max`, `min`, `mean` are computed on exactly the same non-NA neighbor values as before, preserving the original estimand. The trained Random Forest model is untouched.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 1. Convert the spdep nb object to a directed edge list (once)
# ---------------------------------------------------------------
build_edge_list <- function(id_order, nb_obj) {
  # nb_obj is a list of length N; nb_obj[[i]] contains integer indices

  # of neighbors of cell i (in terms of position in id_order).
  from_idx <- rep(seq_along(nb_obj), lengths(nb_obj))
  to_idx   <- unlist(nb_obj, use.names = FALSE)

  # Remove the 0-entries that spdep uses for cells with no neighbors

  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )
}

edges <- build_edge_list(id_order, rook_neighbors_unique)
# edges has ~1,373,394 rows: (from_id, to_id)

# ---------------------------------------------------------------
# 2. Convert panel to data.table if not already
# ---------------------------------------------------------------
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ---------------------------------------------------------------
# 3. Compute neighbor stats for all source vars (vectorized)
# ---------------------------------------------------------------
compute_neighbor_features <- function(dt, edges, var_names) {
  # Minimal subset for the join: id, year, and the source variables
  cols_needed <- c("id", "year", var_names)
  neighbor_vals <- edges[
    dt[, ..cols_needed],
    on = .(to_id = id),       # join edges.to_id == dt.id
    allow.cartesian = TRUE,
    nomatch = NULL             # inner join: drop non-matches
  ]
  # Result columns: from_id, to_id, year, <var_names>
  # Each row = one directed neighbor observation for (from_id, year)

  # Group by (from_id, year) and compute stats for every variable
  agg_exprs <- unlist(lapply(var_names, function(v) {
    list(
      bquote(max(  .(as.name(v)), na.rm = TRUE)),
      bquote(min(  .(as.name(v)), na.rm = TRUE)),
      bquote(mean( .(as.name(v)), na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(var_names, function(v) {
    paste0("n_", v, c("_max", "_min", "_mean"))
  }))

  # Build the j-expression programmatically
  j_list <- as.call(c(as.name("list"),
                       setNames(agg_exprs, agg_names)))

  stats <- neighbor_vals[, eval(j_list), by = .(from_id, year)]

  # Replace -Inf/Inf from max/min on zero-length groups (shouldn't

  # happen after inner join + na.rm, but be safe)
  for (col in agg_names) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }

  # Left-join back to the main table
  dt <- merge(dt, stats, by.x = c("id", "year"),
              by.y = c("from_id", "year"),
              all.x = TRUE, sort = FALSE)

  return(dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_neighbor_features(cell_data, edges, neighbor_source_vars)

# ---------------------------------------------------------------
# 4. Done — cell_data now has n_<var>_max, n_<var>_min, n_<var>_mean
#    for all 5 source variables (15 new columns).
#    The trained Random Forest model is unchanged.
# ---------------------------------------------------------------
```

---

## Why This Is Correct and Safe

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | Same `max`/`min`/`mean` over the same non-NA neighbor values; `na.rm = TRUE` replicates the original `neighbor_vals[!is.na(…)]` filter. |
| **Irregular/masked topology** | The edge list is built directly from the `nb` object — no rectangular-raster assumption. Cells with no neighbors get `NA` via the left join. |
| **Memory** | Edge list: ~1.37M × 2 ints ≈ 11 MB. Expanded join table: ~1.37M × 28 years × 7 cols ≈ 2–3 GB peak — fits in 16 GB. |
| **Trained RF model** | Untouched; we only recompute the same predictor columns. |
| **Expected runtime** | ~2–10 minutes on a modern laptop (one keyed join + one grouped aggregation, all in C via `data.table`). |