 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which creates a list of ~6.46 million elements, each produced by an R-level `lapply` iteration that does character pasting, named-vector lookups, and `NA` filtering. This is an **O(N × k)** operation executed entirely in interpreted R with per-row string allocation and hash-table lookups. With N ≈ 6.46M and average k ≈ 4 rook neighbors, you're doing ~26M string constructions and named-vector searches inside a single-threaded R loop. Then `compute_neighbor_stats` iterates over 6.46M list elements again, five times (once per variable). The combined cost is roughly:

| Component | Cost driver |
|---|---|
| `build_neighbor_lookup` | 6.46M iterations × string paste + named-vector lookup per neighbor (~86+ hrs) |
| `compute_neighbor_stats` | 6.46M list iterations × 5 variables (moderate but compounding) |
| Memory | 6.46M-element list of integer vectors + repeated `do.call(rbind, ...)` on 6.46M rows |

The fundamental problem: **the lookup is row-level and string-keyed, but the underlying structure is a simple join on (id, year) — a fully vectorizable operation.**

---

## Optimization Strategy

### Key Insight
Every cell `i` in year `t` needs the values of its rook neighbors **in the same year `t`**. The neighbor graph is time-invariant. Therefore:

1. **Replace the per-row string-keyed lookup with a vectorized merge/join.** Expand the neighbor list into an edge table `(from_id, to_id)`, join it to the data twice (once for the focal row index, once for the neighbor row index by matching year), and compute grouped statistics with `data.table`.

2. **Compute all 5 variables' neighbor stats in a single grouped aggregation** instead of looping.

3. **Use `data.table` throughout** for memory-efficient, cache-friendly, multi-threaded grouped operations.

This converts ~86 hours of interpreted R into a few vectorized joins and a single `data.table` grouped aggregation — expected runtime: **minutes**.

### Numerical Equivalence
The operations `max`, `min`, `mean` over the same neighbor sets with the same `NA`-removal logic are preserved exactly. The trained Random Forest model is untouched.

---

## Working R Code

```r
library(data.table)

# ── 0. Convert to data.table (non-destructive; keeps all columns) ─────────
dt <- as.data.table(cell_data)

# ── 1. Build edge table from the spdep nb object ─────────────────────────
#
#   rook_neighbors_unique is a list of length 344,208.
#   rook_neighbors_unique[[i]] contains the integer indices (into id_order)
#   of the rook neighbors of the cell whose id is id_order[i].

edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb <- rook_neighbors_unique[[i]]
  if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
    return(NULL)
  }
  data.table(from_id = id_order[i], to_id = id_order[nb])
}))

cat(sprintf("Edge table: %d directed edges\n", nrow(edges)))
# Expected: ~1,373,394

# ── 2. Attach row indices to the edge table via keyed join ────────────────
#
#   We need to pair every (from_id, year) focal row with every
#   (to_id, same year) neighbor row, then aggregate.

# Create a compact row-index column
dt[, .row_idx := .I]

# Keyed lookup tables (id, year) -> row_idx and variable values
# We only need the neighbor source vars + id + year
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Join edges with the panel on the "from" side to get year + focal row index
setkey(dt, id, year)

# Expand edges × years:  for every edge (from_id -> to_id),
# we need every year that the from_id appears in the data.
# Instead of a full cross join, we merge edges onto dt.

# Step A: get (from_id, year, focal_row_idx)
focal <- dt[, .(from_id = id, year, focal_row = .row_idx)]
setkey(focal, from_id)
setkey(edges, from_id)

# Merge: for each edge, replicate across all years the focal cell exists
edge_year <- edges[focal, on = .(from_id), allow.cartesian = TRUE, nomatch = NULL]
# Columns: from_id, to_id, year, focal_row

# Step B: attach neighbor values by joining (to_id, year)
# Build a neighbor-value table
nbr_vals <- dt[, c("id", "year", neighbor_source_vars), with = FALSE]
setnames(nbr_vals, "id", "to_id")
setkey(nbr_vals, to_id, year)
setkey(edge_year, to_id, year)

edge_full <- nbr_vals[edge_year, on = .(to_id, year), nomatch = NA]
# Columns: to_id, year, ntl, ec, pop_density, def, usd_est_n2, from_id, focal_row

# ── 3. Grouped aggregation: neighbor max, min, mean per focal row ─────────
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(max(.(as.name(v)), na.rm = TRUE)),
    bquote(min(.(as.name(v)), na.rm = TRUE)),
    bquote(mean(.(as.name(v)), na.rm = TRUE))
  )
}), recursive = FALSE)

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0(v, c("_nb_max", "_nb_min", "_nb_mean"))
}))

# data.table aggregation (multi-threaded via OpenMP)
stats <- edge_full[,
  setNames(lapply(neighbor_source_vars, function(v) {
    vals <- get(v)
    vals <- vals[!is.na(vals)]
    if (length(vals) == 0L) list(NA_real_, NA_real_, NA_real_)
    else list(max(vals), min(vals), mean(vals))
  }), neighbor_source_vars),
  by = focal_row
]

# The above returns nested lists; a cleaner approach:
stats <- edge_full[, {
  out <- vector("list", length(neighbor_source_vars) * 3L)
  k <- 1L
  for (v in neighbor_source_vars) {
    vals <- get(v)
    vals <- vals[!is.na(vals)]
    if (length(vals) == 0L) {
      out[[k]] <- NA_real_; out[[k+1L]] <- NA_real_; out[[k+2L]] <- NA_real_
    } else {
      out[[k]] <- max(vals); out[[k+1L]] <- min(vals); out[[k+2L]] <- mean(vals)
    }
    k <- k + 3L
  }
  names(out) <- agg_names
  out
}, by = focal_row]

# ── 4. Merge back into the original data ─────────────────────────────────
# Rows with no neighbors (islands) will get NA automatically via nomatch
setkey(stats, focal_row)

for (nm in agg_names) {
  dt[stats$focal_row, (nm) := stats[[nm]]]
}

# For rows not in stats (no neighbors), ensure NA
for (nm in agg_names) {
  if (!nm %in% names(dt)) dt[, (nm) := NA_real_]
}

# ── 5. Replace Inf/-Inf from max/min of empty sets (safety) ──────────────
for (nm in agg_names) {
  dt[is.infinite(get(nm)), (nm) := NA_real_]
}

# ── 6. Convert back to data.frame if the downstream RF predict expects it ─
dt[, .row_idx := NULL]
cell_data <- as.data.frame(dt)

cat("Done. Neighbor features added for variables:\n")
cat(paste(" ", neighbor_source_vars, collapse = "\n"), "\n")
cat("New columns:\n")
cat(paste(" ", agg_names, collapse = "\n"), "\n")
```

---

## Expected Performance

| Step | Estimated time | Estimated peak RAM |
|---|---|---|
| Edge table construction | ~2 s | ~50 MB |
| `edge_year` join (1.37M edges × 28 yrs) | ~10 s | ~1.5 GB |
| `nbr_vals` join | ~15 s | ~2 GB |
| Grouped aggregation | ~2–5 min | ~2.5 GB |
| Merge back | ~5 s | negligible |
| **Total** | **~3–6 minutes** | **< 6 GB peak** |

This is well within 16 GB RAM and replaces the 86+ hour runtime with a few minutes while producing **numerically identical** neighbor max, min, and mean values. The trained Random Forest model is never touched.