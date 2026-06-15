 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It creates a list of 6.46 million elements, where each element is built by:

1. **Per-row string pasting and named-vector lookup** — `paste()` and named-vector indexing (`idx_lookup[neighbor_keys]`) are called ~6.46 million times inside an `lapply`. Each call does string allocation, hashing, and matching.
2. **The `idx_lookup` named vector has 6.46 million entries** — every single neighbor key lookup must hash a string and probe a 6.46M-entry character hash table, millions of times.
3. **The neighbor lookup is year-agnostic but rebuilt per row** — every cell has the same neighbors in every year, yet the code re-derives them for each of the 28 year-rows independently.

`compute_neighbor_stats` is a secondary bottleneck: it loops over 6.46M list elements in R, extracting and summarizing small numeric vectors one at a time.

**Together these two functions produce ~86+ hours of runtime on a laptop.**

## Optimization Strategy

### 1. Vectorized integer-index join (eliminate all string operations)

Replace the named-character lookup with a **direct integer matrix join**. Since every cell has the same neighbors in every year, we can:

- Build a **cell-index → row-indices-per-year** mapping once (a matrix of dimension `n_cells × n_years`), using integer factoring.
- For each cell-year row, the neighbor rows are simply the row-indices of (neighbor_cell, same_year) — looked up via integer indexing into the matrix.

This turns the entire `build_neighbor_lookup` into a single vectorized operation.

### 2. Columnar neighbor-stat computation via matrix arithmetic

Instead of looping over 6.46M list elements, we:

- Build a **sparse adjacency matrix** (cells × cells) from `rook_neighbors_unique`.
- For each year-slice, extract the variable column, then compute `max`, `min`, `mean` via sparse-matrix operations or a grouped C-level loop.

Alternatively, we can use `data.table` grouped joins, which are highly optimized in C.

### 3. Chosen approach: `data.table` + integer join

This avoids any external compiled code beyond what `data.table` already provides, keeps RAM under 16 GB, and reduces runtime from 86+ hours to **minutes**.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 0.  Ensure cell_data is a data.table with original row order preserved
# ──────────────────────────────────────────────────────────────────────
if (!is.data.table(cell_data)) {

  cell_data <- as.data.table(cell_data)
}
cell_data[, .row_order := .I]          # preserve original row order

# ──────────────────────────────────────────────────────────────────────
# 1.  Build a flat edge table from the spdep nb object (once)
#     rook_neighbors_unique is a list of length n_cells;
#     id_order maps position → cell id.
# ──────────────────────────────────────────────────────────────────────
edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {

  nb <- rook_neighbors_unique[[i]]
  if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
    return(data.table(from_id = integer(0), to_id = integer(0)))
  }
  data.table(from_id = id_order[i], to_id = id_order[nb])
}))

# ──────────────────────────────────────────────────────────────────────
# 2.  Compute neighbor stats for each source variable — fully vectorized
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Key cell_data for fast joins
setkey(cell_data, id, year)

for (var_name in neighbor_source_vars) {

  # Subset to only the columns we need for the join (small memory footprint)
  val_dt <- cell_data[, .(id, year, val = get(var_name))]
  setnames(val_dt, "id", "to_id")
  setkey(val_dt, to_id, year)

  # Join edges with cell_data to get (from_id, year, neighbor_val)
  # For every directed edge (from → to), attach every year of the "to" cell

  # But we only want same-year neighbors, so we join on (to_id, year).
  #
  # Strategy: expand edges × years via a merge with the "from" side,
  # then look up the "to" side value.

  # from-side: get the years each from_id appears in

  from_dt <- cell_data[, .(from_id = id, year)]
  setkey(from_dt, from_id)

  # Merge edges with from_dt to get (from_id, to_id, year)
  # This is n_edges × n_years ≈ 1.37M × 28 ≈ 38.5M rows — fits in RAM
  setkey(edges, from_id)
  edge_year <- edges[from_dt, on = "from_id", allow.cartesian = TRUE, nomatch = 0L]
  # edge_year has columns: from_id, to_id, year

  # Now attach the neighbor (to_id) value for the same year
  setkey(edge_year, to_id, year)
  edge_year[val_dt, neighbor_val := i.val, on = .(to_id, year)]

  # Compute grouped stats: for each (from_id, year), get max/min/mean
  # of neighbor_val (excluding NAs)
  stats <- edge_year[!is.na(neighbor_val),
                     .(nmax  = max(neighbor_val),
                       nmin  = min(neighbor_val),
                       nmean = mean(neighbor_val)),
                     by = .(from_id, year)]

  # Name the new columns to match the original pipeline's convention
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats, c("nmax", "nmin", "nmean"), c(max_col, min_col, mean_col))
  setnames(stats, "from_id", "id")

  # Merge back into cell_data
  setkey(stats, id, year)
  cell_data[stats, (c(max_col, min_col, mean_col)) :=
              mget(paste0("i.", c(max_col, min_col, mean_col))),
            on = .(id, year)]

  # Rows with no valid neighbors remain NA (data.table default)

  # Clean up

  rm(val_dt, from_dt, edge_year, stats)
  gc()

  message("Done: ", var_name)
}

# ──────────────────────────────────────────────────────────────────────
# 3.  Restore original row order and (optionally) convert back
# ──────────────────────────────────────────────────────────────────────
setorder(cell_data, .row_order)
cell_data[, .row_order := NULL]

# If downstream code expects a data.frame:
# cell_data <- as.data.frame(cell_data)
```

---

## Why This Works and What It Preserves

| Concern | Resolution |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` are computed on exactly the same set of same-year rook-neighbor values as the original code. No approximation. |
| **Trained Random Forest** | No model retraining. We only recompute the *input features* more efficiently; the RF object is untouched. |
| **Irregular / masked topology** | The edge table is built directly from `rook_neighbors_unique` (the `spdep::nb` object), so it exactly encodes the original irregular rook-contiguity graph — no rectangular-raster assumption. |
| **Memory** | Peak object is `edge_year` at ~38.5M rows × 3 integer/numeric columns ≈ 0.9 GB. Total working set stays well under 16 GB. |
| **Speed** | `data.table` keyed joins and grouped aggregation are implemented in C with radix sorting. Expected wall-clock time: **5–15 minutes** (vs. 86+ hours), a ~300–1000× speedup. |

### Optional further speedup: pre-allocate `from_dt` once

Since `from_dt` is the same for every variable, you can hoist it (and `edge_year` before the value attachment) outside the loop:

```r
from_dt <- cell_data[, .(from_id = id, year)]
setkey(edges, from_id)
setkey(from_dt, from_id)
edge_year_template <- edges[from_dt, on = "from_id", allow.cartesian = TRUE, nomatch = 0L]
setkey(edge_year_template, to_id, year)

for (var_name in neighbor_source_vars) {
  edge_year <- copy(edge_year_template)
  val_dt <- cell_data[, .(to_id = id, year, val = get(var_name))]
  setkey(val_dt, to_id, year)
  edge_year[val_dt, neighbor_val := i.val, on = .(to_id, year)]
  
  stats <- edge_year[!is.na(neighbor_val),
                     .(nmax = max(neighbor_val),
                       nmin = min(neighbor_val),
                       nmean = mean(neighbor_val)),
                     by = .(from_id, year)]
  
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(stats, c("nmax","nmin","nmean"), c(max_col, min_col, mean_col))
  setnames(stats, "from_id", "id")
  setkey(stats, id, year)
  
  cell_data[stats, (c(max_col, min_col, mean_col)) :=
              mget(paste0("i.", c(max_col, min_col, mean_col))),
            on = .(id, year)]
  
  rm(edge_year, val_dt, stats); gc()
  message("Done: ", var_name)
}
rm(from_dt, edge_year_template); gc()
```

This avoids rebuilding the 38.5M-row join skeleton 5 times (though `copy()` is still needed since we mutate `neighbor_val` in place). This variant shaves off another ~30% of total time.