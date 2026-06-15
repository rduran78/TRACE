 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which runs an R-level `lapply` over **~6.46 million rows**, each iteration performing character coercion, `paste`, and named-vector lookups. This produces a list of 6.46M integer vectors. Then `compute_neighbor_stats` iterates over that list again 5 times (once per variable), each time subsetting with per-element R loops. The combined cost:

| Source | Why it's slow |
|---|---|
| `build_neighbor_lookup` | 6.46M iterations of `paste()` + named-vector lookup (`idx_lookup[neighbor_keys]`). Named vector lookup is O(n) hash probing per call, and `paste` allocates millions of small strings. Estimated: **~60-70 hours alone**. |
| `compute_neighbor_stats` | 6.46M R-level iterations × 5 variables. Each iteration subsets a numeric vector, removes NAs, computes max/min/mean. Estimated: **~15-20 hours**. |
| Memory | Storing a 6.46M-element list of integer vectors is large (~10+ GB with R list overhead), risking swap on a 16 GB machine. |

The fundamental problem: the operation is **embarrassingly vectorizable** but the code uses **scalar R loops** over millions of rows.

---

## Optimization Strategy

### 1. Replace the row-level list with a sparse adjacency matrix approach using `data.table` joins

Instead of building a per-row neighbor list, we:

1. **Expand** the `nb` object into an edge-list of `(cell_id, neighbor_cell_id)` — only ~1.37M edges.
2. **Cross-join with years** to get `(cell_id, year, neighbor_cell_id, year)` — ~1.37M × 28 = ~38.5M rows, but this is a simple equi-join, not a Cartesian product, because both sides share the same year.
3. **Join** against the data to pull neighbor values.
4. **Aggregate** (max, min, mean) grouped by `(cell_id, year)`.

This replaces 6.46M R-level iterations with **vectorized `data.table` grouped aggregation** on ~38.5M rows — typically finishing in **seconds to a few minutes**.

### 2. Memory management

- The edge list is ~1.37M × 2 integers ≈ 11 MB.
- After year-expansion and join: ~38.5M rows × a few columns ≈ < 2 GB.
- Well within 16 GB.

### 3. Preserve the estimand

The new code computes **exactly** the same `max`, `min`, and `mean` of rook-neighbor values per `(cell_id, year)`, excluding `NA` neighbors (same semantics as the original). The trained Random Forest model is never touched.

**Expected runtime: ~2–10 minutes total** (down from 86+ hours).

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# Step 1: Convert spdep nb object to an edge list (done once)
# ---------------------------------------------------------------
# rook_neighbors_unique is a list where element i contains the
# indices (into id_order) of rook neighbors of cell id_order[i].
# id_order is the vector of cell IDs corresponding to each nb element.

build_edge_list <- function(id_order, neighbors) {
  # neighbors is an nb object (list of integer vectors of neighbor indices)
  from <- rep(
    seq_along(neighbors),
    times = lengths(neighbors)
  )
  to <- unlist(neighbors, use.names = FALSE)
  data.table(
    id          = id_order[from],
    neighbor_id = id_order[to]
  )
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
# edge_dt has columns: id, neighbor_id  (~1.37M rows)

# ---------------------------------------------------------------
# Step 2: Convert cell_data to data.table (if not already)
# ---------------------------------------------------------------
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure keyed for fast joins
setkey(cell_data, id, year)

# ---------------------------------------------------------------
# Step 3: For each neighbor source variable, compute neighbor
#         max, min, mean via a single vectorized join + aggregate
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_neighbor_features_fast <- function(cell_data, edge_dt, var_name) {
  # Subset to only the columns we need for the join
  # cell_data must have: id, year, <var_name>
  lookup_cols <- c("id", "year", var_name)
  lookup_dt   <- cell_data[, ..lookup_cols]

  # Rename for neighbor side of join
  setnames(lookup_dt, old = c("id", var_name),
           new = c("neighbor_id", "nval"))

  # Join edges with years:
  #   edge_dt (id, neighbor_id) joined to lookup_dt (neighbor_id, year, nval)
  #   This gives us (id, neighbor_id, year, nval) for every
  #   cell-year-neighbor combination.
  setkey(lookup_dt, neighbor_id, year)
  setkey(edge_dt, neighbor_id)

  # Expand edges × years by joining on neighbor_id

  joined <- lookup_dt[edge_dt, on = "neighbor_id", allow.cartesian = TRUE, nomatch = NA]
  # joined now has columns: neighbor_id, year, nval, id
  # (id comes from edge_dt via the join)

  # Remove rows where year is NA (shouldn't happen) or nval is NA
  joined <- joined[!is.na(nval)]

  # Aggregate: for each (id, year), compute max, min, mean of nval
  agg <- joined[, .(
    nmax  = max(nval),
    nmin  = min(nval),
    nmean = mean(nval)
  ), by = .(id, year)]

  # Rename columns to match original naming convention
  # Original code creates columns like: ntl_neighbor_max, ntl_neighbor_min, ntl_neighbor_mean
  new_names <- paste0(var_name, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
  setnames(agg, old = c("nmax", "nmin", "nmean"), new = new_names)

  agg
}

# ---------------------------------------------------------------
# Step 4: Compute and merge all neighbor features into cell_data
# ---------------------------------------------------------------
for (var_name in neighbor_source_vars) {
  cat("Computing neighbor stats for:", var_name, "\n")

  agg <- compute_neighbor_features_fast(cell_data, edge_dt, var_name)

  # Remove old columns if they exist (idempotent re-runs)
  old_cols <- paste0(var_name, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
  for (col in old_cols) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }

  # Merge back onto cell_data
  cell_data <- merge(cell_data, agg, by = c("id", "year"), all.x = TRUE)

  cat("  Done. Rows:", nrow(cell_data), "\n")
}

# Re-key after all merges
setkey(cell_data, id, year)

cat("All neighbor features computed.\n")
cat("Dimensions:", nrow(cell_data), "x", ncol(cell_data), "\n")
```

---

## Why This Is Correct and Safe

| Concern | How it's handled |
|---|---|
| **Irregular topology / masked cells** | The edge list is built directly from the `nb` object — only actual rook-neighbor pairs appear. No rectangular-grid assumption is made. |
| **Missing neighbors (boundary cells)** | If a cell has no neighbors in `edge_dt`, the `merge(..., all.x = TRUE)` leaves `NA` — identical to the original `c(NA, NA, NA)` return. |
| **NA variable values** | `joined <- joined[!is.na(nval)]` filters them out before aggregation, matching the original `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]`. |
| **Numerical equivalence** | `max`, `min`, `mean` are computed on exactly the same sets of values. The estimand is preserved bit-for-bit (within floating-point associativity). |
| **Trained RF model untouched** | No model code is invoked. Only feature columns are computed. Column names follow the same convention. |
| **Memory on 16 GB laptop** | Peak is ~38.5M rows × ~4 columns of the join table ≈ 1.2 GB, plus `cell_data` itself (~5 GB). Total well under 16 GB. |
| **Runtime** | `data.table` keyed joins and grouped aggregation on ~38.5M rows: typically **2–10 minutes** total for all 5 variables. |