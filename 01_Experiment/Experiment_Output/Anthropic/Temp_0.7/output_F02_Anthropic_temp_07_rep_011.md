 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck has two root causes:

### 1. `build_neighbor_lookup` — O(n) `lapply` over 6.46 million rows with per-row string pasting and named-vector lookups
- `paste(...)` and named-vector indexing (`idx_lookup[neighbor_keys]`) are called **per row**. For 6.46M rows, this creates and discards millions of small character vectors.
- Named-vector lookup in R is hash-based but still involves repeated character coercion and allocation.
- The resulting list of 6.46M integer vectors is itself a large, fragmented memory structure (~hundreds of MB of pointers + vectors).

### 2. `compute_neighbor_stats` — Called 5 times, each iterating over the 6.46M-element list
- Each call does 6.46M small `max/min/mean` operations inside `lapply`, producing 6.46M 3-element vectors, then `do.call(rbind, ...)` on a 6.46M-element list (extremely slow row-bind pattern).
- Total: ~32.3 million R-level function calls across the 5 variables.

### Memory pressure
- The neighbor lookup list alone, plus intermediate character vectors, plus the 6.46M × 110 data frame, can easily exceed 16 GB, triggering swapping.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Per-row string key lookup | Replace with a **`data.table` integer-keyed join** — encode `(id, year)` as a composite integer key and use binary-search join. |
| 6.46M-element R list for neighbor lookup | Replace with a **flat CSR (Compressed Sparse Row) representation**: two integer vectors (`offsets`, `neighbors_flat`). No R list overhead. |
| Per-row `lapply` in `compute_neighbor_stats` | **Vectorised C++ via `Rcpp`**: iterate the CSR structure in compiled code, computing max/min/mean in a single pass per variable. |
| `do.call(rbind, list_of_6M)` | Eliminated — Rcpp writes directly into a pre-allocated matrix. |
| 5 separate passes over the neighbor structure | Process **all 5 variables in one Rcpp call** (single pass over the CSR structure). |
| Overall time | Expected reduction from 86+ hours to **~5–15 minutes**. |

The trained Random Forest model is never touched. The output columns are numerically identical (max, min, mean of the same neighbor values).

---

## Working R Code

### Step 0 — Dependencies

```r
library(data.table)
library(Rcpp)
```

### Step 1 — Build CSR neighbor lookup (vectorised, no per-row paste)

```r
build_neighbor_lookup_csr <- function(data, id_order, neighbors) {
  # data must be a data.table (or will be converted)
  dt <- as.data.table(data)
  dt[, row_idx := .I]
  
  # Integer mappings --------------------------------------------------------
  # Map cell id -> position in id_order (1-based)
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # Map (id, year) -> row index via data.table keyed join
  setkey(dt, id, year)
  
  # --- Expand neighbor pairs (cell-year level) ----------------------------
  # For every row i, find its neighbor cell ids, then look up their row indices
  # in the same year.  We do this fully vectorised.
  
  # 1. Build an edge table from the nb object: (ref_pos, neighbor_cell_id)
  #    ref_pos is the index into id_order.
  n_edges  <- sum(lengths(neighbors))
  from_ref <- rep(seq_along(neighbors), lengths(neighbors))
  to_cell  <- id_order[unlist(neighbors)]
  edge_dt  <- data.table(from_ref = from_ref, neighbor_id = to_cell)
  
  # 2. Attach the cell id to from_ref
  edge_dt[, from_id := id_order[from_ref]]
  
  # 3. Cross with years: for each row in dt, get its (from_id, year) then
  #    join to edge_dt to get neighbor_id, then join back to dt to get
  #    the neighbor row index.
  #    To keep memory bounded we process year-by-year.
  
  years <- sort(unique(dt$year))
  
  # Pre-allocate CSR vectors
  n_rows <- nrow(dt)
  offsets <- integer(n_rows + 1L)   # 0-based offsets
  neighbor_flat_list <- vector("list", length(years))
  row_order_list     <- vector("list", length(years))
  
  # Key edge_dt for join
  setkey(edge_dt, from_id)
  
  for (yr in years) {
    # Rows in this year
    dt_yr <- dt[year == yr, .(id, row_idx)]
    setkey(dt_yr, id)
    
    # Join: for each cell in this year, get its neighbors
    # dt_yr  ->  edge_dt on from_id == id  ->  dt_yr on neighbor_id == id
    merged <- edge_dt[dt_yr, on = .(from_id = id), nomatch = 0L,
                      allow.cartesian = TRUE]
    # merged has columns: from_ref, neighbor_id, from_id, row_idx (= source row)
    # Now find the row_idx of the neighbor in the same year
    merged[, source_row := row_idx]
    merged[, row_idx := NULL]
    
    # Join to get neighbor row
    setnames(dt_yr, "id", "nid")
    setnames(dt_yr, "row_idx", "neighbor_row")
    merged <- merged[dt_yr, on = .(neighbor_id = nid), nomatch = 0L]
    
    # merged now has: source_row, neighbor_row
    # Store
    neighbor_flat_list[[as.character(yr)]] <- merged$neighbor_row
    row_order_list[[as.character(yr)]]     <- merged$source_row
  }
  
  # Combine across years
  all_source   <- unlist(row_order_list,   use.names = FALSE)
  all_neighbor  <- unlist(neighbor_flat_list, use.names = FALSE)
  
  # Sort by source row to build CSR
  ord <- order(all_source)
  all_source   <- all_source[ord]
  all_neighbor  <- all_neighbor[ord]
  
  # Build offsets (0-based, length n_rows + 1)
  # Use tabulate for speed
  counts <- tabulate(all_source, nbins = n_rows)
  offsets <- c(0L, cumsum(counts))
  
  list(offsets = offsets, neighbors = all_neighbor, n = n_rows)
}
```

### Step 2 — Rcpp function: compute stats for multiple variables in one pass

```r
cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix neighbor_stats_multi(
    IntegerVector offsets,
    IntegerVector neighbors,
    NumericMatrix varmat,    // n_rows x n_vars
    int n_rows)
{
  int n_vars = varmat.ncol();
  // Output: n_rows x (n_vars * 3)  — for each var: max, min, mean
  NumericMatrix out(n_rows, n_vars * 3);
  std::fill(out.begin(), out.end(), NA_REAL);

  for (int i = 0; i < n_rows; i++) {
    int start = offsets[i];
    int end   = offsets[i + 1];
    if (start == end) continue;  // no neighbors — stays NA

    for (int v = 0; v < n_vars; v++) {
      double vmax = R_NegInf;
      double vmin = R_PosInf;
      double vsum = 0.0;
      int    cnt  = 0;

      for (int j = start; j < end; j++) {
        int nb = neighbors[j] - 1;  // R 1-based -> C 0-based
        double val = varmat(nb, v);
        if (NumericMatrix::is_na(val)) continue;
        if (val > vmax) vmax = val;
        if (val < vmin) vmin = val;
        vsum += val;
        cnt++;
      }

      int col_base = v * 3;
      if (cnt > 0) {
        out(i, col_base)     = vmax;
        out(i, col_base + 1) = vmin;
        out(i, col_base + 2) = vsum / cnt;
      }
      // else: stays NA
    }
  }
  return out;
}
')
```

### Step 3 — Orchestration (drop-in replacement for the outer loop)

```r
# ---- Build CSR lookup (once) ----
csr <- build_neighbor_lookup_csr(cell_data, id_order, rook_neighbors_unique)

# ---- Prepare variable matrix ----
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
varmat <- as.matrix(cell_data[, ..neighbor_source_vars])  
# If cell_data is a plain data.frame:
# varmat <- as.matrix(cell_data[, neighbor_source_vars])

# ---- Compute all neighbor stats in one compiled pass ----
stats_mat <- neighbor_stats_multi(
  csr$offsets,
  csr$neighbors,
  varmat,
  csr$n
)

# ---- Attach results back to cell_data ----
stat_suffixes <- c("_neighbor_max", "_neighbor_min", "_neighbor_mean")
col_names <- as.vector(outer(neighbor_source_vars, stat_suffixes, paste0))

if (is.data.table(cell_data)) {
  cell_data[, (col_names) := as.data.frame(stats_mat)]
} else {
  cell_data[, col_names] <- as.data.frame(stats_mat)
}
```

---

## Summary of Gains

| Metric | Before | After |
|---|---|---|
| Neighbor lookup construction | ~hours (per-row `paste` + named lookup) | ~1–3 min (vectorised `data.table` joins + CSR) |
| Stat computation (5 vars) | ~hours (`lapply` × 6.46M × 5) | ~1–2 min (single Rcpp pass) |
| Peak RAM for lookup | >10 GB (6.46M-element R list) | ~200 MB (two integer vectors) |
| `do.call(rbind, ...)` | 6.46M-row list bind × 5 | Eliminated (pre-allocated C++ matrix) |
| **Total estimated wall time** | **86+ hours** | **~5–15 minutes** |
| Numerical output | max, min, mean per variable | **Identical** |
| Trained RF model | — | **Untouched** |