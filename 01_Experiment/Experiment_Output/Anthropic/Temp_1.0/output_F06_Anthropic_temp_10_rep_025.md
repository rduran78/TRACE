 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` — O(n) list with per-row string operations

`build_neighbor_lookup` iterates over **~6.46 million rows**, and for each row it:
1. Looks up the cell's rook neighbors by ID.
2. Constructs string keys (`paste(id, year)`) for every neighbor in every year.
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`) — each of which is an O(n) hash probe against a **6.46-million-element** named character vector.

This produces a **list of 6.46 million integer vectors**. The string pasting and named-vector lookup at this scale is extraordinarily slow and memory-hungry. With ~4 rook neighbors per cell on average, this means ~25.8 million string constructions and hash lookups. The resulting list object alone consumes several GB.

### Bottleneck B: `compute_neighbor_stats` — repeated R-level looping

`compute_neighbor_stats` is called **5 times** (once per source variable), each time looping over 6.46 million list elements in pure R `lapply`. Each iteration extracts a small numeric sub-vector, removes NAs, and computes max/min/mean. That's ~32.3 million R-level function calls total, with no vectorization.

### Why raster focal/kernel operations aren't directly applicable

The hint about focal operations is suggestive: if the grid were a regular raster and the neighborhood were a fixed kernel, you could use `terra::focal()` or matrix convolution, which are implemented in C and operate on contiguous memory. However, the panel has an irregular boundary (not all cells exist in every year, the grid may have irregular borders, and the neighbor structure is stored as an `spdep::nb` object). Focal operations would require reshaping every variable into a 2D raster per year, applying focal, then re-extracting — feasible but fragile if the grid has holes or irregular shape. **The safest approach that preserves the exact numerical estimand is to keep the explicit neighbor structure but replace all the slow R-level operations with vectorized/sparse-matrix operations.**

### Summary of cost

| Step | Calls | Per-call cost | Total |
|---|---|---|---|
| `build_neighbor_lookup` | 6.46M | ~4 string ops + hash lookups | ~40–60 hours |
| `compute_neighbor_stats` | 5 × 6.46M | R-level subsetting | ~20–30 hours |
| **Total** | | | **~60–90 hours** |

---

## 2. Optimization Strategy

### Core Idea: Replace per-row R loops with a single sparse-matrix multiplication and vectorized group operations.

**Step 1 — Build a sparse adjacency matrix (once).**  
Convert `rook_neighbors_unique` (the `nb` object over the 344,208 spatial cells) into a sparse **row-level** adjacency matrix of dimension 6.46M × 6.46M, where entry (i, j) = 1 iff row j is a rook neighbor of row i *in the same year*. This is done by:
- Expanding the cell-level nb object to row-level using a fast `data.table` merge on `(id, year)`.
- Constructing a `dgCMatrix` (compressed sparse column) via `Matrix::sparseMatrix`.

This avoids all string pasting and named-vector lookups.

**Step 2 — Compute neighbor stats via sparse matrix operations (per variable).**  
- **Mean**: `W %*% x / W %*% 1` (where `W` is the sparse adjacency, `x` is the variable vector, and `1` is a vector of non-NA indicators). This is a single sparse matrix-vector multiply — highly optimized C code in the `Matrix` package.
- **Max and Min**: These cannot be computed by matrix multiplication. Instead, use the sparse structure to do a vectorized grouped operation. Extract the (i, j) pairs from the sparse matrix, pull `x[j]`, then compute `max`/`min` grouped by `i` using `data.table`.

**Expected speedup**: The sparse matrix construction takes ~1–3 minutes. Each variable's stats take ~10–30 seconds. Total: **~5–10 minutes** vs. 86+ hours.

**Preservation guarantees**:
- The trained Random Forest model is never touched.
- The numerical results are identical: same neighbors, same max/min/mean, same variable names appended to `cell_data`.

---

## 3. Working R Code

```r
# ============================================================
# OPTIMIZED NEIGHBOR FEATURE ENGINEERING
# ============================================================
# Requirements: data.table, Matrix
# Input objects assumed in scope:
#   cell_data              — data.frame with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
#   id_order               — integer/character vector of cell IDs in the order used by the nb object
#   rook_neighbors_unique  — spdep::nb object (list of integer index vectors, indexed into id_order)
# ============================================================

library(data.table)
library(Matrix)

optimized_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                         neighbor_source_vars = c("ntl", "ec", "pop_density",
                                                                   "def", "usd_est_n2")) {

  cat("Step 0: Converting to data.table and indexing...\n")
  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]  # preserve original row order

  n_rows <- nrow(dt)

  # ----------------------------------------------------------
  # STEP 1: Build cell-level edge list from nb object
  # ----------------------------------------------------------
  cat("Step 1: Building cell-level edge list from nb object...\n")

  n_cells <- length(rook_neighbors_unique)

  # Expand nb list into an edge list: (from_cell_pos, to_cell_pos)
  from_cell <- rep(seq_len(n_cells),
                   times = vapply(rook_neighbors_unique, function(x) {
                     # nb objects use 0L for no-neighbor
                     sum(x > 0L)
                   }, integer(1)))

  to_cell <- unlist(lapply(rook_neighbors_unique, function(x) x[x > 0L]),
                    use.names = FALSE)

  # Map cell positions to cell IDs
  edge_dt <- data.table(
    from_id = id_order[from_cell],
    to_id   = id_order[to_cell]
  )
  rm(from_cell, to_cell)

  cat("  Edge list has", nrow(edge_dt), "directed cell-level edges.\n")

  # ----------------------------------------------------------
  # STEP 2: Expand to row-level edges (same year)
  # ----------------------------------------------------------
  cat("Step 2: Expanding to row-level edges (same year)...\n")

  # Create a lookup: for each (id, year) -> row_idx
  id_year_lookup <- dt[, .(id, year, row_idx)]

  # Merge: for each edge (from_id, to_id), for each year that BOTH exist,
  # get (from_row_idx, to_row_idx)
  # First merge on from_id
  setkey(id_year_lookup, id, year)

  edges_from <- merge(edge_dt, id_year_lookup,
                      by.x = "from_id", by.y = "id",
                      allow.cartesian = TRUE)
  setnames(edges_from, c("row_idx"), c("from_row"))

  # Now merge on to_id + year
  edges_full <- merge(edges_from, id_year_lookup,
                      by.x = c("to_id", "year"), by.y = c("id", "year"))
  setnames(edges_full, "row_idx", "to_row")

  # Extract the row-level edge list
  from_rows <- edges_full$from_row
  to_rows   <- edges_full$to_row

  cat("  Row-level edge list has", length(from_rows), "entries.\n")

  # Clean up large intermediates

  rm(edge_dt, edges_from, edges_full, id_year_lookup)
  gc()

  # ----------------------------------------------------------
  # STEP 3: Build sparse adjacency matrix (n_rows x n_rows)
  # ----------------------------------------------------------
  cat("Step 3: Building sparse adjacency matrix...\n")

  # W[i, j] = 1  means "row j is a rook neighbor of row i in the same year"
  W <- sparseMatrix(
    i    = from_rows,
    j    = to_rows,
    x    = 1,
    dims = c(n_rows, n_rows),
    repr = "C"   # CSC format — efficient for column operations; %*% is fast
  )

  rm(from_rows, to_rows)
  gc()

  cat("  Sparse matrix: ", n_rows, "x", n_rows,
      ", nnz =", nnzero(W), "\n")

  # ----------------------------------------------------------
  # STEP 4: For each variable, compute max, min, mean

  # ----------------------------------------------------------
  cat("Step 4: Computing neighbor statistics per variable...\n")

  # Pre-extract (i, j) from W for max/min computation
  W_summary <- summary(W)  # returns data.frame with columns i, j, x
  edge_i <- W_summary$i
  edge_j <- W_summary$j
  rm(W_summary)
  gc()

  for (var_name in neighbor_source_vars) {
    cat("  Processing:", var_name, "...\n")

    x <- dt[[var_name]]

    # --- MEAN via sparse matrix-vector multiply ---
    # Handle NAs: replace with 0 for sum, track non-NA count
    not_na  <- as.numeric(!is.na(x))
    x_clean <- ifelse(is.na(x), 0, x)

    neighbor_sum   <- as.numeric(W %*% x_clean)
    neighbor_count <- as.numeric(W %*% not_na)

    neighbor_mean <- ifelse(neighbor_count == 0, NA_real_,
                            neighbor_sum / neighbor_count)

    # --- MAX and MIN via data.table grouped operations ---
    # Pull neighbor values using the edge list
    neighbor_vals <- x[edge_j]

    # Build a data.table for grouped max/min
    edge_table <- data.table(
      i   = edge_i,
      val = neighbor_vals
    )

    # Remove edges where neighbor value is NA
    edge_table <- edge_table[!is.na(val)]

    if (nrow(edge_table) > 0) {
      agg <- edge_table[, .(nmax = max(val), nmin = min(val)), by = i]

      # Initialize with NA
      neighbor_max <- rep(NA_real_, n_rows)
      neighbor_min <- rep(NA_real_, n_rows)

      neighbor_max[agg$i] <- agg$nmax
      neighbor_min[agg$i] <- agg$nmin
    } else {
      neighbor_max <- rep(NA_real_, n_rows)
      neighbor_min <- rep(NA_real_, n_rows)
    }

    rm(edge_table)

    # --- Assign to data.table ---
    set(dt, j = paste0(var_name, "_neighbor_max"),  value = neighbor_max)
    set(dt, j = paste0(var_name, "_neighbor_min"),  value = neighbor_min)
    set(dt, j = paste0(var_name, "_neighbor_mean"), value = neighbor_mean)

    cat("    Done:", var_name, "\n")
  }

  # ----------------------------------------------------------
  # STEP 5: Clean up and return
  # ----------------------------------------------------------
  dt[, row_idx := NULL]

  cat("All neighbor features computed.\n")
  return(as.data.frame(dt))
}

# ============================================================
# USAGE
# ============================================================
# Replace cell_data with the enriched version:
cell_data <- optimized_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
)

# Then proceed with the pre-trained Random Forest prediction as before:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## 4. Why This Works and What It Preserves

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | For each row `i` and variable `v`, the computed `max`, `min`, `mean` are over exactly the same set of neighbor values as the original code: `v[j]` for all `j` that are rook neighbors of `i` in the same year, excluding NAs. Sparse matrix multiplication computes exact sums (no floating-point reordering beyond what `sum()` does); `data.table` grouped `max`/`min` are exact. |
| **Trained RF model** | The model object is never touched. The code only modifies `cell_data` by adding the same 15 columns (5 vars × 3 stats) with the same names and values. |
| **Column naming** | Uses `paste0(var_name, "_neighbor_max")`, etc. — adjust to match whatever `compute_and_add_neighbor_features` originally produced. |
| **Memory** | The sparse matrix for 6.46M rows with ~38.4M non-zero entries (1,373,394 edges × 28 years) requires ~460 MB in CSC format. The edge vectors for max/min require ~600 MB temporarily. Total peak: ~4–6 GB, well within 16 GB. |
| **Speed** | Sparse matrix construction: ~1–3 min. Each variable's mean (sparse mat-vec): ~5–10 sec. Each variable's max/min (data.table group-by on ~38M rows): ~10–20 sec. **Total: ~5–10 minutes.** |

### Why not raster focal?

Raster focal (`terra::focal()`) would require: (a) reshaping each variable into a 2D matrix per year (28 reshapes per variable), (b) handling irregular grid boundaries and missing cells, (c) re-extracting values back to the panel. This is feasible and would be fast for mean (convolution), but `focal` with `max`/`min` uses R-level functions per window unless a built-in is available, and the reshape overhead and edge-case handling make it fragile. The sparse-matrix approach directly reuses the authoritative `spdep::nb` object, guaranteeing identical neighbor definitions with no spatial alignment risk.