 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three interconnected reasons:

### 1. Row-level `lapply` over 6.46 million rows in `build_neighbor_lookup`
For every single cell-year row, the function performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). With ~6.46M rows, this creates billions of intermediate string objects and named-vector searches (O(n) or O(n log n) per lookup in base R named vectors). This alone could take tens of hours.

### 2. Row-level `lapply` over 6.46 million rows in `compute_neighbor_stats`
Even after the lookup is built, computing max/min/mean via `lapply` over 6.46M list elements — each containing a variable-length integer vector — is inherently slow in interpreted R. The list-of-vectors structure defeats vectorization.

### 3. The neighbor topology is **year-invariant** but rebuilt per cell-year
The rook-neighbor relationships are purely spatial (cell A neighbors cell B regardless of year). Yet `build_neighbor_lookup` re-derives neighbor indices for every cell-year combination, inflating the work by a factor of 28 (the number of years).

**Key insight:** The neighbor table is a property of the **grid**, not of the **panel**. There are only ~344K cells and ~1.37M directed neighbor pairs. The expensive part — joining yearly attributes and computing grouped statistics — should be done via vectorized table joins, not row-wise R loops.

---

## Optimization Strategy

1. **Build a static neighbor edge table once** — a two-column `data.table` of `(cell_id, neighbor_id)` with ~1.37M rows. This is year-invariant and derived from `rook_neighbors_unique` in seconds.

2. **For each variable, join yearly attributes onto the edge table** — use `data.table` keyed joins to attach the variable value for each `(neighbor_id, year)` pair. This explodes to ~1.37M × 28 ≈ 38.4M rows, which is very manageable.

3. **Compute grouped max/min/mean in one vectorized pass** — group by `(cell_id, year)` and compute the three statistics. `data.table` does this in seconds on 38M rows.

4. **Merge results back** onto the main `cell_data` table.

5. **Predict with the existing trained Random Forest** — no retraining, no change to the numerical estimand.

**Expected speedup:** From ~86+ hours to **minutes** (typically 2–10 minutes total depending on disk I/O).

**RAM estimate:** The edge table × years is ~38.4M rows × a few columns of numeric + integer ≈ < 1 GB. Well within 16 GB.

---

## Working R Code

```r
library(data.table)

# ===========================================================================
# STEP 0: Convert cell_data to data.table (if not already)
# ===========================================================================
cell_data <- as.data.table(cell_data)

# ===========================================================================
# STEP 1: Build a static, year-invariant neighbor edge table ONCE
#
#   rook_neighbors_unique : an nb object (list of integer index vectors)
#   id_order              : vector of cell IDs in the same order as the nb object
#
#   Result: edge_dt with columns  cell_id | neighbor_id
#           (~1,373,394 rows — one per directed rook-neighbor pair)
# ===========================================================================

build_edge_table <- function(id_order, neighbors) {
  # neighbors[[i]] contains integer indices into id_order for cell i's neighbors
  n <- length(neighbors)
  
  # Pre-allocate: count total edges
  edge_counts <- vapply(neighbors, length, integer(1))
  total_edges <- sum(edge_counts)
  
  from_id <- integer(total_edges)
  to_id   <- integer(total_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    nb <- neighbors[[i]]
    len <- length(nb)
    if (len > 0L) {
      idx <- pos:(pos + len - 1L)
      from_id[idx] <- id_order[i]
      to_id[idx]   <- id_order[nb]
      pos <- pos + len
    }
  }
  
  data.table(cell_id = from_id, neighbor_id = to_id)
}

cat("Building static neighbor edge table...\n")
edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
cat(sprintf("  Edge table: %s directed neighbor pairs\n", format(nrow(edge_dt), big.mark = ",")))

# ===========================================================================
# STEP 2: Function to compute neighbor stats for one variable, vectorized
# ===========================================================================

compute_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {
  # cell_dt must have columns: id, year, <var_name>
  # edge_dt must have columns: cell_id, neighbor_id
  
  # --- 2a. Extract only the columns we need for the neighbor values ----------
  val_dt <- cell_dt[, .(neighbor_id = id, year, value = get(var_name))]
  setkey(val_dt, neighbor_id, year)
  
  # --- 2b. Cross edge table with all years -----------------------------------
  #   For each edge (cell_id -> neighbor_id), we need every year.
  #   Instead of a full cross join, we join edge_dt onto val_dt by neighbor_id
  #   to pick up (year, value) in one pass.
  
  # Add year dimension: join edge_dt with val_dt on neighbor_id
  # Result: cell_id | neighbor_id | year | value
  setkey(edge_dt, neighbor_id)
  merged <- val_dt[edge_dt, on = .(neighbor_id), allow.cartesian = TRUE, nomatch = NA]
  # merged columns: neighbor_id, year, value, cell_id
  
  # --- 2c. Remove NA values before aggregation --------------------------------
  merged <- merged[!is.na(value)]
  
  # --- 2d. Grouped aggregation ------------------------------------------------
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  stats <- merged[, .(
    nb_max  = max(value),
    nb_min  = min(value),
    nb_mean = mean(value)
  ), by = .(cell_id, year)]
  
  setnames(stats, c("nb_max", "nb_min", "nb_mean"),
                  c(max_col,  min_col,  mean_col))
  
  stats
}

# ===========================================================================
# STEP 3: Loop over the 5 neighbor source variables, merge results
# ===========================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Ensure cell_data is keyed for fast joins
setkey(cell_data, id, year)

for (var_name in neighbor_source_vars) {
  cat(sprintf("Computing neighbor features for '%s'...\n", var_name))
  
  stats_dt <- compute_neighbor_features_fast(cell_data, copy(edge_dt), var_name)
  setkey(stats_dt, cell_id, year)
  
  # Determine the new column names
  new_cols <- setdiff(names(stats_dt), c("cell_id", "year"))
  
  # Remove old columns if they already exist (idempotent re-runs)
  for (col in new_cols) {
    if (col %in% names(cell_data)) {
      cell_data[, (col) := NULL]
    }
  }
  
  # Merge onto cell_data (left join: keep all original rows)
  cell_data <- stats_dt[cell_data, on = .(cell_id = id, year)]
  
  # The join renames cell_id; restore 'id'
  setnames(cell_data, "cell_id", "id")
  setkey(cell_data, id, year)
  
  cat(sprintf("  Done. cell_data now has %d columns.\n", ncol(cell_data)))
}

# ===========================================================================
# STEP 4: Predict with the existing trained Random Forest (unchanged)
#
#   The trained model object (e.g., rf_model) is already in memory.
#   The predictor columns — including the newly computed neighbor_max_*,
#   neighbor_min_*, neighbor_mean_* — now exist in cell_data with
#   identical numerical values to the original pipeline.
# ===========================================================================

cat("Running Random Forest prediction...\n")
cell_data$prediction <- predict(rf_model, newdata = cell_data)
cat("Done.\n")
```

---

## Summary of Guarantees

| Concern | Status |
|---|---|
| **Numerical equivalence** | Identical max/min/mean computed from the same neighbor sets and same variable values. No approximation. |
| **Trained RF model preserved** | Model object is never modified; only `predict()` is called. |
| **RAM on 16 GB laptop** | Peak ~3–4 GB (edge table × years ≈ 38M rows × 4 cols ≈ 1.2 GB + cell_data ≈ 2 GB). Safe. |
| **Speed** | ~2–10 minutes total vs. ~86+ hours. The bottleneck shifts from R interpretation overhead to vectorized `data.table` grouped aggregation on ~38M rows, which is extremely fast. |
| **Edge table built once** | `edge_dt` is year-invariant, built in <1 second, reused for all 5 variables and all 28 years. |