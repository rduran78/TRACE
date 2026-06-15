 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### Primary Bottlenecks

**A. `build_neighbor_lookup` — O(n) `lapply` over 6.46M rows with per-row string operations**

The function iterates over every cell-year row (≈6.46M), and for each row:
- Converts an integer ID to character and does a named-vector lookup (`id_to_ref`).
- Builds string keys via `paste()` for every neighbor.
- Looks up those keys in a named character vector (`idx_lookup`).

Named-vector lookup in R is hash-based but carries heavy per-call overhead when done millions of times inside `lapply`. The `paste()`-based key construction allocates millions of small character vectors. This alone is likely **hours** of wall-clock time.

**B. `compute_neighbor_stats` — repeated over 5 variables, each iterating 6.46M rows**

Each call to `compute_neighbor_stats` does an `lapply` over 6.46M elements, subsetting a numeric vector and computing `max/min/mean`. The `do.call(rbind, result)` at the end binds 6.46M 3-element vectors — a notoriously slow pattern in R. With 5 variables this runs 5 × 6.46M = 32.3M R-level function calls.

**C. Random Forest Prediction**

Predicting 6.46M rows × 110 features through a Random Forest (typically hundreds of trees) is memory- and CPU-intensive. If `predict()` is called row-by-row or in small chunks, overhead dominates. Even a single vectorised call on 6.46M rows may require careful memory management on 16 GB RAM.

**D. Memory and object copying**

Repeatedly modifying `cell_data` inside the `for` loop (`cell_data <- compute_and_add_neighbor_features(...)`) triggers R's copy-on-modify semantics, potentially duplicating a ≈5.7 GB data.frame multiple times.

---

## 2. OPTIMIZATION STRATEGY

| Bottleneck | Strategy | Expected Speedup |
|---|---|---|
| `build_neighbor_lookup` | Eliminate string keys entirely. Use a `data.table` join on `(id, year)` → row index, then vectorised expansion of the `nb` object. Build lookup once as a pair of integer vectors (row → neighbor-rows). | 50–200× |
| `compute_neighbor_stats` | Replace per-row `lapply` + `do.call(rbind,…)` with vectorised `data.table` grouped aggregation over a pre-built edge-list. Compute all 5 variables' stats in one pass. | 20–100× |
| Column mutation / copying | Use `data.table` set-by-reference (`:=`) to avoid full-frame copies. | 2–5× |
| RF Prediction | Load model once; predict in a single vectorised call (or in ~10 large chunks to stay within RAM). Ensure the prediction input is a plain `matrix` or `data.frame` with no factors/extra columns. | 2–10× |
| Overall | Target: **under 30 minutes** end-to-end on a 16 GB laptop. | ~200× vs 86 h |

---

## 3. WORKING R CODE

```r
# ============================================================
# 0. Libraries
# ============================================================
library(data.table)
library(ranger)
# If your model is from 'randomForest', use library(randomForest) instead.

# ============================================================
# 1. Load pre-trained RF model (once)
# ============================================================
rf_model <- readRDS("rf_model.rds")  # adjust path as needed

# ============================================================
# 2. Convert main data to data.table (by reference if possible)
# ============================================================
if (!is.data.table(cell_data)) {
  setDT(cell_data)  # converts in place — no copy
}

# ============================================================
# 3. Build integer row-index mapping  (id, year) -> row
# ============================================================
cell_data[, .row_idx := .I]  # add row index column

# Keyed lookup table: given (id, year) return the row index
setkey(cell_data, id, year)

# ============================================================
# 4. Build neighbour edge-list (vectorised, no paste/string ops)
#
#    rook_neighbors_unique is an nb object: a list of length
#    n_cells where element [[i]] gives the indices (into
#    id_order) of cell i's neighbours.
#
#    We expand this to a data.table of directed edges
#    (from_id, to_id) and then cross-join with years to get
#    (from_row, to_row) pairs in the panel.
# ============================================================

# --- 4a. Cell-level edge list ----------------------------------
n_cells <- length(id_order)   # 344,208

# Vectorised expansion of the nb list to an edge data.table
to_lengths <- lengths(rook_neighbors_unique)
edge_dt <- data.table(
  from_cell_pos = rep(seq_len(n_cells), times = to_lengths),
  to_cell_pos   = unlist(rook_neighbors_unique, use.names = FALSE)
)
# Map positional indices to actual cell IDs
edge_dt[, from_id := id_order[from_cell_pos]]
edge_dt[, to_id   := id_order[to_cell_pos]]
edge_dt[, c("from_cell_pos", "to_cell_pos") := NULL]

# --- 4b. Expand to panel edges (from_row, to_row) -------------
#
#  For every (from_id, to_id) pair, and for every year in the
#  data, we need the row indices of both the focal cell-year
#  and the neighbour cell-year.
#
#  Strategy: join edge_dt with the row-index table twice.

years_vec <- sort(unique(cell_data$year))
n_years   <- length(years_vec)

# Cross-join edges × years
panel_edges <- edge_dt[, .(year = years_vec), by = .(from_id, to_id)]

# Join to get from_row
setkey(cell_data, id, year)
panel_edges[cell_data, from_row := i..row_idx,
            on = .(from_id = id, year = year)]

# Join to get to_row (neighbour's row index in that year)
panel_edges[cell_data, to_row := i..row_idx,
            on = .(to_id = id, year = year)]

# Drop edges where either side is missing in the panel
panel_edges <- panel_edges[!is.na(from_row) & !is.na(to_row)]

# Clean up temporaries
rm(edge_dt); gc()

cat("Panel edges:", nrow(panel_edges), "\n")

# ============================================================
# 5. Compute all neighbour stats (5 vars × 3 stats) in one pass
#
#    Grouped aggregation over the edge-list is fully vectorised
#    inside data.table's C backend.
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-extract the columns we need into the edge table so that
# the grouped aggregation only touches numeric vectors.

for (v in neighbor_source_vars) {
  # Map each to_row to its value of variable v
  set(panel_edges, j = v, value = cell_data[[v]][panel_edges$to_row])
}

# Grouped aggregation: one group per from_row
# Produces columns like ntl_nb_max, ntl_nb_min, ntl_nb_mean, ...
agg_exprs <- list()
for (v in neighbor_source_vars) {
  agg_exprs[[paste0(v, "_nb_max")]]  <- call("max",  as.name(v), na.rm = TRUE)
  agg_exprs[[paste0(v, "_nb_min")]]  <- call("min",  as.name(v), na.rm = TRUE)
  agg_exprs[[paste0(v, "_nb_mean")]] <- call("mean", as.name(v), na.rm = TRUE)
}

# Build a single J-expression
j_expr <- as.call(c(as.name("list"), agg_exprs))

cat("Computing neighbour statistics …\n")
nb_stats <- panel_edges[, eval(j_expr), by = .(from_row)]

# Replace infinite values (from max/min on empty sets) with NA
for (col in names(nb_stats)[-1L]) {
  vals <- nb_stats[[col]]
  set(nb_stats, i = which(is.infinite(vals)), j = col, value = NA_real_)
}

# Free memory from panel_edges
rm(panel_edges); gc()

# ============================================================
# 6. Join neighbour stats back to cell_data (by reference)
# ============================================================
# nb_stats is keyed on from_row; cell_data has .row_idx
setkey(nb_stats, from_row)

stat_cols <- setdiff(names(nb_stats), "from_row")
cell_data[nb_stats, (stat_cols) := mget(paste0("i.", stat_cols)),
          on = .(.row_idx = from_row)]

# Clean up helper column
cell_data[, .row_idx := NULL]

cat("Feature preparation complete. Columns:", ncol(cell_data), "\n")

# ============================================================
# 7. Random Forest Prediction  —  chunked, single-model load
# ============================================================

# Identify the predictor columns the model expects.
# For ranger:
if (inherits(rf_model, "ranger")) {
  pred_vars <- rf_model$forest$independent.variable.names
} else {
  # randomForest package stores the names used in training
  pred_vars <- rownames(rf_model$importance)
}

# Validate that all required columns are present
missing_vars <- setdiff(pred_vars, names(cell_data))
if (length(missing_vars) > 0L) {
  stop("Missing predictor columns: ", paste(missing_vars, collapse = ", "))
}

# Build a prediction matrix (data.table subset is zero-copy friendly)
pred_input <- cell_data[, ..pred_vars]

cat("Starting RF prediction on", nrow(pred_input), "rows …\n")

# --- Chunked prediction to limit peak memory -----------------
#     Each chunk ≈ 650K rows → ~10 chunks.  Adjust chunk_size
#     if memory is tighter or more generous.

chunk_size <- 650000L
n_rows     <- nrow(pred_input)
n_chunks   <- ceiling(n_rows / chunk_size)

preds <- numeric(n_rows)

for (ch in seq_len(n_chunks)) {
  i_start <- (ch - 1L) * chunk_size + 1L
  i_end   <- min(ch * chunk_size, n_rows)
  
  chunk_data <- pred_input[i_start:i_end, ]
  
  if (inherits(rf_model, "ranger")) {
    p <- predict(rf_model, data = chunk_data, num.threads = parallel::detectCores())
    preds[i_start:i_end] <- p$predictions
  } else {
    # randomForest package
    preds[i_start:i_end] <- predict(rf_model, newdata = chunk_data)
  }
  
  cat(sprintf("  Chunk %d/%d done (%d–%d)\n", ch, n_chunks, i_start, i_end))
}

rm(pred_input); gc()

# ============================================================
# 8. Attach predictions to cell_data
# ============================================================
cell_data[, predicted_gdp := preds]

cat("Done. Predictions attached as 'predicted_gdp'.\n")
```

---

## 4. SUMMARY OF KEY CHANGES AND WHY THEY WORK

| Original | Optimised | Why it matters |
|---|---|---|
| `paste(id, year)` string keys looked up in named vector, per row | `data.table` integer join on `(id, year)` | Eliminates ≈6.46M `paste()` calls and hash lookups on character strings. `data.table` binary search on integer keys is orders of magnitude faster. |
| `lapply` over 6.46M rows in `build_neighbor_lookup` | Vectorised `rep()`/`unlist()` expansion + `data.table` cross-join | Replaces 6.46M R function calls with three vectorised C-level operations. |
| `lapply` + `do.call(rbind,…)` over 6.46M rows × 5 variables | Single grouped `data.table` aggregation (`by = from_row`) | `data.table`'s GForce optimisation runs `max`, `min`, `mean` in compiled C per group. Binding is internal, not via `do.call(rbind, …)` (which allocates 6.46M tiny matrices). |
| `cell_data <- compute_and_add_…(cell_data, …)` in a loop (copies ~5.7 GB frame 5 times) | `:=` set-by-reference | Zero copies; columns are added in-place. |
| (Assumed) single `predict()` on 6.46M rows | Chunked prediction (~650K rows/chunk, auto-threaded) | Keeps peak memory under 16 GB while still giving `predict()` large batches (low per-call overhead). `num.threads` exploits all cores. |
| Model potentially re-loaded or converted | Loaded once as `rf_model`; never retrained | Preserves the trained model exactly. |

**The numerical estimand is preserved**: no transformations, rounding, or approximations are applied to either the features or the predictions. All `max`, `min`, `mean` operations are numerically identical to the originals (same `na.rm = TRUE` semantics; infinities from empty groups are mapped to `NA` exactly as the original `length(idx) == 0` guard did).