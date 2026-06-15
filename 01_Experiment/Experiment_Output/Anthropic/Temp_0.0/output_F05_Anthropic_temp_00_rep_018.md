 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Inside the Loop

`build_neighbor_lookup` iterates over all ~6.46 million rows. For **each row**, it:

1. Converts `data$id[i]` to character and looks it up in `id_to_ref` — **6.46M character coercions + named-vector lookups**.
2. Subsets `id_order[neighbors[[ref_idx]]]` — fine, but done row-by-row.
3. Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` — **allocates a new character vector per row** (~4 neighbor keys × 6.46M rows ≈ 25.8M string constructions).
4. Looks up `idx_lookup[neighbor_keys]` — **named-vector lookup via string matching on a 6.46M-element named vector**, repeated 6.46M times.

The string-key lookup on a named vector of length 6.46M is approximately O(N) per probe (R's named vector lookup is hash-based, but building and probing a 6.46M-entry hash 6.46M times with freshly-allocated string keys is extremely expensive). Total: **~25.8 billion character comparisons worth of hashing work**.

### The Deeper Structural Insight

The neighbor relationship is **time-invariant**: cell A's neighbors are the same in every year. The lookup is joining `(cell_id, year)` pairs, but since the panel is balanced (every cell appears in every year), the **row offset between any cell and its neighbor is deterministic given the data's sort order**. We don't need string keys at all.

### Why `compute_neighbor_stats` Is Also Suboptimal

After building the lookup, `compute_neighbor_stats` runs an `lapply` over 6.46M entries, extracting `vals[idx]` and computing `max/min/mean` in R-level loops. This is repeated 5 times (once per variable). With vectorized/matrix operations, all 5 variables can be processed simultaneously.

---

## Optimization Strategy

1. **Eliminate all string-key construction.** Build a direct integer-index mapping exploiting the balanced panel structure.
2. **Replace the row-level `lapply` with a sparse-matrix multiplication** (or equivalent vectorized operation). A row-normalized adjacency matrix times a column of values gives the neighbor mean; similar constructions give max and min.
3. **Process all 5 variables in one pass** where possible (mean via sparse matrix multiply is trivially vectorized; max/min require grouped operations).
4. **Use `data.table` for grouped operations** on the neighbor edge list to compute max/min efficiently.

Expected speedup: from ~86 hours to **~2–10 minutes**.

---

## Working R Code

```r
library(data.table)
library(Matrix)

# ==============================================================
# STEP 0: Ensure data is a data.table, sorted for fast indexing
# ==============================================================
cell_dt <- as.data.table(cell_data)

# Create a unique integer index for each cell id, preserving id_order mapping
# id_order is the vector of cell IDs aligned with rook_neighbors_unique
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

# Ensure consistent year ordering
cell_dt[, row_idx := .I]  # original row index

# Build a fast (id, year) -> row_idx lookup via data.table keying
cell_dt[, id_chr := as.character(id)]
setkey(cell_dt, id_chr, year)

# ==============================================================
# STEP 1: Build a directed edge list (cell_pos_from, cell_pos_to)
#         from the nb object — done ONCE, no year dimension
# ==============================================================
build_edge_list <- function(nb_obj) {
  # nb_obj is a list of integer vectors (neighbor positions)
  from <- rep(seq_along(nb_obj), lengths(nb_obj))
  to   <- unlist(nb_obj, use.names = FALSE)
  data.table(from_pos = from, to_pos = to)
}

edge_dt <- build_edge_list(rook_neighbors_unique)
# from_pos and to_pos index into id_order
# Map to actual cell IDs
edge_dt[, from_id := as.character(id_order[from_pos])]
edge_dt[, to_id   := as.character(id_order[to_pos])]

cat("Edge list built:", nrow(edge_dt), "directed edges\n")

# ==============================================================
# STEP 2: Expand edge list across years and join to row indices
#         This creates (row_i, row_j) pairs: row_i's neighbor is row_j
# ==============================================================
years <- sort(unique(cell_dt$year))

# Create lookup: (id_chr, year) -> row_idx
row_lookup <- cell_dt[, .(id_chr, year, row_idx)]
setkey(row_lookup, id_chr, year)

# Expand edges across all years using a cross join
# ~1.37M edges × 28 years = ~38.5M rows — fits in memory
edge_year <- CJ(edge_idx = seq_len(nrow(edge_dt)), year = years)
edge_year[, from_id := edge_dt$from_id[edge_idx]]
edge_year[, to_id   := edge_dt$to_id[edge_idx]]

# Join to get row indices for "from" (the focal cell-year)
setkey(edge_year, from_id, year)
edge_year[row_lookup, from_row := i.row_idx, on = .(from_id = id_chr, year)]

# Join to get row indices for "to" (the neighbor cell-year)
setkey(edge_year, to_id, year)
edge_year[row_lookup, to_row := i.row_idx, on = .(to_id = id_chr, year)]

# Drop any edges where either cell-year is missing (boundary / unbalanced)
edge_year <- edge_year[!is.na(from_row) & !is.na(to_row)]

cat("Expanded edge-year list:", nrow(edge_year), "rows\n")

# Keep only what we need
edge_year <- edge_year[, .(from_row, to_row)]

# ==============================================================
# STEP 3: Compute neighbor stats (max, min, mean) for each var
#         using vectorized data.table grouped operations
# ==============================================================
N <- nrow(cell_dt)

compute_neighbor_stats_fast <- function(cell_dt, edge_year, var_name) {
  # Extract neighbor values via vectorized indexing
  vals <- cell_dt[[var_name]]
  
  # Build a working table: for each (from_row), the neighbor's value
  work <- data.table(
    from_row = edge_year$from_row,
    nval     = vals[edge_year$to_row]
  )
  
  # Remove edges where neighbor value is NA
  work <- work[!is.na(nval)]
  
  # Grouped aggregation — extremely fast in data.table
  agg <- work[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = from_row]
  
  # Initialize result columns with NA
  max_col  <- rep(NA_real_, N)
  min_col  <- rep(NA_real_, N)
  mean_col <- rep(NA_real_, N)
  
  # Fill in computed values
  max_col[agg$from_row]  <- agg$nb_max
  min_col[agg$from_row]  <- agg$nb_min
  mean_col[agg$from_row] <- agg$nb_mean
  
  list(max_col = max_col, min_col = min_col, mean_col = mean_col)
}

# ==============================================================
# STEP 4: Loop over the 5 variables and attach columns
# ==============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Computing neighbor stats for:", var_name, "...\n")
  
  stats <- compute_neighbor_stats_fast(cell_dt, edge_year, var_name)
  
  # Use the same column naming convention as the original code
  # (adjust these names to match your existing trained RF model's expected features)
  max_name  <- paste0(var_name, "_neighbor_max")
  min_name  <- paste0(var_name, "_neighbor_min")
  mean_name <- paste0(var_name, "_neighbor_mean")
  
  set(cell_dt, j = max_name,  value = stats$max_col)
  set(cell_dt, j = min_name,  value = stats$min_col)
  set(cell_dt, j = mean_name, value = stats$mean_col)
  
  cat("  Done:", var_name, "\n")
}

# ==============================================================
# STEP 5: Restore original row order and clean up helper columns
# ==============================================================
setorder(cell_dt, row_idx)
cell_dt[, c("row_idx", "id_chr") := NULL]

# Convert back to data.frame if downstream code expects it
cell_data <- as.data.frame(cell_dt)

cat("All neighbor features computed.\n")
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | The `max`, `min`, and `mean` are computed over exactly the same set of neighbor values as the original code. The edge list is derived from the same `rook_neighbors_unique` nb object. The NA-handling logic is identical (drop NA neighbor values, return NA if no valid neighbors remain). |
| **Trained RF model** | No model retraining occurs. The output columns have the same values and are attached to `cell_data` for downstream `predict()` calls. Ensure column names match what the RF expects (adjust the `paste0` naming pattern if needed). |
| **Column names** | Adjust `max_name`, `min_name`, `mean_name` to match whatever naming convention your `compute_and_add_neighbor_features` originally produced. |

---

## Complexity Comparison

| Step | Original | Optimized |
|---|---|---|
| Build lookup | O(N) string pastes + O(N) hash build on 6.46M strings | O(E) integer edge list, one-time |
| Per-row neighbor resolution | 6.46M × (paste + hash probe on 6.46M keys) | One vectorized `data.table` join: O(E × Y) ≈ 38.5M rows |
| Stats computation | 6.46M R-level `lapply` calls × 5 vars | 5 × one `data.table` grouped aggregation on 38.5M rows |
| **Total estimated time** | **~86 hours** | **~2–10 minutes** |
| **Peak memory** | ~moderate (but slow) | ~3–5 GB for edge_year (fits in 16 GB) |

The fundamental shift is from **row-level string-key lookups** to a **single materialized edge list joined via integer indices**, with grouped aggregation handled by `data.table`'s radix-sort-based grouping engine.