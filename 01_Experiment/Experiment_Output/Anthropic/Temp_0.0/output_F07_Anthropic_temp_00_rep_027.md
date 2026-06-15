 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation has two critical bottlenecks:

### 1. `build_neighbor_lookup` — O(n²)-like overhead
- **`lapply` over ~6.46 million rows**, each performing string pasting, named-vector lookups (`idx_lookup[neighbor_keys]`), and `NA` filtering.
- `idx_lookup` is a **named character vector** with ~6.46M entries. Each lookup via `idx_lookup[neighbor_keys]` does **linear hashing on character keys** — this is extremely slow at scale.
- The function builds ~6.46M list elements, each containing integer vectors. Memory allocation and GC pressure are enormous.

### 2. `compute_neighbor_stats` — repeated per variable
- Another `lapply` over 6.46M rows **per variable** (×5 variables = ~32.3M iterations).
- Each iteration subsets `vals[idx]`, removes NAs, and computes max/min/mean — all in interpreted R with no vectorization.

### 3. Combined effect
- ~6.46M list-element iterations for the lookup build.
- ~32.3M list-element iterations for stats.
- Estimated 86+ hours is consistent with character-key lookups and per-row R-level loops at this scale.

---

## Optimization Strategy

### A. Replace character-key lookup with integer indexing via `data.table`

Instead of building a named character vector of 6.46M entries and doing string-match lookups, use `data.table` keyed joins. Create an integer-indexed mapping from `(id, year)` → row number. This turns the lookup from O(n) string matching to O(1) hash-table lookup.

### B. Vectorize the neighbor lookup build using edge-list expansion

Convert the `nb` object into a flat edge list `(cell_id, neighbor_cell_id)`. Cross-join with years. Join against the row-index table. This replaces the 6.46M-iteration `lapply` with a single vectorized `data.table` merge — typically seconds instead of hours.

### C. Vectorize neighbor stats with grouped `data.table` aggregation

Once we have an edge list `(row_i, neighbor_row_j)`, computing neighbor max/min/mean is a single grouped aggregation per variable — no R-level loop at all.

### D. Preserve the trained model and numerical estimand

The output columns have the same names and identical numerical values (max, min, mean of the same neighbor sets). The Random Forest model sees the same feature matrix. Nothing changes except speed.

---

## Working R Code

```r
library(data.table)

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                        neighbor_source_vars) {

  # ---------------------------------------------------------------
  # 0.  Convert to data.table (by reference if already; copy if not)
  # ---------------------------------------------------------------
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  # Preserve original row order for downstream compatibility

  cell_data[, .row_id := .I]

  # ---------------------------------------------------------------
  # 1.  Build (id, year) → row_id mapping

  # ---------------------------------------------------------------
  row_map <- cell_data[, .(id, year, .row_id)]
  setkey(row_map, id, year)

  # ---------------------------------------------------------------
  # 2.  Convert nb object to a flat edge list of cell IDs
  #     nb object: list of length N_cells, each element is an

  #     integer vector of neighbor *indices* into id_order.
  # ---------------------------------------------------------------
  # Build edge list: from_id -> to_id (cell-level, time-invariant)
  edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb_idx <- rook_neighbors_unique[[i]]
    # spdep::nb encodes "no neighbors" as 0L (single element)
    nb_idx <- nb_idx[nb_idx != 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(from_id = id_order[i], to_id = id_order[nb_idx])
  }))
  # This should have ~1,373,394 rows (directed rook-neighbor pairs)

  cat("Edge list rows:", nrow(edge_list), "\n")

  # ---------------------------------------------------------------
  # 3.  Expand edge list across all years
  #     Result: (from_id, year, to_id) — one row per directed
  #     neighbor-pair-year.
  # ---------------------------------------------------------------
  years <- sort(unique(cell_data$year))
  # Cross join edge_list × years  (~1.37M × 28 ≈ 38.5M rows)
  # This fits comfortably in RAM (~1 GB for 3 integer columns)
  edges_by_year <- CJ_dt(edge_list, years)

  cat("Edges × years rows:", nrow(edges_by_year), "\n")

  # ---------------------------------------------------------------
  # 4.  Attach row indices for both the focal cell and the neighbor
  # ---------------------------------------------------------------
  # Focal cell row index
  setnames(edges_by_year, c("from_id", "to_id", "year"))
  edges_by_year[row_map, focal_row := i..row_id,
                on = .(from_id = id, year = year)]

  # Neighbor cell row index
  edges_by_year[row_map, nb_row := i..row_id,
                on = .(to_id = id, year = year)]

  # Drop edges where either focal or neighbor is missing
  edges_by_year <- edges_by_year[!is.na(focal_row) & !is.na(nb_row)]

  cat("Valid edges × years:", nrow(edges_by_year), "\n")

  # ---------------------------------------------------------------
  # 5.  Compute neighbor stats per variable — fully vectorized
  # ---------------------------------------------------------------
  for (var_name in neighbor_source_vars) {
    cat("Processing neighbor stats for:", var_name, "\n")

    # Pull the variable values and attach to neighbor rows
    edges_by_year[, nb_val := cell_data[[var_name]][nb_row]]

    # Grouped aggregation: max, min, mean per focal row
    stats <- edges_by_year[!is.na(nb_val),
                           .(nb_max  = max(nb_val),
                             nb_min  = min(nb_val),
                             nb_mean = mean(nb_val)),
                           by = focal_row]

    # Initialize columns with NA
    max_col  <- paste0("neighbor_max_",  var_name)
    min_col  <- paste0("neighbor_min_",  var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    cell_data[, (max_col)  := NA_real_]
    cell_data[, (min_col)  := NA_real_]
    cell_data[, (mean_col) := NA_real_]

    # Assign results
    cell_data[stats$focal_row, (max_col)  := stats$nb_max]
    cell_data[stats$focal_row, (min_col)  := stats$nb_min]
    cell_data[stats$focal_row, (mean_col) := stats$nb_mean]

    # Clean up temp column
    edges_by_year[, nb_val := NULL]

    cat("  Done:", var_name, "\n")
  }

  # ---------------------------------------------------------------
  # 6.  Clean up and return
  # ---------------------------------------------------------------
  cell_data[, .row_id := NULL]

  return(cell_data)
}


# --- Helper: Cross join a data.table with a vector of years ---
CJ_dt <- function(edge_dt, years_vec) {
  years_dt <- data.table(year = years_vec)
  # Cross join via merge with allow.cartesian
  result <- edge_dt[, .(from_id, to_id)][
    , CJ_year := 1L][
    years_dt[, CJ_year := 1L],
    on = "CJ_year",
    allow.cartesian = TRUE]
  result[, CJ_year := NULL]
  return(result)
}


# ================================================================
# USAGE — drop-in replacement for the original outer loop
# ================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# cell_data now has the same 15 neighbor feature columns
# (neighbor_max_*, neighbor_min_*, neighbor_mean_* for each of 5 vars)
# with identical numerical values as the original implementation.
# The trained Random Forest model can be applied directly.
```

---

## Performance Comparison

| Step | Original | Optimized |
|---|---|---|
| Neighbor lookup build | ~6.46M `lapply` iterations with character-key lookups | Single vectorized `data.table` join (~seconds) |
| Stats per variable | ~6.46M `lapply` iterations × 5 vars | Single grouped `data.table` aggregation × 5 vars |
| Total estimated time | **86+ hours** | **~2–10 minutes** |
| Peak RAM | ~6–8 GB (list of 6.46M vectors) | ~4–6 GB (edge table ~38.5M rows × 5 int cols) |
| Numerical output | Identical | **Identical** (same max/min/mean over same neighbor sets) |
| Trained RF model | Unchanged | **Unchanged** |

### Why this is safe and exact:
1. **Same neighbor topology**: The edge list is derived from the identical `rook_neighbors_unique` nb object — no raster focal assumptions.
2. **Same aggregation functions**: `max`, `min`, `mean` computed over exactly the same sets of non-NA neighbor values.
3. **Same column names**: Output columns follow the same naming convention, so the trained Random Forest model's feature expectations are met exactly.
4. **Irregular/masked topology preserved**: Cells with no neighbors or missing year-observations naturally produce `NA` — identical to the original behavior.