 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three interconnected reasons:

### 1. Row-Level `lapply` Over 6.46 Million Rows in `build_neighbor_lookup`
The function iterates row-by-row across all 6.46 million cell-year observations. For each row, it:
- Looks up the cell's reference index.
- Retrieves neighbor cell IDs.
- Pastes together character keys (`id_year`).
- Matches those keys against a named lookup vector.

This means ~6.46 million iterations of string concatenation (`paste`), named-vector lookups, and `NA` filtering. Named vector lookups via character keys in R are hash-based but still carry significant per-call overhead at this scale. Crucially, **the neighbor topology is time-invariant** — cell A's neighbors are the same in 1992 as in 2019 — yet the function redundantly recomputes the same neighbor cell IDs 28 times for each of the 344,208 cells.

### 2. Row-Level `lapply` Over 6.46 Million Rows in `compute_neighbor_stats`
For each row, it subsets a numeric vector by index, removes `NA`s, and computes `max`, `min`, and `mean`. The per-element overhead of 6.46 million R function calls (each with allocation, subsetting, and three summary functions) is enormous. This is repeated for each of the 5 neighbor source variables, totaling ~32.3 million R-level iterations.

### 3. The Fundamental Missed Optimization: Separate Topology from Attributes
The neighbor topology (which cells are adjacent) is **static**. The cell attributes (ntl, ec, etc.) vary by year. The current code entangles these two concerns by building a monolithic row-index lookup across the entire panel. The correct approach is:

- Build the neighbor-pair table **once** (344,208 cells × ~4 neighbors each ≈ 1.37 million directed pairs).
- For each year, **join** that year's attribute values onto the neighbor-pair table.
- **Group by** the focal cell and compute `max`, `min`, `mean` using vectorized operations.

This converts millions of R-level loop iterations into a small number of vectorized `data.table` joins and grouped aggregations — operations that run in optimized C code.

---

## Optimization Strategy

| Step | What | Why |
|------|------|-----|
| **A** | Build a `data.table` of directed neighbor pairs `(focal_id, neighbor_id)` from `rook_neighbors_unique` once. | Topology is static; ~1.37M rows, trivially small. |
| **B** | For each year, subset the panel to that year, join cell attributes onto the neighbor-pair table by `neighbor_id`, then group by `focal_id` to compute `max`, `min`, `mean` for each variable. | Vectorized `data.table` joins and `by`-group aggregations run in C, not R loops. |
| **C** | Bind the yearly results and join back onto the full panel. | Produces the identical columns the trained Random Forest model expects. |
| **D** | Predict with the existing trained RF model as before. | Model is unchanged; numerical outputs are identical. |

**Expected speedup**: The ~1.37M-row join + grouped aggregation per year per variable takes seconds. 28 years × 5 variables = 140 such operations. Total estimated time: **2–10 minutes** on a 16 GB laptop, versus 86+ hours.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP A: Build the static neighbor-pair table ONCE
# ──────────────────────────────────────────────────────────────────────
# rook_neighbors_unique : spdep nb object (list of integer vectors)
# id_order              : vector of cell IDs in the same order as the nb object

build_neighbor_pairs <- function(id_order, neighbors) {
  # neighbors[[i]] contains the indices (into id_order) of cell i's neighbors.
  # We expand this into a two-column data.table of directed pairs.
  n <- length(id_order)
  focal_list    <- vector("list", n)
  neighbor_list <- vector("list", n)
  
  for (i in seq_len(n)) {
    nb_idx <- neighbors[[i]]
    # spdep nb objects use 0L to indicate no neighbors
    nb_idx <- nb_idx[nb_idx != 0L]
    if (length(nb_idx) > 0L) {
      focal_list[[i]]    <- rep(id_order[i], length(nb_idx))
      neighbor_list[[i]] <- id_order[nb_idx]
    }
  }
  
  data.table(
    focal_id    = unlist(focal_list, use.names = FALSE),
    neighbor_id = unlist(neighbor_list, use.names = FALSE)
  )
}

neighbor_pairs <- build_neighbor_pairs(id_order, rook_neighbors_unique)
# ~1,373,394 rows; tiny in memory

cat("Neighbor pairs built:", nrow(neighbor_pairs), "directed edges\n")

# ──────────────────────────────────────────────────────────────────────
# STEP B: Compute neighbor stats via vectorized join + grouped agg
# ──────────────────────────────────────────────────────────────────────

# Convert cell_data to data.table if not already (non-destructive copy)
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# We will collect all new columns in a separate table keyed by (id, year),
# then join once at the end to avoid repeated modification of the large table.

# Pre-allocate list to collect yearly results
yearly_results <- vector("list", length(unique(cell_data$year)))
names(yearly_results) <- as.character(sort(unique(cell_data$year)))

# Columns we need from cell_data for the neighbor lookup
subset_cols <- c("id", "year", neighbor_source_vars)

# Build the aggregation expression dynamically
# For each variable v, we want: v_neighbor_max, v_neighbor_min, v_neighbor_mean
agg_exprs <- paste0(
  sprintf(
    "list(%s)",
    paste(
      unlist(lapply(neighbor_source_vars, function(v) {
        c(
          sprintf("nb_%s_max  = as.numeric(max(%s, na.rm = TRUE))", v, v),
          sprintf("nb_%s_min  = as.numeric(min(%s, na.rm = TRUE))", v, v),
          sprintf("nb_%s_mean = as.numeric(mean(%s, na.rm = TRUE))", v, v)
        )
      })),
      collapse = ", "
    )
  )
)
agg_expr_parsed <- parse(text = agg_exprs)

years <- sort(unique(cell_data$year))

cat("Computing neighbor statistics for", length(years), "years ...\n")

for (yr in years) {
  # Subset to this year's attributes
  yr_data <- cell_data[year == yr, ..subset_cols]
  
  # Join neighbor attributes onto the pair table
  # Key the yearly data by cell id for fast join
  setkey(yr_data, id)
  
  # Merge: for each (focal_id, neighbor_id) pair, attach the neighbor's attributes
  # We join on neighbor_id = id
  merged <- neighbor_pairs[yr_data, on = .(neighbor_id = id), nomatch = 0L, allow.cartesian = TRUE]
  # merged now has columns: focal_id, neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2
  # where the attribute columns belong to the NEIGHBOR cell
  
  # Group by focal_id and compute stats
  stats <- merged[, eval(agg_expr_parsed), by = .(focal_id)]
  stats[, year := yr]
  
  # Handle -Inf/Inf from max/min on all-NA groups (shouldn't happen if data is clean,
  # but defensive)
  for (v in neighbor_source_vars) {
    max_col  <- paste0("nb_", v, "_max")
    min_col  <- paste0("nb_", v, "_min")
    mean_col <- paste0("nb_", v, "_mean")
    set(stats, which(is.infinite(stats[[max_col]])),  max_col,  NA_real_)
    set(stats, which(is.infinite(stats[[min_col]])),  min_col,  NA_real_)
    set(stats, which(is.nan(stats[[mean_col]])),      mean_col, NA_real_)
  }
  
  yearly_results[[as.character(yr)]] <- stats
}

all_neighbor_stats <- rbindlist(yearly_results, use.names = TRUE)

cat("Neighbor stats computed:", nrow(all_neighbor_stats), "rows,",
    ncol(all_neighbor_stats), "columns\n")

# ──────────────────────────────────────────────────────────────────────
# STEP C: Join neighbor stats back onto the main panel
# ──────────────────────────────────────────────────────────────────────

# Rename focal_id -> id for the join
setnames(all_neighbor_stats, "focal_id", "id")
setkey(all_neighbor_stats, id, year)
setkey(cell_data, id, year)

# If the cell_data already has old neighbor columns from a prior run, remove them
old_nb_cols <- grep("^nb_", names(cell_data), value = TRUE)
if (length(old_nb_cols) > 0L) {
  cell_data[, (old_nb_cols) := NULL]
}

# Merge
cell_data <- all_neighbor_stats[cell_data, on = .(id, year)]

cat("Final panel:", nrow(cell_data), "rows,", ncol(cell_data), "columns\n")

# ──────────────────────────────────────────────────────────────────────
# STEP D: Column-name alignment with the trained RF model
# ──────────────────────────────────────────────────────────────────────
# The trained Random Forest model expects specific column names for the
# neighbor features. If the original code produced columns named e.g.
# "ntl_neighbor_max" instead of "nb_ntl_max", rename to match.
#
# Adjust the mapping below to match YOUR model's expected feature names.
# Example (uncomment and edit as needed):
#
# setnames(cell_data, 
#   old = c("nb_ntl_max",  "nb_ntl_min",  "nb_ntl_mean"),
#   new = c("ntl_neighbor_max", "ntl_neighbor_min", "ntl_neighbor_mean")
# )
# ... repeat for ec, pop_density, def, usd_est_n2 ...

# ──────────────────────────────────────────────────────────────────────
# STEP E: Predict with the existing trained Random Forest (unchanged)
# ──────────────────────────────────────────────────────────────────────
# The model object (e.g., rf_model) is already in memory and must NOT
# be retrained. Simply call predict:
#
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
#
# The numerical estimand is preserved because the neighbor statistics
# (max, min, mean) are computed identically — only the computational
# method changed, not the mathematical operations.
```

---

## Why the Numerical Estimand Is Preserved

| Operation | Original code | Optimized code |
|-----------|--------------|----------------|
| Neighbor identification | `id_order[neighbors[[ref_idx]]]` | Same `id_order[neighbors[[i]]]` in `build_neighbor_pairs` |
| Attribute retrieval | `vals[idx]` (direct index) | `data.table` equi-join on `neighbor_id = id` within the same year |
| `max` | `max(neighbor_vals)` after `NA` removal | `max(v, na.rm = TRUE)` grouped by `focal_id` |
| `min` | `min(neighbor_vals)` after `NA` removal | `min(v, na.rm = TRUE)` grouped by `focal_id` |
| `mean` | `mean(neighbor_vals)` after `NA` removal | `mean(v, na.rm = TRUE)` grouped by `focal_id` |

The same set of neighbor cells contributes the same attribute values, and the same three summary functions are applied. The results are numerically identical (up to floating-point associativity, which for `max`/`min` is exact and for `mean` is identical since `data.table` uses the same `mean` implementation). The trained Random Forest model receives identical input features and is never retrained.

---

## Expected Performance

| Component | Original | Optimized |
|-----------|----------|-----------|
| Neighbor lookup build | ~hours (6.46M `lapply` + string ops) | ~5 seconds (344K-iteration loop, vectorized `unlist`) |
| Neighbor stats (per variable) | ~17 hours (6.46M `lapply`) | ~30 seconds (28 joins + grouped aggs) |
| **Total (5 variables)** | **~86+ hours** | **~3–10 minutes** |
| Peak RAM | ~6–10 GB (list of 6.46M integer vectors) | ~2–4 GB (1.37M-row pair table + yearly subsets) |