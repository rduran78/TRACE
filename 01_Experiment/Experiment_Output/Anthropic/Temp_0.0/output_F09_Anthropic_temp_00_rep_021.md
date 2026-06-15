 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(N²)-like in practice
This function iterates over **all 6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs **year-specific string keys** (`paste(neighbor_id, year, sep="_")`) and matches them against a named character vector (`idx_lookup`) of 6.46 million entries.

Named vector lookup in R via `[` on character names is **O(n)** per probe in the worst case (hash collisions aside, the overhead of repeated character hashing and matching across 6.46M keys is enormous). Doing this 6.46 million times produces billions of character operations.

### 2. The lookup is **rebuilt monolithically** even though the spatial topology is time-invariant
The neighbor graph is purely spatial — cell A's rook neighbors are the same in 1992 as in 2019. Yet the function fuses space and time into one giant lookup, recomputing string keys for every cell-year. This is the core waste: **the adjacency structure only needs to be defined once over 344,208 cells, not over 6.46 million cell-years.**

### 3. `compute_neighbor_stats` uses row-level `lapply` over 6.46M rows
Even after the lookup is built, computing stats via `lapply` with per-element R function calls is slow. Each call to the anonymous function has interpreter overhead. With 5 variables × 6.46M rows = 32.3 million R function invocations, this adds hours.

### Summary of bottlenecks

| Component | Calls | Cost per call | Total |
|---|---|---|---|
| `build_neighbor_lookup` (string key construction + named vector lookup) | 6.46M | ~µs–ms (character hashing over 6.46M-entry vector) | **Tens of hours** |
| `compute_neighbor_stats` (R-level lapply) | 5 × 6.46M | ~µs | **Hours** |

---

## Optimization Strategy

**Core insight:** Separate the time-invariant spatial adjacency from the time-varying attributes. Build the adjacency table once (344K cells), then use vectorized joins and grouped operations.

### Step-by-step plan

1. **Build a static edge table once** — a two-column `data.table` of `(cell_id, neighbor_id)` from the `nb` object. This has ~1.37M rows and never changes.

2. **Join yearly attributes onto the edge table** — For each year, join the cell-year attribute values onto the `neighbor_id` column. This is a keyed `data.table` join: O(n log n) once, then O(1) per probe.

3. **Compute grouped aggregates** — Group by `(cell_id, year)` and compute `max`, `min`, `mean` of each neighbor variable in one vectorized pass using `data.table`'s `[, .(…), by=]`.

4. **Join results back** to the main dataset.

This replaces 6.46M R-level function calls with a handful of vectorized `data.table` operations.

### Expected speedup

| Operation | Old | New |
|---|---|---|
| Build adjacency | ~hours (string matching over 6.46M keys) | <1 second (integer edge list from nb object) |
| Neighbor stats (per variable) | ~17 hours (lapply over 6.46M rows) | ~5–30 seconds (data.table grouped aggregation) |
| **Total for 5 variables** | **~86 hours** | **~2–5 minutes** |

---

## Working R Code

```r
library(data.table)

# ==============================================================
# STEP 0: Ensure cell_data is a data.table with proper columns
# ==============================================================
# cell_data must have columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order is the vector of cell IDs corresponding to indices in rook_neighbors_unique
# rook_neighbors_unique is the spdep nb object (list of integer index vectors)

# Convert to data.table if not already
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ==============================================================
# STEP 1: Build static spatial edge table ONCE
#         This encodes the rook adjacency among 344,208 cells.
# ==============================================================
build_edge_table <- function(id_order, neighbors_nb) {
  # neighbors_nb is an nb object: a list where element i contains

  # integer indices of neighbors of cell i (0 means no neighbors in spdep).
  edges <- rbindlist(lapply(seq_along(neighbors_nb), function(i) {
    nb_idx <- neighbors_nb[[i]]
    # spdep uses 0L to denote "no neighbors"; filter those out
    nb_idx <- nb_idx[nb_idx > 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(cell_id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  return(edges)
}

cat("Building static edge table...\n")
edge_table <- build_edge_table(id_order, rook_neighbors_unique)
# edge_table has ~1,373,394 rows: (cell_id, neighbor_id)
cat(sprintf("Edge table: %d directed edges\n", nrow(edge_table)))

# ==============================================================
# STEP 2: Compute neighbor features for all variables at once
#         using vectorized data.table joins and grouped aggregation.
# ==============================================================
compute_all_neighbor_features <- function(cell_dt, edge_dt, source_vars) {
  # We need: for each (cell_id, year), the max/min/mean of each source_var

  # across that cell's rook neighbors in the same year.

  # Subset to only the columns we need for the join
  join_cols <- c("id", "year", source_vars)
  attr_dt <- cell_dt[, ..join_cols]

  # Create the expanded neighbor-attribute table:
  # For every edge (cell_id -> neighbor_id), join the neighbor's year-specific attributes.
  # First, cross edge_table with all years via join on neighbor_id.
  # Key the attribute table for fast join.
  setkey(attr_dt, id, year)

  # Expand: join neighbor attributes onto edge table.
  # We want: for each (cell_id, neighbor_id) edge and each year,
  # the neighbor_id's attribute values in that year.
  # This is: edge_table joined to attr_dt on (neighbor_id = id).
  # Result has nrow(edge_table) * n_years rows in the worst case,
  # but we do it efficiently by joining.

  # Rename for clarity before join
  setnames(attr_dt, "id", "neighbor_id")

  # Keyed join: for each (neighbor_id, year) in attr_dt,
  # find matching rows in edge_table by neighbor_id.
  # We want the Cartesian-ish result: each edge × each year where neighbor has data.
  setkey(edge_dt, neighbor_id)
  setkey(attr_dt, neighbor_id, year)

  # Merge: every edge gets expanded by all years the neighbor has data

  cat("Joining neighbor attributes onto edge table...\n")
  expanded <- merge(edge_dt, attr_dt, by = "neighbor_id", allow.cartesian = TRUE)
  # expanded columns: neighbor_id, cell_id, year, ntl, ec, pop_density, def, usd_est_n2
  # rows: ~1.37M edges × 28 years ≈ 38.5M rows (fits in 16GB RAM easily)

  rm(attr_dt)
  gc()

  # Now group by (cell_id, year) and compute stats for each variable
  cat("Computing grouped neighbor statistics...\n")

  # Build aggregation expressions dynamically
  agg_exprs <- unlist(lapply(source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }))

  agg_names <- unlist(lapply(source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  names(agg_exprs) <- agg_names

  # Construct the call
  agg_call <- as.call(c(as.name("list"), agg_exprs))

  neighbor_stats <- expanded[, eval(agg_call), by = .(cell_id, year)]

  rm(expanded)
  gc()

  # Replace Inf/-Inf (from max/min on all-NA) with NA
  for (col_name in agg_names) {
    vals <- neighbor_stats[[col_name]]
    set(neighbor_stats, i = which(is.infinite(vals)), j = col_name, value = NA_real_)
  }

  return(neighbor_stats)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing all neighbor features...\n")
neighbor_features <- compute_all_neighbor_features(cell_data, edge_table, neighbor_source_vars)

# ==============================================================
# STEP 3: Join neighbor features back onto cell_data
# ==============================================================
cat("Joining neighbor features back to cell_data...\n")

# Remove any pre-existing neighbor columns to avoid duplication
existing_neighbor_cols <- grep("^neighbor_", names(cell_data), value = TRUE)
if (length(existing_neighbor_cols) > 0) {
  cell_data[, (existing_neighbor_cols) := NULL]
}

setkey(cell_data, id, year)
setkey(neighbor_features, cell_id, year)

cell_data <- merge(cell_data, neighbor_features,
                   by.x = c("id", "year"),
                   by.y = c("cell_id", "year"),
                   all.x = TRUE)

rm(neighbor_features)
gc()

cat("Done. Neighbor features added to cell_data.\n")
cat(sprintf("cell_data: %d rows, %d columns\n", nrow(cell_data), ncol(cell_data)))

# ==============================================================
# STEP 4: Predict with the existing trained Random Forest model
#         (model object is assumed to already exist in memory)
# ==============================================================
# The trained RF model is preserved exactly as-is; no retraining.
# Ensure column names match what the model expects.
# Example (adjust 'rf_model' to your actual model object name):
#
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Key Points

| Concern | How it's handled |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` with `na.rm=TRUE` are identical operations to the original; `Inf`/`-Inf` from empty groups are converted to `NA`, matching the original `c(NA,NA,NA)` return. |
| **Trained RF model preserved** | The code only modifies predictor columns in `cell_data`; the model object is never touched or retrained. |
| **RAM on 16 GB laptop** | The largest intermediate object (expanded edge × year table) is ~38.5M rows × ~7 columns of doubles ≈ **2.2 GB**. Combined with `cell_data` (~6.46M × 110 cols ≈ 5.7 GB), total peak is ~10–12 GB. Feasible on 16 GB. |
| **Runtime** | Edge table build: <1s. Merge: ~10–30s. Grouped aggregation: ~30–120s. Total: **~2–5 minutes** vs. 86+ hours. |