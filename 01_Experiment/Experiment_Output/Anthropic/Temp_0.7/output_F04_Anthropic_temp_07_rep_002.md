 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, the function:

1. Performs a character-based hash lookup (`id_to_ref[as.character(...)]`).
2. Indexes into the `neighbors` list.
3. Constructs character key strings via `paste(neighbor_cell_ids, data$year[i], sep = "_")`.
4. Performs another character-based hash lookup (`idx_lookup[neighbor_keys]`).

String construction and named-vector lookup are extremely slow in R when iterated millions of times in an interpreted loop. With ~6.46M rows and an average of ~4 rook neighbors per cell, this produces roughly **25.8 million `paste` calls and named-vector lookups** inside a sequential `lapply`. This is the dominant cost.

**`compute_neighbor_stats`** is a secondary bottleneck: another `lapply` over 6.46M rows, each calling `max`, `min`, `mean` on small vectors. This is repeated 5 times (once per variable), totaling ~32.3M R-level function invocations.

The Random Forest inference, by contrast, is a single vectorized `predict()` call on a matrix — fast by comparison.

## Optimization Strategy

### 1. Vectorized Neighbor Lookup via Integer Arithmetic (eliminate all `paste`/string ops)

Instead of building character keys, exploit the panel structure: every cell appears once per year in a fixed order. Build a direct integer matrix mapping `(cell_index, year_index) → row_number`. Then neighbor row indices can be retrieved via integer matrix indexing — no strings at all.

### 2. Vectorized Neighbor Stats via Matrix Indexing (eliminate per-row `lapply`)

Unroll the neighbor lookup into a long-form `(row, neighbor_row)` edge list. Then use vectorized grouped operations (via `data.table` or `rowsum`) to compute max/min/mean in one pass per variable — no R-level loop over 6.46M rows.

### 3. Preserve Numerical Equivalence

The operations (max, min, mean of non-NA neighbor values) are reproduced exactly.

## Optimized R Code

```r
library(data.table)

# ==============================================================================
# STEP 1: Build a vectorized neighbor edge list (replaces build_neighbor_lookup)
# ==============================================================================
build_neighbor_edgelist <- function(data_dt, id_order, neighbors) {
  # data_dt: a data.table with columns 'id' and 'year', row order matters
  # id_order: vector of cell IDs (same order as neighbors list)
  # neighbors: spdep nb object (list of integer index vectors into id_order)

  n_cells <- length(id_order)
  years   <- sort(unique(data_dt$year))
  n_years <- length(years)

  # Map cell id -> integer index in id_order (1-based)
  cell_idx <- setNames(seq_along(id_order), as.character(id_order))

  # Map (cell_index, year) -> row number in data_dt
  # Assumes data_dt is keyed/sorted by (id, year) or we build a lookup
  data_dt[, row_id := .I]
  row_lookup <- data_dt[, .(row_id, cell_idx = cell_idx[as.character(id)], year)]

  # Build a (cell_index, year_index) -> row_id matrix for O(1) lookup
  year_idx_map <- setNames(seq_along(years), as.character(years))
  row_matrix <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  row_lookup[, year_idx := year_idx_map[as.character(year)]]
  row_matrix[cbind(row_lookup$cell_idx, row_lookup$year_idx)] <- row_lookup$row_id

  # Build the edge list: for each cell, expand its neighbors across all years

  # Pre-compute neighbor counts
  n_neighbors <- lengths(neighbors)  # integer vector, length n_cells

  # Total directed edges across all cell-years
  total_edges <- sum(as.numeric(n_neighbors)) * n_years

  # Pre-allocate vectors
  from_row <- integer(total_edges)
  to_row   <- integer(total_edges)

  offset <- 0L
  for (ci in seq_len(n_cells)) {
    nb <- neighbors[[ci]]
    if (length(nb) == 0L) next
    for (yi in seq_len(n_years)) {
      src <- row_matrix[ci, yi]
      if (is.na(src)) next
      tgts <- row_matrix[nb, yi]
      valid <- !is.na(tgts)
      n_valid <- sum(valid)
      if (n_valid == 0L) next
      idx_range <- (offset + 1L):(offset + n_valid)
      from_row[idx_range] <- src
      to_row[idx_range]   <- tgts[valid]
      offset <- offset + n_valid
    }
  }

  # Trim to actual size
  from_row <- from_row[seq_len(offset)]
  to_row   <- to_row[seq_len(offset)]

  data.table(from_row = from_row, to_row = to_row)
}

# ==============================================================================
# STEP 2: Vectorized neighbor stats (replaces compute_neighbor_stats)
# ==============================================================================
compute_neighbor_stats_vec <- function(data_dt, edge_dt, var_name) {
  # Extract neighbor values via vectorized indexing
  vals <- data_dt[[var_name]]
  n_rows <- nrow(data_dt)

  edge_dt[, nbr_val := vals[to_row]]

  # Remove NAs
  valid_edges <- edge_dt[!is.na(nbr_val)]

  # Grouped aggregation — one pass
  stats <- valid_edges[, .(
    nb_max  = max(nbr_val),
    nb_min  = min(nbr_val),
    nb_mean = mean(nbr_val)
  ), by = from_row]

  # Allocate full-length result columns (NA default)
  result <- data.table(
    nb_max  = rep(NA_real_, n_rows),
    nb_min  = rep(NA_real_, n_rows),
    nb_mean = rep(NA_real_, n_rows)
  )
  result[stats$from_row, `:=`(
    nb_max  = stats$nb_max,
    nb_min  = stats$nb_min,
    nb_mean = stats$nb_mean
  )]

  # Clean up temporary column
  edge_dt[, nbr_val := NULL]

  setnames(result,
    c("nb_max", "nb_min", "nb_mean"),
    paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  )
  result
}

# ==============================================================================
# STEP 3: Main pipeline (replaces outer loop)
# ==============================================================================
run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {
  cell_dt <- as.data.table(cell_data)

  message("Building neighbor edge list...")
  t0 <- Sys.time()
  edge_dt <- build_neighbor_edgelist(cell_dt, id_order, rook_neighbors_unique)
  message("  Edge list built: ", nrow(edge_dt), " edges in ",
          round(difftime(Sys.time(), t0, units = "mins"), 1), " min")

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    message("Computing neighbor stats for: ", var_name)
    t1 <- Sys.time()
    stats_dt <- compute_neighbor_stats_vec(cell_dt, edge_dt, var_name)
    cell_dt <- cbind(cell_dt, stats_dt)
    message("  Done in ", round(difftime(Sys.time(), t1, units = "secs"), 1), " sec")
  }

  # Return as data.frame if original was data.frame
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    return(as.data.frame(cell_dt))
  }
  cell_dt
}

# ==============================================================================
# Usage (drop-in replacement):
# ==============================================================================
# cell_data <- run_neighbor_feature_pipeline(cell_data, id_order, rook_neighbors_unique)
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

## Further Optimization: Rcpp Inner Loop for Edge List Construction

The `build_neighbor_edgelist` function still has a double `for` loop over 344K cells × 28 years. If this remains slow (~10-20 min), replace it with Rcpp:

```r
Rcpp::cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List build_edgelist_cpp(IntegerMatrix row_matrix, List neighbors,
                        int n_cells, int n_years) {
  // Estimate size
  int total = 0;
  for (int ci = 0; ci < n_cells; ci++) {
    IntegerVector nb = neighbors[ci];
    total += nb.size() * n_years;
  }

  IntegerVector from_row(total);
  IntegerVector to_row(total);
  int offset = 0;

  for (int ci = 0; ci < n_cells; ci++) {
    IntegerVector nb = neighbors[ci];  // 1-based indices
    int n_nb = nb.size();
    if (n_nb == 0) continue;

    for (int yi = 0; yi < n_years; yi++) {
      int src = row_matrix(ci, yi);  // NA_INTEGER if missing
      if (src == NA_INTEGER) continue;

      for (int ni = 0; ni < n_nb; ni++) {
        int tgt = row_matrix(nb[ni] - 1, yi);  // nb is 1-based
        if (tgt == NA_INTEGER) continue;
        from_row[offset] = src;
        to_row[offset]   = tgt;
        offset++;
      }
    }
  }

  from_row = from_row[Range(0, offset - 1)];
  to_row   = to_row[Range(0, offset - 1)];

  return List::create(Named("from_row") = from_row,
                      Named("to_row")   = to_row);
}
')
```

## Expected Performance

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~60-80 hrs (paste/named lookup × 6.46M) | ~1-5 min (integer matrix + Rcpp) |
| `compute_neighbor_stats` (×5 vars) | ~6-10 hrs (lapply × 6.46M × 5) | ~30-90 sec (data.table grouped agg) |
| **Total neighbor features** | **~86+ hrs** | **~5-10 min** |
| RF prediction | Unchanged | Unchanged |
| Numerical output | — | Identical (max/min/mean preserved) |

The trained Random Forest model is untouched; only the feature-construction preprocessing is replaced. The computed neighbor max, min, and mean values are numerically identical to the original implementation.