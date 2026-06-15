 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is **`build_neighbor_lookup`**, which calls `lapply` over ~6.46 million rows, performing per-row string pasting, named-vector lookups, and NA filtering. Named vector lookups via character keys in R are hash-table operations, but doing millions of `paste()` + `[` lookups inside a loop is extremely slow. The second bottleneck is **`compute_neighbor_stats`**, which iterates over the same 6.46 million entries to subset numeric vectors and compute `max/min/mean`. Together, repeated across 5 variables, this produces the estimated 86+ hour runtime.

**Specific problems:**

1. **`build_neighbor_lookup` is O(N × K) with high constant overhead.** For each of the ~6.46M rows, it does string concatenation (`paste`), named-vector character lookups (`idx_lookup[neighbor_keys]`), and NA removal. Character-keyed lookups on a 6.46M-element named vector are expensive per call.

2. **`compute_neighbor_stats` returns a list of 6.46M 3-element vectors, then `do.call(rbind, ...)`**, which is a notoriously slow pattern for large list-to-matrix conversion.

3. **No vectorization or use of data.table/matrix operations.** Everything is scalar-loop R code operating on millions of rows.

4. **Memory:** With 6.46M rows × 110 columns, the base data frame is ~5–6 GB. Building a 6.46M-element list of integer vectors for `neighbor_lookup` adds substantial memory overhead, and the `do.call(rbind, ...)` on a 6.46M-length list creates a temporary copy.

---

## Optimization Strategy

### Key Idea: Replace per-row lookups with vectorized join + grouped aggregation via `data.table`

Instead of building a row-index lookup list and iterating row by row, we:

1. **Build an edge table** (a two-column data.table of `id`→`neighbor_id` from the `nb` object) — done once.
2. **Join** the edge table to the panel data by `(neighbor_id, year)` to get neighbor variable values — this is a vectorized keyed merge, extremely fast in `data.table`.
3. **Aggregate** (`max`, `min`, `mean`) grouped by `(id, year)` — also vectorized.
4. **Merge** results back into the main table.

This eliminates all per-row `paste`/lookup/subsetting and replaces it with operations `data.table` executes in C. Expected runtime: **minutes, not hours**. Memory stays within 16 GB because the edge table has ~1.37M edges, and the join expands to ~1.37M × 28 ≈ 38.4M rows per variable (manageable in chunks).

We **do not** retrain the Random Forest. We **preserve the original numerical output** (max, min, mean of neighbor values) — the aggregation logic is identical.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 0. Convert main data to data.table (in-place, no copy)
# ---------------------------------------------------------------
setDT(cell_data)

# ---------------------------------------------------------------
# 1. Build edge table from the nb object (one-time, fast)
#    rook_neighbors_unique is a list of integer index vectors
#    id_order is the vector mapping index position -> cell id
# ---------------------------------------------------------------
build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate vectors
  n_edges <- sum(lengths(neighbors))
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_i <- neighbors[[i]]
    # spdep nb objects use 0L to denote "no neighbors"
    nb_i <- nb_i[nb_i != 0L]
    n_i  <- length(nb_i)
    if (n_i > 0L) {
      from_id[pos:(pos + n_i - 1L)] <- id_order[i]
      to_id[pos:(pos + n_i - 1L)]   <- id_order[nb_i]
      pos <- pos + n_i
    }
  }
  # Trim if any 0-neighbor cells shortened the vector
  data.table(id = from_id[1:(pos - 1L)],
             neighbor_id = to_id[1:(pos - 1L)])
}

edges <- build_edge_table(id_order, rook_neighbors_unique)

cat(sprintf("Edge table: %d directed edges\n", nrow(edges)))

# ---------------------------------------------------------------
# 2. Compute neighbor features for each source variable
#    Strategy: join edges × year to cell_data to get neighbor
#    values, then aggregate by (id, year).
# ---------------------------------------------------------------

# Key the main table for fast joins
setkey(cell_data, id, year)

# We will join on (neighbor_id, year) -> (id, year) in cell_data,
# so we need a lookup keyed on (id, year).

# Create a minimal lookup table (only id, year, and the source vars)
# to keep memory down during the join.
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
lookup_cols <- c("id", "year", neighbor_source_vars)
cell_lookup <- cell_data[, ..lookup_cols]
setkey(cell_lookup, id, year)

# Get unique years to iterate in chunks (reduces peak memory)
all_years <- sort(unique(cell_data$year))

compute_and_add_all_neighbor_features <- function(cell_data, edges,
                                                   cell_lookup,
                                                   neighbor_source_vars,
                                                   all_years,
                                                   year_chunk_size = 7L) {
  
  # We'll accumulate results per year-chunk, then rbind at the end
  # and merge once.  This controls peak memory.
  
  year_chunks <- split(all_years,
                       ceiling(seq_along(all_years) / year_chunk_size))
  
  agg_list <- vector("list", length(year_chunks))
  
  for (ch_idx in seq_along(year_chunks)) {
    yrs <- year_chunks[[ch_idx]]
    cat(sprintf("  Processing years: %s\n", paste(yrs, collapse = ", ")))
    
    # Subset lookup to these years
    chunk_lookup <- cell_lookup[year %in% yrs]
    setkey(chunk_lookup, id, year)
    
    # Cross edges with years in this chunk to get (id, neighbor_id, year)
    # More memory-efficient: join edges to chunk_lookup on neighbor_id
    # We rename columns for the join:
    #   edges: id, neighbor_id
    #   chunk_lookup keyed on (id, year) — but we want to look up by neighbor_id
    
    # Expand edges × chunk years
    edge_year <- CJ_edges_years(edges, yrs)
    # edge_year has columns: id, neighbor_id, year
    
    # Join to get neighbor values
    setkey(edge_year, neighbor_id, year)
    setkey(chunk_lookup, id, year)
    
    # Merge: edge_year[neighbor_id, year] -> chunk_lookup[id, year]
    edge_year <- chunk_lookup[edge_year,
                               on = .(id = neighbor_id, year = year),
                               nomatch = NA,
                               allow.cartesian = FALSE]
    
    # After join, 'id' is the neighbor's id (from chunk_lookup).
    # The focal cell's id came from edges and is in 'i.id'.
    # Rename for clarity:
    setnames(edge_year, c("id", "i.id"), c("neighbor_id", "id"))
    
    # Aggregate by (id, year)
    agg_exprs <- list()
    for (v in neighbor_source_vars) {
      agg_exprs[[paste0("nb_max_", v)]] <-
        substitute(suppressWarnings(max(VAR, na.rm = TRUE)),
                   list(VAR = as.name(v)))
      agg_exprs[[paste0("nb_min_", v)]] <-
        substitute(suppressWarnings(min(VAR, na.rm = TRUE)),
                   list(VAR = as.name(v)))
      agg_exprs[[paste0("nb_mean_", v)]] <-
        substitute(mean(VAR, na.rm = TRUE), list(VAR = as.name(v)))
    }
    
    agg_result <- edge_year[,
                             lapply(agg_exprs, eval, envir = .SD),
                             by = .(id, year)]
    
    # Replace Inf/-Inf from max/min on all-NA groups with NA
    inf_cols <- grep("^nb_max_|^nb_min_", names(agg_result), value = TRUE)
    for (col in inf_cols) {
      set(agg_result, which(is.infinite(agg_result[[col]])), col, NA_real_)
    }
    
    agg_list[[ch_idx]] <- agg_result
    
    # Free memory
    rm(chunk_lookup, edge_year, agg_result)
    gc()
  }
  
  all_agg <- rbindlist(agg_list, use.names = TRUE)
  rm(agg_list); gc()
  
  # Merge back into cell_data
  setkey(all_agg, id, year)
  setkey(cell_data, id, year)
  cell_data <- all_agg[cell_data, on = .(id, year)]
  
  return(cell_data)
}

# Helper: cross join edges with a vector of years
CJ_edges_years <- function(edges, yrs) {
  yr_dt <- data.table(year = yrs)
  result <- edges[, .(neighbor_id = neighbor_id), by = id][
    , CJ_dt := 1L][
    yr_dt[, CJ_dt := 1L],
    on = "CJ_dt",
    allow.cartesian = TRUE]
  result[, CJ_dt := NULL]
  result
}

# --- Actually, a cleaner CJ helper: ---
CJ_edges_years <- function(edges, yrs) {
  # Repeat each edge length(yrs) times
  n_e <- nrow(edges)
  n_y <- length(yrs)
  data.table(
    id          = rep(edges$id, each = n_y),
    neighbor_id = rep(edges$neighbor_id, each = n_y),
    year        = rep(yrs, times = n_e)
  )
}

# ---------------------------------------------------------------
# 3. Run it
# ---------------------------------------------------------------
cat("Computing neighbor features...\n")
cell_data <- compute_and_add_all_neighbor_features(
  cell_data, edges, cell_lookup, neighbor_source_vars, all_years,
  year_chunk_size = 7L   # 7 years per chunk → 4 chunks for 28 years
)
cat("Done.\n")

# ---------------------------------------------------------------
# 4. Predict with the existing trained Random Forest (unchanged)
# ---------------------------------------------------------------
# cell_data$predicted_gdp <- predict(trained_rf_model, newdata = cell_data)
```

---

### Simpler, More Robust Version (Recommended)

The above is somewhat complex due to the chunked cross-join. Here is a cleaner version that is still fast and stays within 16 GB:

```r
library(data.table)
setDT(cell_data)

# --- 1. Build edge table ---
build_edge_table <- function(id_order, neighbors) {
  from <- rep(id_order, times = lengths(neighbors))
  to   <- id_order[unlist(neighbors)]
  # Remove any spdep 0-index artifacts
  valid <- (to != 0L) & !is.na(to)
  data.table(id = from[valid], neighbor_id = to[valid])
}

edges <- build_edge_table(id_order, rook_neighbors_unique)

# --- 2. Prepare a slim lookup keyed by (id, year) ---
nb_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
slim <- cell_data[, c("id", "year", nb_vars), with = FALSE]
setkey(slim, id, year)

# --- 3. For each year, join and aggregate ---
#     Process one year at a time to control memory.
agg_all <- rbindlist(lapply(sort(unique(cell_data$year)), function(yr) {
  cat(yr, " ")
  
  # All focal cells this year
  focal_ids <- cell_data[year == yr, .(id)]
  
  # Edges for focal cells this year
  ey <- edges[focal_ids, on = "id", nomatch = 0L]  # id, neighbor_id
  
  # Look up neighbor values
  ey[, year := yr]
  nb_vals <- slim[ey, on = .(id = neighbor_id, year), nomatch = NA]
  # nb_vals now has columns: id (=neighbor_id), year, ntl, ..., i.id (=focal id)
  setnames(nb_vals, "i.id", "focal_id")
  
  # Aggregate
  nb_vals[, .(
    nb_max_ntl         = fifelse(all(is.na(ntl)),         NA_real_, max(ntl,         na.rm = TRUE)),
    nb_min_ntl         = fifelse(all(is.na(ntl)),         NA_real_, min(ntl,         na.rm = TRUE)),
    nb_mean_ntl        = mean(ntl, na.rm = TRUE),
    nb_max_ec          = fifelse(all(is.na(ec)),          NA_real_, max(ec,          na.rm = TRUE)),
    nb_min_ec          = fifelse(all(is.na(ec)),          NA_real_, min(ec,          na.rm = TRUE)),
    nb_mean_ec         = mean(ec, na.rm = TRUE),
    nb_max_pop_density = fifelse(all(is.na(pop_density)), NA_real_, max(pop_density, na.rm = TRUE)),
    nb_min_pop_density = fifelse(all(is.na(pop_density)), NA_real_, min(pop_density, na.rm = TRUE)),
    nb_mean_pop_density= mean(pop_density, na.rm = TRUE),
    nb_max_def         = fifelse(all(is.na(def)),         NA_real_, max(def,         na.rm = TRUE)),
    nb_min_def         = fifelse(all(is.na(def)),         NA_real_, min(def,         na.rm = TRUE)),
    nb_mean_def        = mean(def, na.rm = TRUE),
    nb_max_usd_est_n2  = fifelse(all(is.na(usd_est_n2)), NA_real_, max(usd_est_n2, na.rm = TRUE)),
    nb_min_usd_est_n2  = fifelse(all(is.na(usd_est_n2)), NA_real_, min(usd_est_n2, na.rm = TRUE)),
    nb_mean_usd_est_n2 = mean(usd_est_n2, na.rm = TRUE)
  ), by = .(id = focal_id, year)]
}))
cat("\n")

# --- 4. Merge back ---
setkey(agg_all, id, year)
setkey(cell_data, id, year)
cell_data <- agg_all[cell_data, on = .(id, year)]

# --- 5. Predict (model unchanged) ---
# cell_data$predicted_gdp <- predict(trained_rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M-iteration `lapply` with `paste`/character key lookups | Vectorized edge-table join via `data.table` keyed merge |
| **Aggregation** | 6.46M-iteration `lapply` + `do.call(rbind, ...)` | `data.table` grouped `[, .(...), by = .(id, year)]` in C |
| **Variables** | Separate loop per variable, rebuilding structures | All 5 variables computed in a single join + aggregate pass |
| **Memory control** | 6.46M-element list of integer vectors (~2–3 GB) | Year-by-year processing; peak ≈ 3–5 GB working set |
| **Estimated runtime** | 86+ hours | **~10–30 minutes** on the same laptop |
| **RF model** | Unchanged | Unchanged |
| **Numerical output** | max, min, mean of non-NA neighbor values | Identical max, min, mean of non-NA neighbor values |