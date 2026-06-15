 **Diagnosis**  
The main bottleneck is `build_neighbor_lookup()` constructing a per-row neighbor list by iterating over ~6.46M rows and generating strings (`paste()`) for lookups. This is extremely inefficient in both time and memory. `compute_neighbor_stats()` repeats expensive list traversals across 5 variables. Random Forest inference is negligible compared to this.

---

**Optimization Strategy**  
1. **Eliminate repeated string concatenation and `lapply`** by precomputing numeric indices for cells (not per-row strings).
2. **Use vectorized matrix joins**: Represent neighbors as a sparse adjacency structure indexed by cell IDs and reuse it for all years.
3. **Work chunk-wise** or by year to keep memory at ~6M rows manageable.
4. **Use `data.table` for fast joins and aggregations** instead of per-row `lapply`.
5. **Compute neighbor stats in one grouped join**, not per row per variable.

---

**Optimized R Code (concept)**  

```r
library(data.table)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# Precompute neighbor pairs from rook_neighbors_unique
# Build a two-column edge list: (cell_id -> neighbor_id)
edges <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor = unlist(rook_neighbors_unique, use.names = FALSE)
)

# For each year, self-join on neighbors, then aggregate
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Result container
results_list <- vector("list", length(vars))
names(results_list) <- vars

# Process year by year to reduce memory footprint
years <- unique(cell_data$year)

for (yr in years) {
  dt_year <- cell_data[year == yr, .(id, year, ntl, ec, pop_density, def, usd_est_n2)]
  
  # Join neighbor values
  dt_n <- edges[dt_year, on = .(neighbor = id)]
  # dt_n now has: id (from edges), neighbor, and neighbor vars
  
  # Join back to target id as key
  setnames(dt_n, old = "id", new = "orig_id")
  
  # Aggregate neighbor stats for each orig_id
  nb_stats <- dt_n[, lapply(.SD, function(x) {
    if (all(is.na(x))) return(rep(NA_real_, 3))
    c(max(x, na.rm=TRUE), min(x, na.rm=TRUE), mean(x, na.rm=TRUE))
  }), by = orig_id, .SDcols = vars]
  
  # Reshape into wide format: one row per orig_id, columns per var and stat
  # This step depends on how you want to name the columns
  stat_names <- c("max", "min", "mean")
  col_order <- c("orig_id",
                 as.vector(outer(vars, stat_names, paste, sep = "_nb_")))
  nb_stats_long <- melt(nb_stats, id.vars="orig_id", measure.vars = patterns(vars),
                         variable.name="var", value.name="vals")
  
  # Integrate back into cell_data by id and year
  cell_data[year == yr, paste0(vars, "_nb_max") := nb_stats[match(id, orig_id), lapply(.SD, `[`, 1), .SDcols=vars]]
  cell_data[year == yr, paste0(vars, "_nb_min") := nb_stats[match(id, orig_id), lapply(.SD, `[`, 2), .SDcols=vars]]
  cell_data[year == yr, paste0(vars, "_nb_mean") := nb_stats[match(id, orig_id), lapply(.SD, `[`, 3), .SDcols=vars]]
}

```

---

**Key Speed Gains**  
- `edges` defines cell-to-neighbor relationships **once**.
- String concatenation removed.
- Only 28 yearly joins instead of 6.46M per-row operations.
- Vectorized aggregation via `data.table`, not per-row `lapply`.
- Memory-friendly since only one year processed at a time.

**Expected Outcome**: From 86+ hours to **minutes-to-several-hours** on a laptop. Random Forest inference remains untouched, and original numerical estimand (max, min, mean neighbors per variable) is preserved.