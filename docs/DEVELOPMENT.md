# Development Guidelines

## ⚠️ Data Freshness Requirements

**CRITICAL: Always sync local data before testing scripts that modify ledgers or snapshots.**

### Before Running Locally

```bash
# 1. Pull latest from GitHub (ledgers, snapshots)
git pull

# 2. If testing snapshot scripts, the local market.db may be stale
# The workflows use a fresh cache from GitHub Actions
# Your local cache is NOT automatically updated

# 3. NEVER commit data generated from local testing without verification
```

### Why This Matters

The trading workflows run in Docker containers with fresh market data from GitHub Actions cache. Your local `data/market.db` is likely outdated.

**What can go wrong:**
- Local test creates snapshots with Dec 22 prices
- You push these snapshots  
- Dashboard shows wrong values until next workflow run

### Safe Testing Pattern

```bash
# For viewing/debugging only (safe)
python scripts/utils/compute_portfolio_snapshot.py --strategy ml

# For production (use workflows)
# Go to GitHub Actions → trigger the workflow manually
```

### Key Files That Require Fresh Data

| File | Source of Truth |
|------|-----------------|
| `data/portfolio_snapshot.json` | GitHub workflows |
| `data/snapshots/*.json` | GitHub workflows |
| `ledger_*.csv` | GitHub workflows |
| `data/market.db` | GitHub Actions cache |

### Local Development

For local development that doesn't affect production data:
- Create a separate branch
- Use `--dry-run` flags where available
- Test with mock data

---

## Workflow Architecture

### Schedule (UTC)

| Workflow | Schedule | Runs |
|----------|----------|------|
| Cache Refresh | 21:00 | Daily Mon-Fri |
| ML Trade | 21:30 | Daily Mon-Fri |
| LSTM Trade | 21:45 | Daily Mon-Fri |
| Momentum Trade | 22:00 | 1st-3rd of month |

### Per-Strategy Snapshots (v1.9.0+)

Each workflow writes to its own file to avoid merge conflicts:
- `data/snapshots/ml.json`
- `data/snapshots/lstm.json`
- `data/snapshots/momentum.json`

Then consolidates into `data/portfolio_snapshot.json`.
