"""
AWS Lambda Handler for Paper Trader
-----------------------------------
Entry point for Lambda invocations.
Downloads data from S3, runs trading, uploads results to S3 AND commits to GitHub.

IMPORTANT: 
- Lambda's /var/task is read-only. All data goes to /tmp/data/
- Dashboard reads from GitHub repo, so we must commit changes
- S3 is backup/source of truth for market.db
"""

import os
import json
import boto3
import shutil
import base64
import urllib.request
import urllib.error
from datetime import datetime
from zoneinfo import ZoneInfo

# Pacific timezone for consistent date handling
PT = ZoneInfo('America/Los_Angeles')

def pt_now():
    """Current time in Pacific."""
    return datetime.now(PT)

# S3 client
s3 = boto3.client('s3')

# Configuration from environment variables
BUCKET_NAME = os.environ.get('S3_BUCKET', 'paper-trader-data-pat0216')
GITHUB_PAT = os.environ.get('GITHUB_PAT', '')
GITHUB_REPO = 'PAT0216/paper-trader'


def get_strategy_from_event(event):
    """Get strategy from EventBridge event input, fallback to env var.
    
    EventBridge sends: {"strategy": "momentum"}
    """
    if event and isinstance(event, dict):
        if 'strategy' in event:
            return event['strategy']
    return os.environ.get('STRATEGY', 'momentum')

# Lambda-specific paths (must use /tmp for write access)
TMP_DATA_DIR = '/tmp/data'
TMP_DB_PATH = f'{TMP_DATA_DIR}/market.db'
TMP_LEDGER_DIR = f'{TMP_DATA_DIR}/ledgers'
TMP_SNAPSHOT_DIR = f'{TMP_DATA_DIR}/snapshots'


def setup_tmp_dirs():
    """Create directory structure in /tmp"""
    for d in [TMP_DATA_DIR, TMP_LEDGER_DIR, TMP_SNAPSHOT_DIR]:
        os.makedirs(d, exist_ok=True)
    print(f"Created /tmp/data structure")


def cleanup_tmp():
    """Clean up /tmp to handle Lambda warm container reuse"""
    if os.path.exists(TMP_DATA_DIR):
        shutil.rmtree(TMP_DATA_DIR)
    print("Cleaned up /tmp/data")


def download_from_s3(strategy: str):
    """Download market.db and ledgers from S3 to /tmp/data/"""
    try:
        s3.download_file(BUCKET_NAME, 'market.db', TMP_DB_PATH)
        print(f"Downloaded market.db from S3 to {TMP_DB_PATH}")
    except Exception as e:
        print(f"No market.db in S3 (first run?): {e}")
    
    # Download S&P 500 ticker list (prevents Wikipedia fetch timeout)
    try:
        tickers_path = f'{TMP_DATA_DIR}/sp500_tickers.txt'
        s3.download_file(BUCKET_NAME, 'sp500_tickers.txt', tickers_path)
        print(f"Downloaded sp500_tickers.txt from S3")
    except Exception as e:
        print(f"No sp500_tickers.txt in S3: {e}")
    
    try:
        ledger_key = f'ledgers/ledger_{strategy}.csv'
        ledger_path = f'{TMP_LEDGER_DIR}/ledger_{strategy}.csv'
        s3.download_file(BUCKET_NAME, ledger_key, ledger_path)
        print(f"Downloaded ledger from S3")
    except Exception as e:
        print(f"No ledger in S3 (first run?): {e}")
    
    # Download snapshot for consolidated update
    try:
        snapshot_key = f'snapshots/{strategy}.json'
        snapshot_path = f'{TMP_SNAPSHOT_DIR}/{strategy}.json'
        s3.download_file(BUCKET_NAME, snapshot_key, snapshot_path)
        print(f"Downloaded snapshot from S3")
    except Exception as e:
        print(f"No snapshot in S3 (first run?): {e}")


def upload_to_s3(strategy: str):
    """Upload per-strategy results back to S3 (backup)."""
    # Upload ledger
    ledger_path = f'{TMP_LEDGER_DIR}/ledger_{strategy}.csv'
    if os.path.exists(ledger_path):
        s3.upload_file(ledger_path, BUCKET_NAME, f'ledgers/ledger_{strategy}.csv')
        print(f"Uploaded ledger to S3")
    
    # Upload snapshot
    snapshot_path = f'{TMP_SNAPSHOT_DIR}/{strategy}.json'
    if os.path.exists(snapshot_path):
        s3.upload_file(snapshot_path, BUCKET_NAME, f'snapshots/{strategy}.json')
        print(f"Uploaded snapshot to S3")


def download_from_github(strategy: str):
    """Download ledger and portfolio_snapshot from GitHub (source of truth).
    
    This ensures we don't overwrite existing data when S3 is empty/stale.
    GitHub is the source of truth for ledgers, S3 is just a backup.
    """
    if not GITHUB_PAT:
        print("No GITHUB_PAT, skipping GitHub download")
        return
    
    headers = {
        'Authorization': f'token {GITHUB_PAT}',
        'Accept': 'application/vnd.github.v3.raw',  # Get raw file content
        'User-Agent': 'paper-trader-lambda'
    }
    
    files_to_download = [
        (f'data/ledgers/ledger_{strategy}.csv', f'{TMP_LEDGER_DIR}/ledger_{strategy}.csv'),
        ('data/portfolio_snapshot.json', f'{TMP_DATA_DIR}/portfolio_snapshot.json'),
    ]
    
    for repo_path, local_path in files_to_download:
        try:
            url = f'https://api.github.com/repos/{GITHUB_REPO}/contents/{repo_path}'
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                content = resp.read()
                with open(local_path, 'wb') as f:
                    f.write(content)
                print(f"Downloaded {repo_path} from GitHub")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"{repo_path} not found in GitHub (first run?)")
            else:
                print(f"Error downloading {repo_path}: {e}")
        except Exception as e:
            print(f"Error downloading {repo_path}: {e}")


# ============ GitHub API Functions ============

def _github_api_request(url: str, method: str, data: dict = None):
    """Make GitHub API request."""
    headers = {
        'Authorization': f'token {GITHUB_PAT}',
        'Accept': 'application/vnd.github.v3+json',
        'Content-Type': 'application/json',
        'User-Agent': 'paper-trader-lambda'
    }
    
    req = urllib.request.Request(url, method=method, headers=headers)
    if data:
        req.data = json.dumps(data).encode('utf-8')
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None  # File doesn't exist
        raise


def _github_get_sha(repo_path: str) -> str:
    """Get SHA of existing file (or None if not exists)."""
    api_url = f'https://api.github.com/repos/{GITHUB_REPO}/contents/{repo_path}'
    try:
        response = _github_api_request(api_url, 'GET')
        return response.get('sha') if response else None
    except Exception as e:
        print(f"Could not get SHA for {repo_path}: {e}")
        return None


def _github_update_file(local_path: str, repo_path: str, strategy: str, commit_msg: str = None):
    """Update a single file in GitHub repo via API."""
    if not os.path.exists(local_path):
        print(f"File not found: {local_path}")
        return False
    
    # Read local file
    with open(local_path, 'rb') as f:
        content = base64.b64encode(f.read()).decode('utf-8')
    
    # Get current file SHA (if exists)
    sha = _github_get_sha(repo_path)
    
    # Prepare commit message
    if commit_msg is None:
        commit_msg = f'{strategy}: Lambda update {datetime.now().strftime("%Y-%m-%d")}'
    
    # Prepare update
    api_url = f'https://api.github.com/repos/{GITHUB_REPO}/contents/{repo_path}'
    data = {
        'message': commit_msg,
        'content': content,
        'branch': 'main'
    }
    if sha:
        data['sha'] = sha
    
    try:
        _github_api_request(api_url, 'PUT', data)
        print(f"Committed {repo_path} to GitHub")
        return True
    except Exception as e:
        print(f"Failed to commit {repo_path}: {e}")
        return False


def commit_to_github(strategy: str):
    """Commit ledger/snapshot changes to GitHub repo."""
    if not GITHUB_PAT:
        print("WARNING: GITHUB_PAT not set, skipping GitHub commit")
        return False
    
    print("\n=== Committing to GitHub ===")
    
    commit_msg = f'{strategy}: Lambda trade {pt_now().strftime("%Y-%m-%d")}'
    
    # Only commit per-strategy files (not portfolio_snapshot.json)
    # Consolidated snapshot is computed by cache_refresh.yml to avoid race conditions
    files_to_commit = [
        (f'{TMP_LEDGER_DIR}/ledger_{strategy}.csv', f'data/ledgers/ledger_{strategy}.csv'),
        (f'{TMP_SNAPSHOT_DIR}/{strategy}.json', f'data/snapshots/{strategy}.json'),
    ]
    
    success_count = 0
    for local_path, repo_path in files_to_commit:
        if os.path.exists(local_path):
            if _github_update_file(local_path, repo_path, strategy, commit_msg):
                success_count += 1
    
    print(f"Committed {success_count}/{len(files_to_commit)} files to GitHub")
    return success_count > 0


def run_snapshot_computation(strategy: str):
    """Run compute_portfolio_snapshot.py for this strategy (same as GitHub Actions).
    
    This recomputes portfolio value from ledger + latest prices in market.db,
    appends a VALUE row to the ledger, and writes per-strategy snapshot JSON.
    """
    print(f"\n=== Computing Portfolio Snapshot ({strategy}) ===")
    import sys
    original_argv = sys.argv
    try:
        sys.argv = ['compute_portfolio_snapshot.py', '--strategy', strategy, '--force']
        from scripts.utils.compute_portfolio_snapshot import main as run_snapshot
        run_snapshot()
    except Exception as e:
        print(f"Snapshot computation failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sys.argv = original_argv


# ============ Rebalance Logic ============

def should_rebalance_today(strategy: str) -> bool:
    """
    Check if today is a rebalance day.
    Momentum strategy rebalances on 1st-3rd of each month.
    """
    day = pt_now().day
    if strategy == 'momentum':
        return day <= 3
    # For other strategies, always run trading
    return True


# ============ Main Handler ============

def handler(event, context):
    """Lambda entrypoint"""
    start_time = pt_now()
    
    # Get strategy from event (EventBridge) or env var
    strategy = get_strategy_from_event(event)
    
    print(f"=== Paper Trader Lambda | {start_time.isoformat()} ===")
    print(f"Strategy: {strategy}")
    print(f"Event: {event}")
    print(f"Bucket: {BUCKET_NAME}")
    print(f"GitHub PAT configured: {'Yes' if GITHUB_PAT else 'No'}")
    
    # Check if this is a rebalance day
    is_rebalance = should_rebalance_today(strategy)
    print(f"Rebalance day: {is_rebalance}")
    
    try:
        # 0. Clean up and setup /tmp/data
        cleanup_tmp()
        setup_tmp_dirs()
        
        # 1. Download data from S3 (market.db, tickers)
        download_from_s3(strategy)
        
        # 1b. Download ledger and snapshot from GitHub (source of truth)
        # This ensures we preserve existing data even if S3 is stale
        download_from_github(strategy)
        
        # 2. Change to /tmp so relative paths work for data/
        # Symlink read-only dirs, copy config/ (needs to be writable)
        for dirname in ['models', 'src', 'scripts']:
            target = f'/var/task/{dirname}'
            link = f'/tmp/{dirname}'
            if os.path.exists(target) and not os.path.exists(link):
                os.symlink(target, link)
                print(f"Symlinked {link} -> {target}")
        # Copy config (must be writable to override retrain_daily)
        config_src = '/var/task/config'
        config_dst = '/tmp/config'
        if os.path.exists(config_src) and not os.path.exists(config_dst):
            shutil.copytree(config_src, config_dst)
            print(f"Copied config/ to /tmp/config/")
        
        original_cwd = os.getcwd()
        os.chdir('/tmp')
        print(f"Changed working directory to /tmp")
        
        # 3. Run the trading logic
        # Note: Even on non-rebalance days, we run main.py which handles VALUE updates
        # The strategy itself determines if trades should be made
        print("\n=== Running Trading Logic ===")
        from main import main as run_trading
        import sys
        
        # Override retrain_daily - Lambda uses pre-trained model from container
        # Retraining exceeds Lambda's 3008 MB memory limit
        import yaml
        config_path = os.path.join('/tmp', 'config', 'settings.yaml')
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = yaml.safe_load(f)
            config['model']['retrain_daily'] = False
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            print("Disabled retrain_daily for Lambda (using pre-trained model)")
        
        # Set arguments for main.py
        sys.argv = ['main.py', '--mode', 'trade', '--strategy', strategy, '--portfolio', strategy]
        
        # Run trading (handles both rebalance and non-rebalance days)
        run_trading()
        
        # 4. Run snapshot computation while CWD is still /tmp
        # (script uses relative paths like data/snapshots/ which must resolve to /tmp/data/)
        run_snapshot_computation(strategy)
        
        # 5. Change back to original directory
        os.chdir(original_cwd)
        
        # 6. Upload to S3 (backup - ledger + per-strategy snapshot only)
        upload_to_s3(strategy)
        
        # 7. Commit to GitHub (for dashboard)
        commit_to_github(strategy)
        
        elapsed = (pt_now() - start_time).total_seconds()
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Trading completed successfully' if is_rebalance else 'Daily VALUE recorded',
                'strategy': strategy,
                'rebalance_day': is_rebalance,
                'elapsed_seconds': elapsed
            })
        }
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }
