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

# S3 client
s3 = boto3.client('s3')

# Configuration from environment variables
BUCKET_NAME = os.environ.get('S3_BUCKET', 'paper-trader-data-pat0216')
STRATEGY = os.environ.get('STRATEGY', 'momentum')
GITHUB_PAT = os.environ.get('GITHUB_PAT', '')
GITHUB_REPO = 'PAT0216/paper-trader'

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


def download_from_s3():
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
        ledger_key = f'ledgers/ledger_{STRATEGY}.csv'
        ledger_path = f'{TMP_LEDGER_DIR}/ledger_{STRATEGY}.csv'
        s3.download_file(BUCKET_NAME, ledger_key, ledger_path)
        print(f"Downloaded ledger from S3")
    except Exception as e:
        print(f"No ledger in S3 (first run?): {e}")
    
    # Download snapshot for consolidated update
    try:
        snapshot_key = f'snapshots/{STRATEGY}.json'
        snapshot_path = f'{TMP_SNAPSHOT_DIR}/{STRATEGY}.json'
        s3.download_file(BUCKET_NAME, snapshot_key, snapshot_path)
        print(f"Downloaded snapshot from S3")
    except Exception as e:
        print(f"No snapshot in S3 (first run?): {e}")


def upload_to_s3():
    """Upload results back to S3 (backup)"""
    # Upload ledger
    ledger_path = f'{TMP_LEDGER_DIR}/ledger_{STRATEGY}.csv'
    if os.path.exists(ledger_path):
        s3.upload_file(ledger_path, BUCKET_NAME, f'ledgers/ledger_{STRATEGY}.csv')
        print(f"Uploaded ledger to S3")
    
    # Upload snapshot
    snapshot_path = f'{TMP_SNAPSHOT_DIR}/{STRATEGY}.json'
    if os.path.exists(snapshot_path):
        s3.upload_file(snapshot_path, BUCKET_NAME, f'snapshots/{STRATEGY}.json')
        print(f"Uploaded snapshot to S3")
    
    # Upload consolidated snapshot
    consolidated_path = f'{TMP_DATA_DIR}/portfolio_snapshot.json'
    if os.path.exists(consolidated_path):
        s3.upload_file(consolidated_path, BUCKET_NAME, 'portfolio_snapshot.json')
        print(f"Uploaded consolidated snapshot to S3")


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


def _github_update_file(local_path: str, repo_path: str, commit_msg: str = None):
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
        commit_msg = f'{STRATEGY}: Lambda update {datetime.now().strftime("%Y-%m-%d")}'
    
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


def commit_to_github():
    """Commit ledger/snapshot changes to GitHub repo."""
    if not GITHUB_PAT:
        print("WARNING: GITHUB_PAT not set, skipping GitHub commit")
        return False
    
    print("\n=== Committing to GitHub ===")
    
    commit_msg = f'{STRATEGY}: Lambda trade {datetime.now().strftime("%Y-%m-%d")}'
    
    files_to_commit = [
        (f'{TMP_LEDGER_DIR}/ledger_{STRATEGY}.csv', f'data/ledgers/ledger_{STRATEGY}.csv'),
        (f'{TMP_SNAPSHOT_DIR}/{STRATEGY}.json', f'data/snapshots/{STRATEGY}.json'),
        (f'{TMP_DATA_DIR}/portfolio_snapshot.json', 'data/portfolio_snapshot.json'),
    ]
    
    success_count = 0
    for local_path, repo_path in files_to_commit:
        if os.path.exists(local_path):
            if _github_update_file(local_path, repo_path, commit_msg):
                success_count += 1
    
    print(f"Committed {success_count}/{len(files_to_commit)} files to GitHub")
    return success_count > 0


def update_consolidated_snapshot():
    """Update the consolidated portfolio_snapshot.json with this strategy's data."""
    print("\n=== Updating Consolidated Snapshot ===")
    
    consolidated_path = f'{TMP_DATA_DIR}/portfolio_snapshot.json'
    
    # Try to download existing consolidated snapshot from S3
    try:
        s3.download_file(BUCKET_NAME, 'portfolio_snapshot.json', consolidated_path)
        with open(consolidated_path, 'r') as f:
            consolidated = json.load(f)
        print("Downloaded existing consolidated snapshot")
    except Exception as e:
        print(f"No existing consolidated snapshot, creating new: {e}")
        consolidated = {'strategies': {}, 'last_updated': None}
    
    # Ensure 'strategies' key exists
    if 'strategies' not in consolidated:
        consolidated['strategies'] = {}
    
    # Load this strategy's snapshot
    snapshot_path = f'{TMP_SNAPSHOT_DIR}/{STRATEGY}.json'
    if os.path.exists(snapshot_path):
        with open(snapshot_path, 'r') as f:
            strategy_data = json.load(f)
        consolidated['strategies'][STRATEGY] = strategy_data
        consolidated['last_updated'] = datetime.now().isoformat()
        print(f"Added {STRATEGY} data to consolidated snapshot")
    
    # Save consolidated snapshot
    with open(consolidated_path, 'w') as f:
        json.dump(consolidated, f, indent=2)
    
    print(f"Saved consolidated snapshot")


# ============ Rebalance Logic ============

def should_rebalance_today() -> bool:
    """
    Check if today is a rebalance day.
    Momentum strategy rebalances on 1st-3rd of each month.
    """
    day = datetime.now().day
    if STRATEGY == 'momentum':
        return day <= 3
    # For other strategies, always run trading
    return True


# ============ Main Handler ============

def handler(event, context):
    """Lambda entrypoint"""
    start_time = datetime.now()
    print(f"=== Paper Trader Lambda | {start_time.isoformat()} ===")
    print(f"Strategy: {STRATEGY}")
    print(f"Bucket: {BUCKET_NAME}")
    print(f"GitHub PAT configured: {'Yes' if GITHUB_PAT else 'No'}")
    
    # Check if this is a rebalance day
    is_rebalance = should_rebalance_today()
    print(f"Rebalance day: {is_rebalance}")
    
    try:
        # 0. Clean up and setup /tmp/data
        cleanup_tmp()
        setup_tmp_dirs()
        
        # 1. Download data from S3
        download_from_s3()
        
        # 2. Change to /tmp so relative paths work
        original_cwd = os.getcwd()
        os.chdir('/tmp')
        print(f"Changed working directory to /tmp")
        
        # 3. Run the trading logic (only on rebalance days)
        if is_rebalance:
            print("\n=== Running Trading Logic ===")
            from main import main as run_trading
            import sys
            
            # Set arguments for main.py
            sys.argv = ['main.py', '--mode', 'trade', '--strategy', STRATEGY, '--portfolio', STRATEGY]
            
            # Run trading
            run_trading()
        else:
            print("\n=== Non-rebalance day: updating VALUE only ===")
            # Run compute_portfolio_snapshot to record daily value
            import sys
            sys.path.insert(0, '/var/task')
            from scripts.utils.compute_portfolio_snapshot import process_single_strategy
            process_single_strategy(STRATEGY)
        
        # 4. Change back
        os.chdir(original_cwd)
        
        # 5. Update consolidated snapshot
        update_consolidated_snapshot()
        
        # 6. Upload to S3 (backup)
        upload_to_s3()
        
        # 7. Commit to GitHub (for dashboard)
        commit_to_github()
        
        elapsed = (datetime.now() - start_time).total_seconds()
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Trading completed successfully' if is_rebalance else 'Daily VALUE recorded',
                'strategy': STRATEGY,
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
