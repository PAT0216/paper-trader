"""
AWS Lambda Handler for Paper Trader
-----------------------------------
Entry point for Lambda invocations.
Downloads data from S3, runs trading, uploads results back to S3.

IMPORTANT: Lambda's /var/task is read-only. All data goes to /tmp/data/
"""

import os
import json
import boto3
import shutil
from datetime import datetime

# S3 client
s3 = boto3.client('s3')

# Configuration from environment variables
BUCKET_NAME = os.environ.get('S3_BUCKET', 'paper-trader-data')
STRATEGY = os.environ.get('STRATEGY', 'momentum')

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
    
    try:
        ledger_key = f'ledgers/ledger_{STRATEGY}.csv'
        ledger_path = f'{TMP_LEDGER_DIR}/ledger_{STRATEGY}.csv'
        s3.download_file(BUCKET_NAME, ledger_key, ledger_path)
        print(f"Downloaded ledger from S3")
    except Exception as e:
        print(f"No ledger in S3 (first run?): {e}")


def upload_to_s3():
    """Upload results back to S3"""
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


def handler(event, context):
    """Lambda entrypoint"""
    start_time = datetime.now()
    print(f"=== Paper Trader Lambda | {start_time.isoformat()} ===")
    print(f"Strategy: {STRATEGY}")
    print(f"Bucket: {BUCKET_NAME}")
    
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
        
        # 3. Run the trading logic
        from main import main as run_trading
        import sys
        
        # Set arguments for main.py
        sys.argv = ['main.py', '--mode', 'trade', '--strategy', STRATEGY, '--portfolio', STRATEGY]
        
        # Run trading
        run_trading()
        
        # 4. Change back and upload results
        os.chdir(original_cwd)
        upload_to_s3()
        
        elapsed = (datetime.now() - start_time).total_seconds()
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': f'Trading completed successfully',
                'strategy': STRATEGY,
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
