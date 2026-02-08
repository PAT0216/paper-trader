"""
AWS Lambda Handler for Paper Trader
-----------------------------------
Entry point for Lambda invocations.
Downloads data from S3, runs trading, uploads results back to S3.
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


def cleanup_tmp():
    """Clean up /tmp to handle Lambda warm container reuse"""
    for item in ['/tmp/market.db', '/tmp/data']:
        if os.path.exists(item):
            if os.path.isdir(item):
                shutil.rmtree(item)
            else:
                os.remove(item)
    print("Cleaned up /tmp")

def download_from_s3():
    """Download market.db and ledgers from S3"""
    try:
        # Download market cache
        s3.download_file(BUCKET_NAME, 'market.db', '/tmp/market.db')
        os.makedirs('data', exist_ok=True)
        # Create symlink only if it doesn't exist
        if not os.path.exists('data/market.db'):
            os.symlink('/tmp/market.db', 'data/market.db')
        print(f"Downloaded market.db from S3")
    except Exception as e:
        print(f"No market.db in S3 (first run?): {e}")
    
    try:
        # Download ledger
        ledger_key = f'ledgers/ledger_{STRATEGY}.csv'
        os.makedirs('data/ledgers', exist_ok=True)
        s3.download_file(BUCKET_NAME, ledger_key, f'data/ledgers/ledger_{STRATEGY}.csv')
        print(f"Downloaded ledger from S3")
    except Exception as e:
        print(f"No ledger in S3 (first run?): {e}")


def upload_to_s3():
    """Upload results back to S3"""
    # Upload ledger
    ledger_path = f'data/ledgers/ledger_{STRATEGY}.csv'
    if os.path.exists(ledger_path):
        s3.upload_file(ledger_path, BUCKET_NAME, f'ledgers/ledger_{STRATEGY}.csv')
        print(f"Uploaded ledger to S3")
    
    # Upload snapshot
    snapshot_path = f'data/snapshots/{STRATEGY}.json'
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
        # 0. Clean up /tmp from previous runs (warm containers)
        cleanup_tmp()
        
        # 1. Download data from S3
        download_from_s3()
        
        # 2. Run the trading logic
        # Import here to avoid cold start overhead if handler fails early
        from main import main as run_trading
        import sys
        
        # Set arguments for main.py
        sys.argv = ['main.py', '--mode', 'trade', '--strategy', STRATEGY, '--portfolio', STRATEGY]
        
        # Run trading
        run_trading()
        
        # 3. Upload results to S3
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
