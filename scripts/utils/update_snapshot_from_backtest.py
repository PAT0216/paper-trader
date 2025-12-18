import json
import datetime
import os

BACKTEST_RESULTS = "results/pit_backtest_oct_dec_2025.json"
SNAPSHOT_FILE = "data/portfolio_snapshot.json"

def main():
    try:
        if not os.path.exists(BACKTEST_RESULTS):
            print(f"Error: Results file not found at {BACKTEST_RESULTS}")
            exit(1)
            
        with open(BACKTEST_RESULTS, "r") as f:
            results = json.load(f)
        
        with open(SNAPSHOT_FILE, "r") as f:
            snapshot = json.load(f)
        
        # Update ML portfolio
        if "ml" not in snapshot["portfolios"]:
            snapshot["portfolios"]["ml"] = {}
            
        ml_data = snapshot["portfolios"]["ml"]
        ml_data["value"] = round(results["final_value"], 2)
        ml_data["return_pct"] = round(results["total_return"] * 100, 2)
        ml_data["cash"] = round(results["final_cash"], 2)
        ml_data["holdings"] = results["final_holdings"]
        ml_data["positions"] = len(results["final_holdings"])
        ml_data["note"] = f"Values from aligned ML model PIT backtest ({results['start_date']} - {results['end_date']})"
        
        # Update timestamp
        snapshot["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"
        snapshot["price_date"] = results["end_date"]
        
        # Update benchmark if available
        if "spy_return" in results and "benchmark" in snapshot:
            snapshot["benchmark"]["return_pct"] = round(results["spy_return"] * 100, 2)
            # Estimate value based on return (assuming $10k start) - logic matches previous manual entries
            snapshot["benchmark"]["value"] = round(10000 * (1 + results["spy_return"]), 2)
            snapshot["benchmark"]["end_date"] = results["end_date"]

        with open(SNAPSHOT_FILE, "w") as f:
            json.dump(snapshot, f, indent=2)
            
        print(f"Successfully updated {SNAPSHOT_FILE} from {BACKTEST_RESULTS}")
        print(f"ML Value: ${ml_data['value']}, Return: {ml_data['return_pct']}%")
        
    except Exception as e:
        print(f"Error updating snapshot: {e}")
        exit(1)

if __name__ == "__main__":
    main()
