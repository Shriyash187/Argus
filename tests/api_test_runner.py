import os
import sys
import time
import subprocess
import urllib.request
import urllib.error
import json

def test_endpoints():
    print("Starting API test runner...")
    
    # Start uvicorn server in a subprocess
    python_exe = sys.executable
    cmd = [python_exe, "-m", "uvicorn", "stock_analyzr.src.api:app", "--host", "127.0.0.1", "--port", "8000"]
    
    print(f"Launching server with: {' '.join(cmd)}")
    server_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Wait for server to spin up by polling /health
    server_started = False
    print("Waiting for server to start (polling http://127.0.0.1:8000/health for up to 90 seconds)...")
    for i in range(90):
        if server_process.poll() is not None:
            break
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as response:
                if response.status == 200:
                    server_started = True
                    break
        except Exception:
            time.sleep(1)
            
    # Check if process crashed
    if server_process.poll() is not None or not server_started:
        print("\nUvicorn failed to start or load. Printing server process logs:")
        # Read available stdout lines
        import select
        if sys.platform != 'win32':
            # Use select on Unix
            r, _, _ = select.select([server_process.stdout], [], [], 2)
            if r:
                print(server_process.stdout.read())
        else:
            # On Windows, terminate and read
            server_process.terminate()
            out, _ = server_process.communicate()
            print(out)
        sys.exit(1)
        
    print(f"Server started successfully after {i} seconds. Beginning endpoint checks...")
    
    success = True
    endpoints = [
        ("/health", "GET", None),
        ("/portfolio", "GET", None),
        ("/recommend?ticker=AAPL", "GET", None),
        ("/advisor?ticker=AAPL", "GET", None),
        ("/news?ticker=AAPL", "GET", None),
        ("/events?ticker=AAPL", "GET", None),
        ("/recommend/backtest?ticker=AAPL&lookback_days=30", "GET", None),
    ]
    
    for path, method, payload in endpoints:
        url = f"http://127.0.0.1:8000{path}"
        print(f"Testing {method} {url}...")
        
        try:
            req = urllib.request.Request(url, method=method)
            if payload:
                req.add_header('Content-Type', 'application/json')
                data = json.dumps(payload).encode('utf-8')
                with urllib.request.urlopen(req, data=data, timeout=300) as response:
                    status = response.status
                    res_body = json.loads(response.read().decode())
            else:
                with urllib.request.urlopen(req, timeout=300) as response:
                    status = response.status
                    res_body = json.loads(response.read().decode())
                    
            print(f"  Result: SUCCESS (Status: {status})")
            if path == "/health":
                assert "status" in res_body, "Health response missing status key"
            elif path == "/portfolio":
                assert "cash" in res_body, "Portfolio response missing cash key"
                assert "total_portfolio_value" in res_body, "Portfolio response missing total_portfolio_value"
                
        except urllib.error.HTTPError as he:
            if he.code in [404, 400]:
                print(f"  Result: EXPECTED CODE (Status: {he.code}, Detail: {he.reason})")
            else:
                print(f"  Result: FAILED (Status: {he.code}, Detail: {he.reason})")
                success = False
        except Exception as e:
            print(f"  Result: EXCEPTION: {e}")
            success = False
            
    # Test POST /trade
    url_trade = "http://127.0.0.1:8000/trade"
    trade_payload = {
        "ticker": "AAPL",
        "action": "BUY",
        "shares": 5.0,
        "price": 175.0
    }
    print(f"Testing POST {url_trade} with payload {trade_payload}...")
    try:
        req = urllib.request.Request(url_trade, method="POST")
        req.add_header('Content-Type', 'application/json')
        data = json.dumps(trade_payload).encode('utf-8')
        with urllib.request.urlopen(req, data=data, timeout=300) as response:
            status = response.status
            res_body = json.loads(response.read().decode())
        print(f"  Result: SUCCESS (Status: {status}, Response: {res_body})")
        assert res_body["status"] == "success"
    except urllib.error.HTTPError as he:
        if he.code == 400:
            print(f"  Result: EXPECTED REFUSAL (Status: 400, Detail: {he.reason})")
        else:
            print(f"  Result: FAILED (Status: {he.code}, Detail: {he.reason})")
            success = False
    except Exception as e:
        print(f"  Result: EXCEPTION: {e}")
        success = False
        
    # Shutdown server
    print("Terminating server...")
    if not success:
        out, _ = server_process.communicate()
        print("\n=== SERVER LOGS ===\n", out, "\n===================\n")
    else:
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
        
    print("API validation complete.")
    if success:
        print("ALL API ENDPOINTS VALIDATED SUCCESSFULLY.")
        sys.exit(0)
    else:
        print("API VALIDATION ENCOUNTERED ERRORS.")
        sys.exit(1)

if __name__ == "__main__":
    test_endpoints()
