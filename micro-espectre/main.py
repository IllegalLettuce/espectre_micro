# main.py (root level) - Safe auto-run with cancel option
import time
import gc
import sys

# wait 5 secs to start (avoids some odd issues I can't figure out)
try:
    for i in range(5, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
except KeyboardInterrupt:
    # cancel
    sys.exit()

# Free memory before starting
gc.collect()

try:
    from src.main import main
    main()
except KeyboardInterrupt:
    print("\nApplication stopped by user")
except Exception as e:
    print(f"\nApplication error: {e}")
    sys.print_exception(e)
