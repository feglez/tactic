# async_runner.py

import asyncio
import threading

# Create an independent event loop
loop = asyncio.new_event_loop()

# Start the loop in a separate thread
def start_loop():
    asyncio.set_event_loop(loop)
    loop.run_forever()

# Launch the loop in a background thread
t = threading.Thread(target=start_loop, daemon=True)
t.start()

# Safely run any coroutine on that loop
def run_async_task(coro):
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()  # Wait for completion and return the result
