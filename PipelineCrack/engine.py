""" import asyncio
from signal1 import *

def get_signal():
    values, time_list = collect_signal()
    return values

async def get_crack_result():
    signal = await get_signal()
    return signal

# Properly await the async function
async def main():
    result = await get_crack_result()
    print(result)

asyncio.run(main())  # This ensures the async function runs properly
 """


import asyncio
from signal1 import *

def get_signal():
    values, time_list = collect_signal()  # Blocking call (runs for 10s)
    return values

async def get_crack_result():
    signal = await asyncio.to_thread(get_signal)  # Run get_signal() in a separate thread
    return signal

async def main():
    result = await get_crack_result()
    print(result)

asyncio.run(main())  # Runs the async event loop