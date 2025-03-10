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
from PipelineCrack.signal1 import *
import joblib
from PipelineCrack.crack_model import run_model,check_correctness;
from sklearn.preprocessing import StandardScaler
#xgb_model = joblib.load("xgboost_model.pkl")
#scaler = joblib.load("scaler.pkl")
loaded_rf_model = joblib.load("PipelineCrack/random_forest_model.pkl")

def get_signal():
    values, time_list = collect_signal()  # Blocking call (runs for 10s)
    return values

async def get_crack_result():
    signal = await asyncio.to_thread(get_signal)  # Run get_signal() in a separate thread
    signal_correalate = check_correctness(signal)
    data = get_features(signal)
    result = run_model(loaded_rf_model,signal_correalate, data)
    return result

async def main():
    result = await get_crack_result()
    print(result)

asyncio.run(main())  # Runs the async event loop