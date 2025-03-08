from signal1 import *


def get_signal():
  values, time_list = collect_signal()
  return get_signal(values)


async def get_crack_result():
  signal = await get_signal()
  result = model.predict(signal)
  return result