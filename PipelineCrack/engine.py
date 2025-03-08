def get_signal():
  return 0


async def get_crack_result():
  signal = await get_signal()
  result = model.predict(signal)
  return result