async def get_crack_result():
  signal = await get_signal()
  result = model.predict(signal)
  return result