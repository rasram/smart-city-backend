import requests
import time

BLYNK_AUTH_TOKEN = "GQT-aWhsxMuj68a51qeqQn-SB5aq7Xem"  # your token
VIRTUAL_PIN = "v0"  # cm distance
url = f"https://blynk.cloud/external/api/get?token={BLYNK_AUTH_TOKEN}&{VIRTUAL_PIN}"
while True:
    try:
        response = requests.get(url)
        if response.status_code == 200:
            distance = float(response.text)
            print("Distance from Blynk:", distance)

            # 🔁 Pass to your model here
            # result = my_model.predict([distance])
            # print("Prediction:", result)

        else:
            print("Failed to fetch data:", response.status_code, response.text)

    except Exception as e:
        print("Error:", e)
    time.sleep(1)  # every 5 seconds

