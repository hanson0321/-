import requests

# 設定 API 的基本資訊
API_KEY = 'CWA-38B857C1-4735-414D-8978-FC2D2720C380'  # 替換為你的 API Key
BASE_URL = 'https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-A0003-001'
STATION_ID = '花蓮市'  # 你想要查詢的測站ID

# 建立查詢參數
params = {
    'Authorization': API_KEY,
    'StationId': STATION_ID
}

# 發送 GET 請求
response = requests.get(BASE_URL, params=params)

# 檢查請求是否成功
if response.status_code == 200:
    data = response.json()
    if data.get('success') == "true":
        station_data = data['records']['Station'][0]
        
        # 提取測站名稱和觀測時間
        station_name = station_data['StationName']
        obs_time = station_data['ObsTime']['DateTime']
        
        # 提取天氣資料
        weather_element = station_data['WeatherElement']
        weather = weather_element.get('Weather', '未知')
        visibility_desc = weather_element.get('VisibilityDescription', '未知')
        sunshine_duration = weather_element.get('SunshineDuration', '未知')
        precipitation = weather_element.get('Now', {}).get('Precipitation', '未知')
        wind_direction = weather_element.get('WindDirection', '未知')
        wind_speed = weather_element.get('WindSpeed', '未知')
        air_temperature = weather_element.get('AirTemperature', '未知')
        relative_humidity = weather_element.get('RelativeHumidity', '未知')
        air_pressure = weather_element.get('AirPressure', '未知')
        uv_index = weather_element.get('UVIndex', '未知')
        peak_gust_speed = weather_element.get('GustInfo', {}).get('PeakGustSpeed', '未知')

        # 顯示結果
        print(f"測站名稱: {station_name}")
        print(f"觀測時間: {obs_time}")
        print(f"天氣狀況: {weather}")
        print(f"能見度描述: {visibility_desc}")
        print(f"日照時數: {sunshine_duration} 小時")
        print(f"降雨量: {precipitation} mm")
        print(f"風向: {wind_direction} 度")
        print(f"風速: {wind_speed} m/s")
        print(f"氣溫: {air_temperature} °C")
        print(f"相對濕度: {relative_humidity} %")
        print(f"氣壓: {air_pressure} hPa")
        print(f"紫外線指數: {uv_index}")
        print(f"最大陣風速度: {peak_gust_speed} m/s")
        
    else:
        print(f"資料取得失敗，錯誤信息：{data.get('msg')}")
else:
    print(f"無法取得天氣資料，錯誤代碼: {response.status_code}")
