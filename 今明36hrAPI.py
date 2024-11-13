import requests

API_KEY = 'CWA-38B857C1-4735-414D-8978-FC2D2720C380'
BASE_URL = 'https://opendata.cwa.gov.tw/api/v1/rest/datastore/F-C0032-001'
LOCATION_NAME = '臺北市'  

params = {
    'Authorization': API_KEY,
    'locationName': LOCATION_NAME
}

# 發送 GET 請求
response = requests.get(BASE_URL, params=params)

# 檢查請求是否成功
if response.status_code == 200:
    data = response.json()
    if data.get('success') == "true":
        location_data = data['records']['location'][0]
        
        print(f"城市: {location_data['locationName']}")
        
        # 解析天氣元素資料
        for element in location_data['weatherElement']:
            element_name = element['elementName']
            
            # 根據不同的元素來顯示相應的資料
            for time_info in element['time']:
                start_time = time_info['startTime']
                end_time = time_info['endTime']
                parameter = time_info['parameter']
                
                if element_name == "Wx":
                    weather_desc = parameter['parameterName']
                    print(f"時間: {start_time} 到 {end_time}")
                    print(f"天氣描述: {weather_desc}")
                
                elif element_name == "PoP":
                    pop = parameter['parameterName']
                    print(f"降雨機率: {pop}{parameter.get('parameterUnit', '')}")

                elif element_name == "MinT":
                    min_temp = parameter['parameterName']
                    print(f"最低溫度: {min_temp}{parameter.get('parameterUnit', '')}")
                
                elif element_name == "MaxT":
                    max_temp = parameter['parameterName']
                    print(f"最高溫度: {max_temp}{parameter.get('parameterUnit', '')}")
                
                elif element_name == "CI":
                    comfort_index = parameter['parameterName']
                    print(f"舒適度: {comfort_index}")

                print("------")
    else:
        print(f"資料取得失敗，錯誤信息：{data.get('msg')}")
else:
    print(f"無法取得天氣資料，錯誤代碼: {response.status_code}")
