import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import os
from datetime import datetime, timedelta

# 1. 加載訓練數據
def load_training_data(data_path='data/'):
    train_files = [f'L{i}_Train.csv' for i in range(1, 18)]
    data_frames = []
    for file in train_files:
        df = pd.read_csv(os.path.join(data_path, file))
        data_frames.append(df)
    train_data = pd.concat(data_frames, ignore_index=True)
    return train_data

# 2. 數據預處理
def preprocess_data(train_data):
    # 2.1 處理日期時間
    if 'DateTime' in train_data.columns:
        train_data['DateTime'] = pd.to_datetime(train_data['DateTime'])
        train_data['Year'] = train_data['DateTime'].dt.year
        train_data['Month'] = train_data['DateTime'].dt.month
        train_data['Day'] = train_data['DateTime'].dt.day
        train_data['Hour'] = train_data['DateTime'].dt.hour
        train_data['Minute'] = train_data['DateTime'].dt.minute
        train_data['Weekday'] = train_data['DateTime'].dt.weekday  # 星期幾
        train_data.drop(columns=['DateTime'], inplace=True)
    else:
        print("訓練數據中缺少 'DateTime' 欄位。")
    
    # 2.2 創建交互特徵
    if 'Temperature(°C)' in train_data.columns and 'Humidity(%)' in train_data.columns:
        train_data['Temp_Humidity'] = train_data['Temperature(°C)'] * train_data['Humidity(%)']
    else:
        print("訓練數據中缺少 'Temperature(°C)' 或 'Humidity(%)' 欄位。")
    
    return train_data

# 3. 處理缺失值與異常值
def handle_missing_values(train_data):
    # 3.1 處理光照度缺陷
    if 'Sunlight(Lux)' in train_data.columns:
        MAX_SUNLIGHT = 117758.2
        train_data['Sunlight_Max'] = train_data['Sunlight(Lux)'] >= MAX_SUNLIGHT
        train_data.loc[train_data['Sunlight_Max'], 'Sunlight(Lux)'] = np.nan
    else:
        print("訓練數據中缺少 'Sunlight(Lux)' 欄位。")
    
    # 3.2 填補風速異常值 (風速計異常時風速為0，使用中位數填補)
    if 'WindSpeed(m/s)' in train_data.columns:
        train_data['WindSpeed(m/s)'].replace(0, np.nan, inplace=True)
        wind_median = train_data['WindSpeed(m/s)'].median()
        train_data['WindSpeed(m/s)'].fillna(wind_median, inplace=True)
    else:
        print("訓練數據中缺少 'WindSpeed(m/s)' 欄位。")
    
    # 3.3 填補其他缺失值（根據需要）
    for col in ['Humidity(%)', 'Temperature(°C)', 'Pressure(hpa)']:
        if col in train_data.columns:
            train_data[col].fillna(train_data[col].median(), inplace=True)
        else:
            print(f"訓練數據中缺少 '{col}' 欄位。")
    
    return train_data

# 4. 建立預測光照度模型
def train_sunlight_model(train_data):
    if 'Sunlight(Lux)' not in train_data.columns:
        print("訓練數據中缺少 'Sunlight(Lux)' 欄位。無法訓練光照度預測模型。")
        return None, None

    # 分離有Sunlight的數據和缺失Sunlight的數據
    sunlight_train = train_data[~train_data['Sunlight(Lux)'].isna()]
    
    # 定義特徵
    sunlight_features = ['Humidity(%)', 'Temperature(°C)', 'Pressure(hpa)', 'WindSpeed(m/s)', 'Temp_Humidity']
    
    X_sun = sunlight_train[sunlight_features]
    y_sun = sunlight_train['Sunlight(Lux)']
    
    # 分割訓練集和驗證集
    X_train_sun, X_val_sun, y_train_sun, y_val_sun = train_test_split(
        X_sun, y_sun, test_size=0.2, random_state=42
    )
    
    # 建立XGBoost模型
    xgb_sun = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    
    # 訓練模型
    xgb_sun.fit(X_train_sun, y_train_sun)
    
    # 預測驗證集
    y_pred_sun = xgb_sun.predict(X_val_sun)
    
    # 評估模型
    mse_sun = mean_squared_error(y_val_sun, y_pred_sun)
    print(f'光照度預測模型的均方誤差（MSE）: {mse_sun:.2f}')
    
    return xgb_sun, sunlight_features

# 5. 填補缺失的光照度
def fill_missing_sunlight(train_data, xgb_sun, sunlight_features):
    if xgb_sun is None:
        return train_data
    
    sunlight_missing = train_data['Sunlight(Lux)'].isna()
    if sunlight_missing.sum() > 0:
        X_missing = train_data.loc[sunlight_missing, sunlight_features]
        predicted_sunlight = xgb_sun.predict(X_missing)
        train_data.loc[sunlight_missing, 'Sunlight(Lux)'] = predicted_sunlight
        train_data['Sunlight_Max'] = train_data['Sunlight_Max'].astype(int)
        print(f'已填補 {sunlight_missing.sum()} 筆缺失的 Sunlight(Lux) 數據。')
    else:
        print("沒有缺失的 Sunlight(Lux) 數據需要填補。")
    
    return train_data

# 6. 建立預測發電量模型
def train_power_model(train_data):
    if 'Power(mW)' not in train_data.columns:
        print("訓練數據中缺少 'Power(mW)' 欄位。無法訓練發電量預測模型。")
        return None, None, None, None

    # 定義目標變數和特徵
    target = 'Power(mW)'
    features_power = ['Sunlight(Lux)']
    
    X_power = train_data[features_power]
    y_power = train_data[target]
    
    # 分割訓練集和驗證集
    X_train_power, X_val_power, y_train_power, y_val_power = train_test_split(
        X_power, y_power, test_size=0.2, random_state=42
    )
    
    # 建立XGBoost模型
    xgb_power = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
    
    # 訓練模型
    xgb_power.fit(X_train_power, y_train_power)
    
    # 預測驗證集
    y_pred_power = xgb_power.predict(X_val_power)
    
    # 評估模型
    mse_power = mean_squared_error(y_val_power, y_pred_power)
    print(f'發電量預測模型的均方誤差（MSE）: {mse_power:.2f}')
    
    return xgb_power, features_power, X_train_power, y_train_power

# 7. 呼叫API獲取即時氣象數據並顯示結果
def fetch_api_data(API_KEY, STATION_ID):
    BASE_URL = 'https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-A0003-001'
    
    params = {
        'Authorization': API_KEY,
        'StationId': STATION_ID
    }
    
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if data.get('success') == "true":
            station_data = data['records']['Station'][0]
    
            # 提取測站名稱和觀測時間
            station_name = station_data.get('StationName', '未知')
            obs_time = station_data.get('ObsTime', {}).get('DateTime', '未知')
    
            # 提取天氣資料並刪除不需要的變數
            weather_elements = station_data.get('WeatherElement', {})
            # 移除不需要的變數: Weather, VisibilityDescription, Precipitation, WindDirection, UVIndex, MaxGustSpeed
            weather_data = {
                'StationName': station_name,
                'ObsTime': obs_time,
                'Humidity(%)': weather_elements.get('RelativeHumidity', 0.0),
                'Temperature(°C)': weather_elements.get('AirTemperature', 0.0),
                'Pressure(hpa)': weather_elements.get('AirPressure', 0.0),
                'WindSpeed(m/s)': weather_elements.get('WindSpeed', 0.0)
            }
    
            # 顯示API回傳結果
            print("\nAPI回傳結果:")
            for key, value in weather_data.items():
                print(f"{key}: {value}")
    
            return weather_data
        else:
            print(f"資料取得失敗，錯誤信息：{data.get('msg')}")
            return None
    else:
        print(f"無法取得天氣資料，錯誤代碼: {response.status_code}")
        return None

# 8. 根據歷史數據計算每個時間段的平均氣象特徵
def calculate_historical_averages(train_data):
    # 計算每個時間段（Hour, Minute）的平均氣象特徵
    average_weather = train_data.groupby(['Hour', 'Minute']).agg({
        'Humidity(%)': 'median',
        'Temperature(°C)': 'median',
        'Pressure(hpa)': 'median',
        'WindSpeed(m/s)': 'median',
        'Temp_Humidity': 'median'
    }).reset_index()
    return average_weather

# 9. 生成預測數據
def generate_prediction_data(prediction_date, average_weather, xgb_sun, features_power, xgb_power, train_data):
    if xgb_sun is None or xgb_power is None:
        print("模型未訓練完成。")
        return None
    
    # 生成從09:00到16:59，每10分鐘一筆，共48筆
    start_time = datetime.strptime(prediction_date + " 09:00", "%Y-%m-%d %H:%M")
    time_slots = [start_time + timedelta(minutes=10*i) for i in range(48)]
    
    prediction_data = []
    for slot in time_slots:
        hour = slot.hour
        minute = slot.minute
        # 找到對應的歷史平均氣象特徵
        avg_weather = average_weather[
            (average_weather['Hour'] == hour) & (average_weather['Minute'] == minute)
        ]
        if not avg_weather.empty:
            avg_humidity = avg_weather['Humidity(%)'].values[0]
            avg_temperature = avg_weather['Temperature(°C)'].values[0]
            avg_pressure = avg_weather['Pressure(hpa)'].values[0]
            avg_wind_speed = avg_weather['WindSpeed(m/s)'].values[0]
            avg_temp_humidity = avg_weather['Temp_Humidity'].values[0]
        else:
            # 如果沒有對應的歷史數據，使用整體中位數
            avg_humidity = train_data['Humidity(%)'].median()
            avg_temperature = train_data['Temperature(°C)'].median()
            avg_pressure = train_data['Pressure(hpa)'].median()
            avg_wind_speed = train_data['WindSpeed(m/s)'].median()
            avg_temp_humidity = train_data['Temp_Humidity'].median()
        
        # 構建數據點
        data_point = {
            'ObsTime': slot,
            'Humidity(%)': avg_humidity,
            'Temperature(°C)': avg_temperature,
            'Pressure(hpa)': avg_pressure,
            'WindSpeed(m/s)': avg_wind_speed,
            'Temp_Humidity': avg_temp_humidity
        }
        prediction_data.append(data_point)
    
    weather_df = pd.DataFrame(prediction_data)
    
    # 轉換觀測時間為 datetime 格式
    weather_df['ObsTime'] = pd.to_datetime(weather_df['ObsTime'], errors='coerce')
    
    # 提取時間特徵
    weather_df['Year'] = weather_df['ObsTime'].dt.year
    weather_df['Month'] = weather_df['ObsTime'].dt.month
    weather_df['Day'] = weather_df['ObsTime'].dt.day
    weather_df['Hour'] = weather_df['ObsTime'].dt.hour
    weather_df['Minute'] = weather_df['ObsTime'].dt.minute
    weather_df['Weekday'] = weather_df['ObsTime'].dt.weekday  # 星期幾
    
    # 編碼地點代號
    LOCATION_CODE = 17  # 根據需要修改
    weather_df['LocationCode'] = LOCATION_CODE
    
    # 獨熱編碼 LocationCode
    weather_df = pd.get_dummies(weather_df, columns=['LocationCode'], prefix='Loc')
    
    # 確保所有LocationCode的獨熱編碼與訓練數據一致
    for col in [f'Loc_{i}' for i in range(1, 18)]:
        if col not in weather_df.columns:
            weather_df[col] = 0
    
    # 填補風速異常值（風速為0時填補為訓練數據的中位數）
    if 'WindSpeed(m/s)' in weather_df.columns:
        wind_median = train_data['WindSpeed(m/s)'].median()
        weather_df['WindSpeed(m/s)'].replace(0, wind_median, inplace=True)
        weather_df['WindSpeed(m/s)'].fillna(wind_median, inplace=True)
    else:
        weather_df['WindSpeed(m/s)'] = train_data['WindSpeed(m/s)'].median()
    
    # 確保所有特徵都存在
    for feature in features_power:
        if feature not in weather_df.columns:
            weather_df[feature] = 0
    
    # 預測Sunlight(Lux)
    X_sun_new = weather_df[['Humidity(%)', 'Temperature(°C)', 'Pressure(hpa)', 'WindSpeed(m/s)', 'Temp_Humidity']]
    predicted_sunlight = xgb_sun.predict(X_sun_new)
    weather_df['Sunlight(Lux)'] = predicted_sunlight
    
    # 預測Power(mW)
    X_power_new = weather_df[['Sunlight(Lux)']]
    predicted_power = xgb_power.predict(X_power_new)
    weather_df['Power(mW)'] = predicted_power
    
    # 生成序號
    # 序號格式：西元年(4碼)+月(2碼)+日(2碼)+預測時間(4碼，HHMM)+裝置代號(2碼)
    # 裝置代號固定為17
    weather_df['序號'] = weather_df.apply(
        lambda row: f"{row['Year']:04d}{row['Month']:02d}{row['Day']:02d}"
                    f"{row['Hour']:02d}{int(row['Minute']/10)*10:02d}{LOCATION_CODE:02d}",
        axis=1
    )
    
    # 選擇需要上傳的欄位
    upload_df = weather_df[['序號', 'Power(mW)']]
    
    # 四捨五入至小數點後兩位
    upload_df['Power(mW)'] = upload_df['Power(mW)'].round(2)
    
    return upload_df

# 10. 保存預測結果
def save_upload_csv(upload_df):
    upload_df.to_csv('upload.csv', index=False)
    print("\n預測結果已保存為 upload.csv")

# 11. 主程式
def main():
    # 設定API資訊
    API_KEY = 'CWA-38B857C1-4735-414D-8978-FC2D2720C380'  # 替換為你的 API Key
    STATION_ID = '466881'  # 替換為你想查詢的測站ID
    
    # 讀取和預處理訓練數據
    train_data = load_training_data(data_path='data/')
    train_data = preprocess_data(train_data)
    train_data = handle_missing_values(train_data)
    
    # 計算歷史平均氣象特徵
    average_weather = calculate_historical_averages(train_data)
    
    # 訓練光照度預測模型
    xgb_sun, sunlight_features = train_sunlight_model(train_data)
    
    # 填補缺失的光照度
    train_data = fill_missing_sunlight(train_data, xgb_sun, sunlight_features)
    
    # 訓練發電量預測模型
    xgb_power, features_power, X_train_power, y_train_power = train_power_model(train_data)
    
    # 可選：超參數調整
    # best_xgb_power = optimize_power_model(xgb_power, X_train_power, y_train_power)
    # 此處未進行超參數調整，若需要請取消註解並調整
    best_xgb_power = xgb_power  # 若不進行調整，使用原模型
    
    # 獲取API數據並顯示結果
    weather_data = fetch_api_data(API_KEY, STATION_ID)
    
    # 設定預測日期
    prediction_date = '2024-11-12'  # 修改為您需要的預測日期
    
    # 生成預測數據
    upload_df = generate_prediction_data(prediction_date, average_weather, xgb_sun, features_power, best_xgb_power, train_data)
    
    # 保存預測結果
    save_upload_csv(upload_df)

if __name__ == "__main__":
    main()
