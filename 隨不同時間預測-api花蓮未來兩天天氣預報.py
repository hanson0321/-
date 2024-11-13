import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import os
from datetime import datetime, timedelta
from tqdm import tqdm  # 引入 tqdm

# 1. 加載訓練數據
def load_training_data(data_path='data/'):
    train_files = [f'L{i}_Train.csv' for i in range(1, 18)]
    data_frames = []
    print("正在加載訓練數據...")
    for file in tqdm(train_files, desc="加載CSV文件"):
        try:
            df = pd.read_csv(os.path.join(data_path, file))
            data_frames.append(df)
        except FileNotFoundError:
            print(f"警告: 找不到文件 {file}。請檢查文件路徑。")
    train_data = pd.concat(data_frames, ignore_index=True)
    print("訓練數據加載完成。")
    return train_data

# 2. 數據預處理
def preprocess_data(train_data):
    print("正在進行數據預處理...")
    # 2.1 處理日期時間
    if 'DateTime' in train_data.columns:
        train_data['DateTime'] = pd.to_datetime(train_data['DateTime'], errors='coerce')
        train_data['Year'] = train_data['DateTime'].dt.year
        train_data['Month'] = train_data['DateTime'].dt.month
        train_data['Day'] = train_data['DateTime'].dt.day
        train_data['Hour'] = train_data['DateTime'].dt.hour
        train_data['Minute'] = train_data['DateTime'].dt.minute
        train_data['Weekday'] = train_data['DateTime'].dt.weekday  # 星期幾
        train_data = train_data.drop(columns=['DateTime'])
    else:
        print("訓練數據中缺少 'DateTime' 欄位。")
    
    # 2.2 創建交互特徵
    if 'Temperature(°C)' in train_data.columns and 'Humidity(%)' in train_data.columns:
        train_data['Temp_Humidity'] = train_data['Temperature(°C)'] * train_data['Humidity(%)']
    else:
        print("訓練數據中缺少 'Temperature(°C)' 或 'Humidity(%)' 欄位。")
    
    # 2.3 獨熱編碼 LocationCode
    if 'LocationCode' in train_data.columns:
        train_data = pd.get_dummies(train_data, columns=['LocationCode'], prefix='Loc')
    else:
        print("訓練數據中缺少 'LocationCode' 欄位。")
    
    print("數據預處理完成。")
    return train_data

# 3. 處理缺失值與異常值
def handle_missing_values(train_data):
    print("正在處理缺失值與異常值...")
    # 3.1 處理光照度缺陷
    if 'Sunlight(Lux)' in train_data.columns:
        MAX_SUNLIGHT = 117758.2
        train_data.loc[train_data['Sunlight(Lux)'] >= MAX_SUNLIGHT, 'Sunlight(Lux)'] = np.nan
    else:
        print("訓練數據中缺少 'Sunlight(Lux)' 欄位。")
    
    # 3.2 填補風速異常值 (風速計異常時風速為0，使用中位數填補)
    if 'WindSpeed(m/s)' in train_data.columns:
        train_data['WindSpeed(m/s)'] = train_data['WindSpeed(m/s)'].replace(0, np.nan)
        wind_median = train_data['WindSpeed(m/s)'].median()
        train_data['WindSpeed(m/s)'] = train_data['WindSpeed(m/s)'].fillna(wind_median)
    else:
        print("訓練數據中缺少 'WindSpeed(m/s)' 欄位。")
    
    # 3.3 填補其他缺失值（根據需要）
    for col in ['Humidity(%)', 'Temperature(°C)', 'Pressure(hpa)']:
        if col in train_data.columns:
            train_data[col] = train_data[col].fillna(train_data[col].median())
        else:
            print(f"訓練數據中缺少 '{col}' 欄位。")
    
    print("缺失值與異常值處理完成。")
    return train_data

# 4. 填補缺失的光照度
def fill_missing_sunlight(train_data, xgb_sun, sunlight_features):
    if xgb_sun is None:
        print("光照度模型未訓練，無法填補缺失值。")
        return train_data
    
    missing_sunlight = train_data['Sunlight(Lux)'].isna()
    if missing_sunlight.sum() > 0:
        print(f"正在填補 {missing_sunlight.sum()} 筆缺失的 Sunlight(Lux) 數據...")
        X_missing = train_data.loc[missing_sunlight, sunlight_features]
        predicted_sunlight = xgb_sun.predict(X_missing)
        train_data.loc[missing_sunlight, 'Sunlight(Lux)'] = predicted_sunlight
        print("缺失的 Sunlight(Lux) 數據已填補。")
    else:
        print("訓練數據中沒有缺失的 Sunlight(Lux) 數據需要填補。")
    
    return train_data

# 5. 建立預測光照度模型
def train_sunlight_model(train_data, sunlight_features):
    if 'Sunlight(Lux)' not in train_data.columns:
        print("訓練數據中缺少 'Sunlight(Lux)' 欄位。無法訓練光照度預測模型。")
        return None
    
    # 分離有Sunlight的數據和缺失Sunlight的數據
    sunlight_train = train_data[~train_data['Sunlight(Lux)'].isna()]
    
    X_sun = sunlight_train[sunlight_features]
    y_sun = sunlight_train['Sunlight(Lux)']
    
    # 建立XGBoost模型
    xgb_sun = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42
    )
    
    # 定義超參數範圍
    param_grid_sun = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [6, 8],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    # 使用 GridSearchCV 進行超參數調整
    print("正在訓練光照度預測模型...")
    grid_search_sun = GridSearchCV(
        estimator=xgb_sun,
        param_grid=param_grid_sun,
        scoring='neg_mean_squared_error',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    
    grid_search_sun.fit(X_sun, y_sun)
    
    best_xgb_sun = grid_search_sun.best_estimator_
    print(f'最佳光照度模型參數: {grid_search_sun.best_params_}')
    
    # 交叉驗證評估
    cv_scores_sun = cross_val_score(best_xgb_sun, X_sun, y_sun, cv=5, scoring='neg_mean_squared_error')
    print(f'光照度模型的交叉驗證均方誤差（MSE）: {-cv_scores_sun.mean():.2f}')
    
    return best_xgb_sun

# 6. 建立預測發電量模型
def train_power_model(train_data, features_power):
    if 'Power(mW)' not in train_data.columns:
        print("訓練數據中缺少 'Power(mW)' 欄位。無法訓練發電量預測模型。")
        return None
    
    X_power = train_data[features_power]
    y_power = train_data['Power(mW)']
    
    # 建立XGBoost模型
    xgb_power = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42
    
    )
    
    # 定義超參數範圍
    param_grid_power = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05],
    'max_depth': [6],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}
    
    # 使用 GridSearchCV 進行超參數調整
    print("正在訓練發電量預測模型...")
    grid_search_power = GridSearchCV(
        estimator=xgb_power,
        param_grid=param_grid_power,
        scoring='neg_mean_squared_error',
        cv=5,
        verbose=1,
        n_jobs=-1
    )
    
    grid_search_power.fit(X_power, y_power)
    
    best_xgb_power = grid_search_power.best_estimator_
    print(f'最佳發電量模型參數: {grid_search_power.best_params_}')
    
    # 交叉驗證評估
    cv_scores_power = cross_val_score(best_xgb_power, X_power, y_power, cv=5, scoring='neg_mean_squared_error')
    print(f'發電量模型的交叉驗證均方誤差（MSE）: {-cv_scores_power.mean():.2f}')
    
    return best_xgb_power

# 7. 呼叫API獲取未來氣象預測數據
def fetch_api_data(API_KEY, STATION_ID):
    BASE_URL = 'https://opendata.cwa.gov.tw/api/v1/rest/datastore/F-D0047-041'  # 更新為正確的 resource_id
    
    params = {
        'Authorization': API_KEY
    }
    
    print("正在呼叫API獲取未來氣象預測數據...")
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        try:
            data = response.json()
        except ValueError:
            print("API 回傳的不是有效的 JSON 格式。")
            return None
        
        if data.get('success') == "true":
            locations = data.get('records', {}).get('locations', [])
            if not locations:
                print("API 回傳的 'locations' 列表為空。")
                print("完整的 API 回傳資料如下:")
                print(data)
                return None
            
            # 尋找目標測站
            target_location = None
            for loc in locations:
                if loc.get('dataid') == STATION_ID:
                    target_location = loc
                    break
            if not target_location:
                print(f"找不到 dataid 為 {STATION_ID} 的測站。")
                return None
            
            # 提取各個 weatherElement
            weather_elements = target_location.get('location', [])[0].get('weatherElement', [])
            weather_data = {}
            for element in weather_elements:
                name = element.get('elementName')
                times = element.get('time', [])
                weather_data[name] = times
            
            # 將各個元素的數據合併到一個 DataFrame
            df_list = []
            for element, times in weather_data.items():
                for time_entry in times:
                    dataTime = time_entry.get('startTime')  # 使用 startTime 作為時間點
                    if element in ['PoP6h', 'PoP12h', 'AT', 'T', 'RH', 'CI', 'Td']:
                        value = time_entry.get('elementValue', [])[0].get('value')
                        if element in ['PoP6h', 'PoP12h', 'AT', 'T', 'RH', 'Td']:
                            try:
                                value = float(value)
                            except:
                                value = np.nan
                        df_list.append({'dataTime': dataTime, element: value})
                    elif element == 'WS':
                        ws = time_entry.get('elementValue', [])[0].get('value')
                        try:
                            ws = float(ws)
                        except:
                            ws = np.nan
                        df_list.append({'dataTime': dataTime, 'WS': ws})
            
            # 創建 DataFrame
            weather_df = pd.DataFrame(df_list)
            weather_df['dataTime'] = pd.to_datetime(weather_df['dataTime'], errors='coerce')
            
            # 合併相同 dataTime 的數據
            weather_df = weather_df.groupby('dataTime').first().reset_index()
            
            # 填補缺失值
            weather_df = weather_df.sort_values('dataTime').reset_index(drop=True)
            weather_df = weather_df.fillna(method='ffill').fillna(method='bfill')
            
            print("API氣象數據獲取並處理完成。")
            return weather_df
        else:
            print(f"資料取得失敗，錯誤信息：{data.get('msg')}")
            return None
    else:
        print(f"無法取得天氣資料，錯誤代碼: {response.status_code}")
        return None

# 8. 生成預測數據
def generate_prediction_data(prediction_date, api_weather, xgb_sun, features_power, xgb_power, train_data, LOCATION_CODE):
    if xgb_sun is None or xgb_power is None:
        print("模型未訓練完成。")
        return None
    
    # 生成從09:00到16:59，每10分鐘一筆，共48筆
    start_time = datetime.strptime(prediction_date + " 09:00", "%Y-%m-%d %H:%M")
    time_slots = [start_time + timedelta(minutes=10*i) for i in range(48)]
    
    prediction_data = []
    print("正在生成預測數據...")
    for slot in tqdm(time_slots, desc="生成預測數據"):
        hour = slot.hour
        minute = slot.minute
        # 尋找最近的 API forecast 時間點
        # 假設 API 提供的時間點為每6小時一次
        # 根據實際API時間點調整此處
        closest_time = slot.replace(minute=0, second=0, microsecond=0)
        # 假設 API 時間點為00:00, 06:00, 12:00, 18:00
        if slot.hour % 6 != 0:
            closest_hour = 6 * (slot.hour // 6)
            closest_time = slot.replace(hour=closest_hour, minute=0, second=0, microsecond=0)
        # 從 API 數據中找到該時間點的氣象數據
        forecast = api_weather[api_weather['dataTime'] == closest_time]
        if not forecast.empty:
            forecast = forecast.iloc[0]
            humidity = forecast.get('RH', train_data['Humidity(%)'].median())
            temperature = forecast.get('T', train_data['Temperature(°C)'].median())
            pressure = train_data['Pressure(hpa)'].median()  # 假設壓力不從API獲取
            wind_speed = forecast.get('WS', train_data['WindSpeed(m/s)'].median())
        else:
            # 如果找不到對應的預測，使用訓練數據的中位數
            humidity = train_data['Humidity(%)'].median()
            temperature = train_data['Temperature(°C)'].median()
            pressure = train_data['Pressure(hpa)'].median()
            wind_speed = train_data['WindSpeed(m/s)'].median()
        
        # 構建數據點
        data_point = {
            'ObsTime': slot,
            'Humidity(%)': humidity,
            'Temperature(°C)': temperature,
            'Pressure(hpa)': pressure,
            'WindSpeed(m/s)': wind_speed,
            'Temp_Humidity': temperature * humidity
        }
        
        prediction_data.append(data_point)
    
    weather_df = pd.DataFrame(prediction_data)
    
    # 提取時間特徵
    weather_df['Year'] = weather_df['ObsTime'].dt.year
    weather_df['Month'] = weather_df['ObsTime'].dt.month
    weather_df['Day'] = weather_df['ObsTime'].dt.day
    weather_df['Hour'] = weather_df['ObsTime'].dt.hour
    weather_df['Minute'] = weather_df['ObsTime'].dt.minute
    weather_df['Weekday'] = weather_df['ObsTime'].dt.weekday  # 星期幾
    
    # 設定 LocationCode
    weather_df['LocationCode'] = LOCATION_CODE
    
    # 獨熱編碼 LocationCode
    weather_df = pd.get_dummies(weather_df, columns=['LocationCode'], prefix='Loc')
    
    # 確保所有 LocationCode 的獨熱編碼與訓練數據一致
    # 假設有 LocationCode 1 到 17
    for i in range(1, 18):
        col = f'Loc_{i}'
        if col not in weather_df.columns:
            weather_df[col] = 0
    
    # 定義 Sunlight(Lux) 模型的特徵（不包含風向）
    sunlight_features = ['Humidity(%)', 'Temperature(°C)', 'Pressure(hpa)', 'WindSpeed(m/s)', 'Temp_Humidity']
    sunlight_features += [f'Loc_{i}' for i in range(1, 18)]
    
    # 預測 Sunlight(Lux)
    X_sun_new = weather_df[sunlight_features]
    predicted_sunlight = xgb_sun.predict(X_sun_new)
    weather_df['Sunlight(Lux)'] = predicted_sunlight
    
    # 定義 Power(mW) 模型的特徵
    features_power_extended = ['Sunlight(Lux)']
    # 如果您希望加入更多特徵，例如 Temperature, Humidity 等，可以在此添加
    # 例如:
    # features_power_extended += ['Temperature(°C)', 'Humidity(%)', 'WindSpeed(m/s)']
    
    # 預測 Power(mW)
    X_power_new = weather_df[features_power_extended]
    predicted_power = xgb_power.predict(X_power_new)
    weather_df['Power(mW)'] = predicted_power
    
    # 生成序號
    # 序號格式：西元年(4碼)+月(2碼)+日(2碼)+預測時間(4碼，HHMM)+裝置代號(2碼)
    # 裝置代號固定為 LocationCode (兩位數)
    weather_df['序號'] = weather_df.apply(
        lambda row: f"{row['Year']:04d}{row['Month']:02d}{row['Day']:02d}"
                    f"{row['Hour']:02d}{int(row['Minute']/10)*10:02d}{LOCATION_CODE:02d}",
        axis=1
    )
    
    # 選擇需要上傳的欄位
    upload_df = weather_df[['序號', 'Power(mW)']].copy()
    
    # 四捨五入至小數點後兩位
    upload_df['Power(mW)'] = upload_df['Power(mW)'].round(2)
    
    print("預測數據生成完成。")


    return upload_df

# 9. 保存預測結果
def save_upload_csv(upload_df):
    upload_df.to_csv('upload.csv', index=False)
    print("\n預測結果已保存為 upload.csv")

# 10. 主程式
def main():
    # 設定 API 資訊
    API_KEY = 'CWA-38B857C1-4735-414D-8978-FC2D2720C380'  # 替換為你的 API Key
    STATION_ID = 'D0047-041'  # 替換為目標測站的 dataid，例如 'D0047-041' 為花蓮縣
    LOCATION_CODE = 1  # 替換為台北的 LocationCode，根據實際情況調整
    
    # 讀取和預處理訓練數據
    train_data = load_training_data(data_path='data/')
    train_data = preprocess_data(train_data)
    train_data = handle_missing_values(train_data)
    
    # 定義 Sunlight(Lux) 模型的特徵（不包含風向）
    sunlight_features = ['Humidity(%)', 'Temperature(°C)', 'Pressure(hpa)', 'WindSpeed(m/s)', 'Temp_Humidity']
    sunlight_features += [f'Loc_{i}' for i in range(1, 18)]
    
    # 訓練光照度預測模型
    xgb_sun = train_sunlight_model(train_data, sunlight_features)
    
    # 填補缺失的光照度
    train_data = fill_missing_sunlight(train_data, xgb_sun, sunlight_features)
    
    # 定義 Power(mW) 模型的特徵
    features_power = ['Sunlight(Lux)']
    # 如果您希望加入更多特徵，例如 Temperature, Humidity 等，可以在此添加
    # 例如:
    # features_power += ['Temperature(°C)', 'Humidity(%)', 'WindSpeed(m/s)']
    
    # 訓練發電量預測模型
    xgb_power = train_power_model(train_data, features_power)
    
    # 呼叫API獲取未來氣象預測數據
    api_weather = fetch_api_data(API_KEY, STATION_ID)
    
    if api_weather is None:
        print("無法獲取有效的氣象數據，程式將結束。")
        return
    
    # 設定預測日期
    prediction_date = '2024-11-12'  # 修改為您需要的預測日期
    
    # 生成預測數據
    upload_df = generate_prediction_data(prediction_date, api_weather, xgb_sun, features_power, xgb_power, train_data, LOCATION_CODE)
    
    if upload_df is not None:
        # 保存預測結果
        save_upload_csv(upload_df)
    else:
        print("預測數據生成失敗。")

if __name__ == "__main__":
    main()
