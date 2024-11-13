import pandas as pd

def load_csv_data(file_paths):
    all_data = []

    for path in file_paths:
        data = pd.read_csv(path)
        all_data.append(data)
    
    combined_data = pd.concat(all_data, ignore_index=True)
    return combined_data

if __name__ == "__main__":
    file_paths = [
        '/Users/yuheng/Desktop/AI cup data forecast/L1_Train.csv',
        '/Users/yuheng/Desktop/AI cup data forecast/L2_Train.csv',
        '/Users/yuheng/Desktop/AI cup data forecast/L3_Train.csv',
        '/Users/yuheng/Desktop/AI cup data forecast/L4_Train.csv',
        '/Users/yuheng/Desktop/AI cup data forecast/L5_Train.csv',
        '/Users/yuheng/Desktop/AI cup data forecast/L6_Train.csv',
        '/Users/yuheng/Desktop/AI cup data forecast/L7_Train.csv',
        '/Users/yuheng/Desktop/AI cup data forecast/L8_Train.csv',
        '/Users/yuheng/Desktop/AI cup data forecast/L9_Train.csv',
        '/Users/yuheng/Desktop/AI cup data forecast/L10_Train.csv'
    ]

    # 讀取所有檔案並合併
    historical_data = load_csv_data(file_paths)
    print(historical_data.head())
