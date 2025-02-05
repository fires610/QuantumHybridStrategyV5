#For Super Earth!!!
# train_model_v5.py
import json
import pandas as pd
import numpy as np
import pywt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer
from lightgbm import LGBMClassifier
import joblib
import talib.abstract as ta
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CCXTDataParser:
    """解析CCXT的JSON数据结构-带进度显示"""
    def __init__(self):
        self.ohlcv_keys = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
    def parse_json(self, file_path):
        with open(file_path, 'r') as f:
            raw_data = json.load(f)
        
        # 带进度条的转换
        parsed_data = []
        for candle in tqdm(raw_data, desc='解析CCXT数据'):
            parsed_data.append([
                candle[0],   # timestamp
                float(candle[1]), 
                float(candle[2]),
                float(candle[3]),
                float(candle[4]),
                float(candle[5]),
            ])
            
        df = pd.DataFrame(parsed_data, columns=self.ohlcv_keys)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df.set_index('timestamp').sort_index()

class MultiTimeframeGenerator:
    """改进的4小时数据合成"""
    def resample_4h(self, df_15m):
        # 确保时间对齐（交易所时间戳处理）
        resampled = df_15m.resample('4H', origin='start', offset='15min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # 验证示例
        logger.info(f"合成数据样例：\n{resampled.head(3)}\n{resampled.tail(3)}")
        return resampled

class EnhancedFeatureEngineer:
    """带特征验证的增强特征工程"""
    def __init__(self):
        self.wavelet = 'db5'
        self.n_bins = 4
        
    def _validate_features(self, merged_df):
        # 检查未来数据泄露
        assert merged_df.isna().sum().sum() == 0, "存在空值特征"
        logger.info(f"特征矩阵形状：{merged_df.shape}")
        
    def transform(self, df_15m, df_4h):
        # 合并数据（前向填充处理）
        merged = pd.merge_asof(
            df_15m, df_4h.add_suffix('_4h'),
            left_index=True, right_index=True,
            direction='forward', tolerance=pd.Timedelta('4h')
        )
        
        # 小波特征（改进边距处理）
        coeffs = pywt.wavedec(merged['close'].values, self.wavelet, level=3)
        merged['wavelet_l3'] = np.concatenate([coeffs[0], np.zeros(len(merged)-len(coeffs[0]))])
        
        # 跨周期动量指标
        merged['delta_4h'] = merged['close_4h'].pct_change(4)
        merged['vol_ratio'] = merged['volume'] / merged['volume_4h'].rolling(6, min_periods=1).mean()
        
        self._validate_features(merged)
        return merged
#Have the taste of 德莫克拉西😈
def main():
    logger.info("=== 开始训练流程 ===")
    
    # 数据加载（带进度）
    parser = CCXTDataParser()
    df_15m = parser.parse_json("BTC_USDT_15m.json")
    logger.info(f"原始数据时间范围：{df_15m.index.min()} ~ {df_15m.index.max()}")
    
    # 4小时数据生成（改进版本）
    mt_gen = MultiTimeframeGenerator()
    df_4h = mt_gen.resample_4h(df_15m)
    
    # 特征工程
    engineer = EnhancedFeatureEngineer()
    with tqdm(total=4, desc="特征工程进度") as pbar:
        feature_df = engineer.transform(df_15m, df_4h)
        pbar.update(1)
        
        # 技术指标计算
        feature_df['adx'] = ta.ADX(feature_df, 14)
        pbar.update(1)
        feature_df['rsi'] = ta.RSI(feature_df, 14)
        pbar.update(1)
        feature_df['ema50'] = ta.EMA(feature_df, 50)
        pbar.update(1)
    
    # 创建标签
    y = (feature_df['close'].shift(-48) > feature_df['close'] * 1.018).astype(int)
    X = feature_df.drop(columns=['close', 'close_4h'])
    
    # 模型训练（带进度回调）
    model = LGBMClassifier(n_estimators=800, verbose=-1)
    model.fit(X, y, 
        eval_set=[(X, y)],
        callbacks=[tqdm(desc="模型训练", total=800)]
    )
    
    # 保存完整pipeline
    joblib.dump({
        'model': model,
        'feature_engineer': engineer,
        'timeframe_generator': mt_gen
    }, 'multi_scale_model_v5.pkl')
    logger.info("=== 训练完成 ===")

if __name__ == "__main__":
    main()
