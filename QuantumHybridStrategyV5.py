"""
QuantumHybrid Hybrid Trading Strategy Statement

——Released by: Huanran Shao

Dear Quant Trading Enthusiasts:

Thank you for your interest in the "QuantumHybrid V5" multi-frequency hybrid strategy developed by me. This strategy is based on a machine learning model (integrating wavelet transform features and GBDT classifiers) to simultaneously analyze dual time frames of 15 minutes and 4 hours, combining multiple factor signals such as Bollinger Bands breakout and EMA dynamic stop-loss to construct a trading decision system.

【Risk Warning】

Extreme Market Volatility Risk: Although the strategy has a hard stop-loss set at -22%, black swan events in the crypto market may still lead to unexpected drawdowns.
Model Failure Alert: Feature engineering relies on historical market structures, and manual intervention for calibration is required when new trading patterns emerge.
Leverage Caution: The strategy is configured with no leverage by default, and liquidity risks must be borne by the user if leverage is applied.
Technical Risk Statement: Real-time signals depend on the stability of the data pipeline, and feature calculation delays may occur under extreme circumstances.
"""

# hybrid_strategy_v5.py
from freqtrade.strategy import IStrategy
from pandas import DataFrame, Series
import talib.abstract as ta
import joblib
import numpy as np
import pywt
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class QuantumHybridStrategyV5(IStrategy):
    timeframe = '15m'
    minimal_roi = {"0": 0.18, "48": 0.1, "144": 0.05}
    stoploss = -0.22
    use_custom_stoploss = True
    process_only_new_candles = True
    
    # 模型配置
    MODEL_PATH = 'multi_scale_model_v5.pkl'
    
    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.model = None
        self.feature_engineer = None
        self.mt_generator = None
        self._load_artifacts()
        logger.info("策略初始化完成")
        
    def _load_artifacts(self):
        """带异常处理的模型加载"""
        try:
            artifacts = joblib.load(self.MODEL_PATH)
            self.model = artifacts['model']
            self.feature_engineer = artifacts['feature_engineer'] 
            self.mt_generator = artifacts['timeframe_generator']
            logger.info("模型加载成功")
        except Exception as e:
            logger.error(f"模型加载失败：{str(e)}")
            raise
        
    def _get_4h_data(self) -> DataFrame:
        """获取实时4小时合成数据"""
        df_15m = self.dp.get_pair_dataframe("BTC/USDT", self.timeframe)
        return self.mt_generator.resample_4h(df_15m)
        
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 实时生成4小时数据
        df_4h = self._get_4h_data()
        
        # 合并特征（防止未来数据）
        merged = self.feature_engineer.transform(dataframe.iloc[-1000:], df_4h)
        features = merged[self.feature_engineer.feature_columns].iloc[-1:]
        
        # 模型预测
        dataframe.loc[dataframe.index[-1], 'ml_proba'] = self.model.predict_proba(features)[0][1]
        return dataframe
    
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        current_row = dataframe.iloc[-1]
        reasons = []
        
        # 信号条件判断
        if current_row['ml_proba'] > 0.72:
            reasons.append("模型置信度高")
        if current_row['close'] > current_row['bb_upper']:
            reasons.append("突破上轨")
        if current_row['volume'] > current_row['volume'].rolling(4).mean() * 1.6:
            reasons.append("放量")
            
        if reasons:
            dataframe.loc[dataframe.index[-1], 'buy'] = 1
            logger.info(f"买入信号触发：{', '.join(reasons)}")
            dataframe.loc[dataframe.index[-1], 'buy_reason'] = '|'.join(reasons)
            
        return dataframe
    
    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        current_row = dataframe.iloc[-1]
        reasons = []
        
        if current_row['ml_proba'] < 0.35:
            reasons.append("模型信心不足")
        if current_row['close'] < current_row['ema25'] * 0.965:
            reasons.append("跌破EMA25")
            
        if reasons:
            dataframe.loc[dataframe.index[-1], 'sell'] = 1 
            logger.info(f"卖出信号触发：{', '.join(reasons)}")
            dataframe.loc[dataframe.index[-1], 'sell_reason'] = '|'.join(reasons)
            
        return dataframe

    def version(self) -> str:
        return "5.1.0"
