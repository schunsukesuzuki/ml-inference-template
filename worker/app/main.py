from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import logging
from typing import List, Optional
import time

# ここに自分のモデルをインポート
from .model import ModelInference

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Inference Worker",
    description="Machine Learning Inference Worker API",
    version="1.0.0"
)

# グローバル変数でモデルを保持
model_inference = None
worker_id = os.getenv("WORKER_ID", "unknown")
gpu_id = os.getenv("CUDA_VISIBLE_DEVICES", "cpu")

# リクエスト/レスポンスモデル
class PredictionRequest(BaseModel):
    """推論リクエスト"""
    input_data: List[float]
    return_probabilities: Optional[bool] = False
    
    class Config:
        json_schema_extra = {
            "example": {
                "input_data": [1.0, 2.0, 3.0, 4.0],
                "return_probabilities": False
            }
        }

class PredictionResponse(BaseModel):
    """推論レスポンス"""
    prediction: List[float]
    worker_id: str
    inference_time_ms: float
    probabilities: Optional[List[float]] = None

class HealthResponse(BaseModel):
    """ヘルスチェックレスポンス"""
    status: str
    worker_id: str
    gpu_id: str
    model_loaded: bool

@app.on_event("startup")
async def startup_event():
    """起動時にモデルをロード"""
    global model_inference
    
    logger.info(f"Starting Worker {worker_id} on GPU/Device: {gpu_id}")
    
    try:
        # モデルの初期化
        model_inference = ModelInference()
        logger.info(f"Model loaded successfully on Worker {worker_id}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """シャットダウン時のクリーンアップ"""
    logger.info(f"Shutting down Worker {worker_id}")

@app.get("/", response_model=dict)
async def root():
    """ルートエンドポイント"""
    return {
        "message": "ML Inference Worker API",
        "worker_id": worker_id,
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """ヘルスチェックエンドポイント"""
    return HealthResponse(
        status="healthy" if model_inference is not None else "unhealthy",
        worker_id=worker_id,
        gpu_id=gpu_id,
        model_loaded=model_inference is not None
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    推論エンドポイント
    
    Args:
        request: 推論リクエスト
        
    Returns:
        推論結果
    """
    if model_inference is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    try:
        start_time = time.time()
        
        # 推論実行
        prediction = model_inference.predict(
            request.input_data,
            return_probabilities=request.return_probabilities
        )
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        logger.info(
            f"Prediction completed on Worker {worker_id} "
            f"in {inference_time:.2f}ms"
        )
        
        response = PredictionResponse(
            prediction=prediction["prediction"],
            worker_id=worker_id,
            inference_time_ms=inference_time
        )
        
        if request.return_probabilities and "probabilities" in prediction:
            response.probabilities = prediction["probabilities"]
            
        return response
        
    except Exception as e:
        logger.error(f"Prediction error on Worker {worker_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/batch_predict", response_model=List[PredictionResponse])
async def batch_predict(requests: List[PredictionRequest]):
    """
    バッチ推論エンドポイント
    
    Args:
        requests: 複数の推論リクエスト
        
    Returns:
        複数の推論結果
    """
    if model_inference is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    try:
        start_time = time.time()
        
        # バッチ推論実行
        batch_input = [req.input_data for req in requests]
        predictions = model_inference.batch_predict(batch_input)
        
        total_time = (time.time() - start_time) * 1000
        avg_time = total_time / len(requests)
        
        logger.info(
            f"Batch prediction ({len(requests)} samples) completed "
            f"on Worker {worker_id} in {total_time:.2f}ms "
            f"(avg: {avg_time:.2f}ms per sample)"
        )
        
        responses = [
            PredictionResponse(
                prediction=pred,
                worker_id=worker_id,
                inference_time_ms=avg_time
            )
            for pred in predictions
        ]
        
        return responses
        
    except Exception as e:
        logger.error(f"Batch prediction error on Worker {worker_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.get("/metrics")
async def metrics():
    """
    メトリクスエンドポイント（Prometheus形式）
    """
    # ここに必要なメトリクスを追加
    return {
        "worker_id": worker_id,
        "gpu_id": gpu_id,
        "model_loaded": model_inference is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
