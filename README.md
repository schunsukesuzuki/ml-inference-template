# ML Inference Template with Go Gateway + Python Workers

機械学習モデルを本番環境にデプロイするための実用的なテンプレートです。Go製の高速ゲートウェイと複数のPython Workerで構成され、スケーラブルで耐障害性のある推論システムを構築できます。

## 🎯 特徴

- **Go Gateway**: 高速・低レイテンシのAPIゲートウェイ
  - ロードバランシング（ラウンドロビン）
  - 自動ヘルスチェック
  - リクエストログ
  - グレースフルシャットダウン

- **Python Workers**: JAX/NumPyro対応の推論ワーカー
  - 完全に独立したプロセス（GIL問題なし）
  - GPU分離（各Workerに専用GPU）
  - FastAPI自動ドキュメント
  - バッチ推論対応

- **Docker対応**: 簡単デプロイ
  - マルチステージビルドで最適化
  - CPU版とGPU版の両対応
  - docker-composeで一発起動

## 📁 プロジェクト構造

```
ml-inference-template/
├── gateway/              # Goゲートウェイ
│   ├── main.go          # ロードバランサー実装
│   ├── go.mod           # Go modules
│   └── Dockerfile       # Gateway用Dockerfile
├── worker/              # Python Worker
│   ├── app/
│   │   ├── main.py     # FastAPIアプリケーション
│   │   └── model.py    # モデル推論ロジック（カスタマイズ可能）
│   ├── requirements.txt
│   └── Dockerfile
├── docker-compose.yml       # CPU版
├── docker-compose.gpu.yml   # GPU版
├── Makefile                 # 便利コマンド集
└── README.md
```

## 🚀 クイックスタート

### 必要要件

- Docker & Docker Compose
- （GPU版の場合）NVIDIA Docker Runtime

### 1. セットアップ

```bash
# リポジトリのクローン
git clone <your-repo-url>
cd ml-inference-template

# イメージのビルド
make build

# サービスの起動（CPU版）
make up

# または GPU版
make build-gpu
make up-gpu
```

### 2. 動作確認

```bash
# 自動テスト
make test

# または手動で確認
# ゲートウェイのヘルスチェック
curl http://localhost:8080/gateway/health

# Workerのヘルスチェック
curl http://localhost:8080/health

# 推論テスト
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"input_data": [1.0, 2.0, 3.0, 4.0]}'
```

### 3. API ドキュメント

ブラウザで以下にアクセス:
- Swagger UI: http://localhost:8080/docs
- ReDoc: http://localhost:8080/redoc

## 📝 モデルのカスタマイズ

### あなたのモデルを統合する

`worker/app/model.py` を編集して、自分のモデルを実装してください。

#### JAXモデルの例

```python
class ModelInference:
    def __init__(self):
        # モデルのロード
        with open('path/to/model.pkl', 'rb') as f:
            self.params = pickle.load(f)
        
        # JITコンパイル
        self.predict_fn = jax.jit(self._predict_fn)
    
    def _predict_fn(self, params, x):
        # あなたのモデルロジック
        return jax.nn.relu(jnp.dot(x, params['W']) + params['b'])
    
    def predict(self, input_data, return_probabilities=False):
        x = jnp.array([input_data])
        prediction = self.predict_fn(self.params, x)
        return {'prediction': prediction.flatten().tolist()}
```

#### NumPyroベイジアンモデルの例

```python
from numpyro.infer import Predictive
import pickle

class ModelInference:
    def __init__(self):
        # MCMCサンプルのロード
        with open('mcmc_samples.pkl', 'rb') as f:
            self.mcmc_samples = pickle.load(f)
        
        # Predictiveオブジェクトの作成
        self.predictive = Predictive(self.model_fn, self.mcmc_samples)
    
    def predict(self, input_data, return_probabilities=False):
        rng_key = jax.random.PRNGKey(0)
        predictions = self.predictive(rng_key, obs=jnp.array(input_data))
        
        return {
            'prediction': jnp.mean(predictions['y'], axis=0).tolist(),
            'std': jnp.std(predictions['y'], axis=0).tolist()
        }
```

### モデルファイルの配置

モデルファイルがある場合:

1. `worker/models/` ディレクトリを作成
2. モデルファイルを配置
3. `worker/Dockerfile` を編集:
```dockerfile
# この行を追加
COPY models/ ./models/
```

## 🔧 設定

### Workerの数を変更

`docker-compose.yml` を編集:

```yaml
services:
  gateway:
    environment:
      - WORKERS=worker1:8000,worker2:8000,worker3:8000,worker4:8000  # worker4を追加

  worker4:  # 新しいWorkerを追加
    build:
      context: ./worker
    environment:
      - WORKER_ID=4
      - CUDA_VISIBLE_DEVICES=3
```

### GPU設定

`docker-compose.gpu.yml` でGPU IDを変更:

```yaml
worker1:
  environment:
    - CUDA_VISIBLE_DEVICES=0  # GPU 0を使用
  deploy:
    resources:
      reservations:
        devices:
          - device_ids: ['0']  # ここも変更
```

## 📊 エンドポイント

### Gateway エンドポイント

| エンドポイント | メソッド | 説明 |
|--------------|---------|------|
| `/gateway/health` | GET | ゲートウェイのヘルスチェック |
| `/gateway/info` | GET | ゲートウェイ情報 |

### Worker エンドポイント（ゲートウェイ経由）

| エンドポイント | メソッド | 説明 |
|--------------|---------|------|
| `/` | GET | API情報 |
| `/health` | GET | Workerヘルスチェック |
| `/predict` | POST | 単一サンプル推論 |
| `/batch_predict` | POST | バッチ推論 |
| `/docs` | GET | Swagger UI |
| `/redoc` | GET | ReDoc |

### リクエスト例

#### 単一推論

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "input_data": [1.0, 2.0, 3.0, 4.0],
    "return_probabilities": false
  }'
```

レスポンス:
```json
{
  "prediction": [0.5],
  "worker_id": "2",
  "inference_time_ms": 1.23
}
```

#### バッチ推論

```bash
curl -X POST http://localhost:8080/batch_predict \
  -H "Content-Type: application/json" \
  -d '[
    {"input_data": [1.0, 2.0, 3.0, 4.0]},
    {"input_data": [2.0, 3.0, 4.0, 5.0]}
  ]'
```

## 🛠️ Makeコマンド

```bash
make help              # ヘルプ表示
make build             # イメージビルド（CPU版）
make build-gpu         # イメージビルド（GPU版）
make up                # サービス起動（CPU版）
make up-gpu            # サービス起動（GPU版）
make down              # サービス停止
make logs              # 全ログ表示
make logs-gateway      # Gatewayログのみ
make logs-workers      # Workerログのみ
make test              # APIテスト
make restart           # 再起動
make clean             # 完全クリーンアップ
make info              # システム情報表示
```

## 🐛 トラブルシューティング

### Workerが起動しない

```bash
# ログを確認
make logs-workers

# 個別のWorkerログを確認
docker logs ml-worker-1
```

### Gatewayがバックエンドを見つけられない

```bash
# ネットワークを確認
docker network inspect ml-inference-template_ml-network

# Workerが起動しているか確認
docker-compose ps
```

### GPU が認識されない

```bash
# NVIDIA Dockerがインストールされているか確認
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# docker-compose.gpu.yml を使用しているか確認
make up-gpu
```

## 📈 スケーリング

### 水平スケーリング

Workerを追加するだけ:

```bash
# docker-compose.yml にworker4, worker5... を追加
# Gatewayの WORKERS 環境変数も更新

make restart
```

### 垂直スケーリング

リソースを増やす:

```yaml
worker1:
  deploy:
    resources:
      limits:
        cpus: '2'
        memory: 4G
```

## 🔒 本番環境への展開

### セキュリティ

1. **認証の追加**: Gatewayに認証ミドルウェアを実装
2. **HTTPS**: NginxやTraefikをフロントに配置
3. **レート制限**: Gatewayにレート制限を実装

### モニタリング

- Prometheusメトリクス: `/metrics` エンドポイントを実装
- ログ集約: ELKスタックやDatadogと統合
- アラート: 異常検知とアラート設定

### Kubernetes展開

Helmチャートを作成して展開可能です（別途提供可能）。

## 🤝 貢献

バグ報告、機能リクエスト、プルリクエストを歓迎します！

## 📄 ライセンス

MIT License

## 🙏 謝辞

このテンプレートは以下の技術を使用しています:
- [Go](https://golang.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [JAX](https://github.com/google/jax)
- [NumPyro](https://num.pyro.ai/)
- [Docker](https://www.docker.com/)

---

**質問やサポートが必要ですか？**
- Issue を開く
- プルリクエストを送る
- ドキュメントを確認する
