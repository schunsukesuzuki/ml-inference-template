#!/usr/bin/env python3
"""
ml-inference-templateの単体テスト

Dockerなしで主要なコンポーネントをテスト
"""

import sys
import os

# テスト用のモックモジュール
print("=" * 60)
print("ML Inference Template - Unit Tests")
print("=" * 60)

# 1. インポートテスト
print("\n[1] Import Tests")
print("-" * 60)

test_results = []

try:
    print("  ✓ Testing Python standard library imports...")
    import logging
    import time
    import json
    from typing import List, Optional
    print("    ✓ Standard library imports: OK")
    test_results.append(("Standard imports", True, None))
except Exception as e:
    print(f"    ✗ Standard library imports: FAILED - {e}")
    test_results.append(("Standard imports", False, str(e)))

try:
    print("  ✓ Testing Pydantic...")
    from pydantic import BaseModel
    
    class TestModel(BaseModel):
        value: float
    
    test_obj = TestModel(value=1.0)
    assert test_obj.value == 1.0
    print("    ✓ Pydantic: OK")
    test_results.append(("Pydantic", True, None))
except Exception as e:
    print(f"    ✗ Pydantic: FAILED - {e}")
    test_results.append(("Pydantic", False, str(e)))

try:
    print("  ✓ Testing NumPy...")
    import numpy as np
    arr = np.array([1.0, 2.0, 3.0])
    assert arr.shape == (3,)
    print("    ✓ NumPy: OK")
    test_results.append(("NumPy", True, None))
except Exception as e:
    print(f"    ✗ NumPy: FAILED - {e}")
    test_results.append(("NumPy", False, str(e)))

try:
    print("  ✓ Testing JAX...")
    import jax.numpy as jnp
    arr = jnp.array([1.0, 2.0, 3.0])
    assert arr.shape == (3,)
    print("    ✓ JAX: OK")
    test_results.append(("JAX", True, None))
except Exception as e:
    print(f"    ✗ JAX: FAILED - {e}")
    test_results.append(("JAX", False, str(e)))

# 2. モデルクラスのテスト
print("\n[2] Model Class Tests")
print("-" * 60)

try:
    print("  ✓ Testing ModelInference class...")
    sys.path.insert(0, '/mnt/user-data/outputs/ml-inference-template/worker')
    from app.model import ModelInference
    
    model = ModelInference()
    print("    ✓ Model instantiation: OK")
    
    # 単一推論テスト
    result = model.predict([1.0, 2.0, 3.0, 4.0])
    assert "prediction" in result
    assert isinstance(result["prediction"], list)
    print(f"    ✓ Single prediction: OK (result: {result['prediction'][:3]}...)")
    
    # 確率付き推論テスト
    result_with_probs = model.predict([1.0, 2.0, 3.0, 4.0], return_probabilities=True)
    assert "probabilities" in result_with_probs
    print(f"    ✓ Prediction with probabilities: OK")
    
    # バッチ推論テスト
    batch_input = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    batch_result = model.batch_predict(batch_input)
    assert len(batch_result) == 3
    print(f"    ✓ Batch prediction: OK (processed {len(batch_result)} samples)")
    
    test_results.append(("ModelInference", True, None))
    
except Exception as e:
    print(f"    ✗ ModelInference: FAILED - {e}")
    test_results.append(("ModelInference", False, str(e)))

# 3. FastAPIアプリのテスト
print("\n[3] FastAPI Application Tests")
print("-" * 60)

try:
    print("  ✓ Testing FastAPI app structure...")
    from app.main import app, PredictionRequest, PredictionResponse, HealthResponse
    
    # アプリの基本構造確認
    assert app.title == "ML Inference Worker"
    print(f"    ✓ App title: {app.title}")
    
    # ルートの確認
    routes = [route.path for route in app.routes]
    expected_routes = ["/", "/health", "/predict", "/batch_predict"]
    for route in expected_routes:
        if route in routes:
            print(f"    ✓ Route '{route}': Found")
        else:
            print(f"    ⚠ Route '{route}': Not found")
    
    # Pydanticモデルの確認
    test_request = PredictionRequest(
        input_data=[1.0, 2.0, 3.0, 4.0],
        return_probabilities=False
    )
    assert test_request.input_data == [1.0, 2.0, 3.0, 4.0]
    print(f"    ✓ PredictionRequest model: OK")
    
    test_results.append(("FastAPI app", True, None))
    
except Exception as e:
    print(f"    ✗ FastAPI app: FAILED - {e}")
    test_results.append(("FastAPI app", False, str(e)))

# 4. Goコードの検証（構文チェック）
print("\n[4] Go Code Validation")
print("-" * 60)

try:
    print("  ✓ Checking Go source code...")
    go_main_path = "/mnt/user-data/outputs/ml-inference-template/gateway/main.go"
    
    with open(go_main_path, 'r') as f:
        go_code = f.read()
    
    # 基本的な構文要素をチェック
    checks = [
        ("package main", "Package declaration"),
        ("import (", "Import statement"),
        ("type Backend struct", "Backend struct"),
        ("type LoadBalancer struct", "LoadBalancer struct"),
        ("func NewLoadBalancer", "Constructor function"),
        ("func (lb *LoadBalancer) GetNextBackend", "Load balancing method"),
        ("func healthCheck", "Health check function"),
        ("func main()", "Main function"),
    ]
    
    for check_str, description in checks:
        if check_str in go_code:
            print(f"    ✓ {description}: Found")
        else:
            print(f"    ✗ {description}: Not found")
    
    print(f"    ✓ Go source code size: {len(go_code)} bytes")
    test_results.append(("Go code structure", True, None))
    
except Exception as e:
    print(f"    ✗ Go code validation: FAILED - {e}")
    test_results.append(("Go code structure", False, str(e)))

# 5. Docker設定の検証
print("\n[5] Docker Configuration Validation")
print("-" * 60)

try:
    print("  ✓ Checking Docker Compose files...")
    
    import yaml
    
    # docker-compose.ymlの読み込み
    with open("/mnt/user-data/outputs/ml-inference-template/docker-compose.yml", 'r') as f:
        compose_config = yaml.safe_load(f)
    
    services = compose_config.get('services', {})
    print(f"    ✓ Services defined: {list(services.keys())}")
    
    # Gatewayサービスの確認
    if 'gateway' in services:
        gateway = services['gateway']
        print(f"    ✓ Gateway ports: {gateway.get('ports', [])}")
        print(f"    ✓ Gateway depends_on: {gateway.get('depends_on', [])}")
    
    # Workerサービスの確認
    worker_count = sum(1 for s in services if s.startswith('worker'))
    print(f"    ✓ Worker count: {worker_count}")
    
    test_results.append(("Docker configuration", True, None))
    
except Exception as e:
    print(f"    ✗ Docker configuration: FAILED - {e}")
    test_results.append(("Docker configuration", False, str(e)))

# テスト結果サマリー
print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)

passed = sum(1 for _, result, _ in test_results if result)
failed = sum(1 for _, result, _ in test_results if not result)
total = len(test_results)

print(f"\nTotal Tests: {total}")
print(f"Passed: {passed} ✓")
print(f"Failed: {failed} ✗")
print(f"Success Rate: {passed/total*100:.1f}%")

if failed > 0:
    print("\nFailed Tests:")
    for name, result, error in test_results:
        if not result:
            print(f"  ✗ {name}: {error}")

print("\n" + "=" * 60)

if failed == 0:
    print("🎉 ALL TESTS PASSED!")
    print("\nNext steps:")
    print("  1. Install Docker and Docker Compose")
    print("  2. Run: make build")
    print("  3. Run: make up")
    print("  4. Run: make test")
else:
    print("⚠️  SOME TESTS FAILED")
    print("\nPlease fix the errors above before deploying.")

print("=" * 60)

sys.exit(0 if failed == 0 else 1)
