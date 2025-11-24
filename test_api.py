#!/usr/bin/env python3
"""
ML Inference API テストスクリプト

使用方法:
    python test_api.py
"""

import requests
import json
import time
import sys

BASE_URL = "http://localhost:8080"

def print_section(title):
    """セクションタイトルを表示"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def test_gateway_health():
    """Gatewayのヘルスチェック"""
    print_section("Gateway Health Check")
    try:
        response = requests.get(f"{BASE_URL}/gateway/health", timeout=5)
        print(f"Status Code: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_gateway_info():
    """Gateway情報の取得"""
    print_section("Gateway Info")
    try:
        response = requests.get(f"{BASE_URL}/gateway/info", timeout=5)
        print(f"Status Code: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_worker_health():
    """Workerのヘルスチェック"""
    print_section("Worker Health Check")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Status Code: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_single_prediction():
    """単一推論のテスト"""
    print_section("Single Prediction Test")
    try:
        data = {
            "input_data": [1.0, 2.0, 3.0, 4.0],
            "return_probabilities": False
        }
        print(f"Request: {json.dumps(data, indent=2)}")
        
        response = requests.post(
            f"{BASE_URL}/predict",
            json=data,
            timeout=10
        )
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_batch_prediction():
    """バッチ推論のテスト"""
    print_section("Batch Prediction Test")
    try:
        data = [
            {"input_data": [1.0, 2.0, 3.0, 4.0]},
            {"input_data": [2.0, 3.0, 4.0, 5.0]},
            {"input_data": [3.0, 4.0, 5.0, 6.0]}
        ]
        print(f"Request (3 samples):")
        
        response = requests.post(
            f"{BASE_URL}/batch_predict",
            json=data,
            timeout=10
        )
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_load_balancing():
    """ロードバランシングのテスト"""
    print_section("Load Balancing Test (10 requests)")
    try:
        worker_counts = {}
        data = {"input_data": [1.0, 2.0, 3.0, 4.0]}
        
        for i in range(10):
            response = requests.post(f"{BASE_URL}/predict", json=data, timeout=5)
            if response.status_code == 200:
                worker_id = response.json().get("worker_id", "unknown")
                worker_counts[worker_id] = worker_counts.get(worker_id, 0) + 1
            time.sleep(0.1)
        
        print("\nWorker distribution:")
        for worker_id, count in sorted(worker_counts.items()):
            print(f"  Worker {worker_id}: {count} requests")
        
        # 均等に分散されているか確認
        if len(worker_counts) > 1:
            print("\n✅ Load balancing is working!")
            return True
        else:
            print("\n⚠️  Only one worker received requests")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_performance():
    """パフォーマンステスト"""
    print_section("Performance Test (100 requests)")
    try:
        data = {"input_data": [1.0, 2.0, 3.0, 4.0]}
        times = []
        
        print("Running 100 predictions...")
        for i in range(100):
            start = time.time()
            response = requests.post(f"{BASE_URL}/predict", json=data, timeout=5)
            if response.status_code == 200:
                times.append(time.time() - start)
            
            if (i + 1) % 20 == 0:
                print(f"  Completed: {i + 1}/100")
        
        if times:
            avg_time = sum(times) / len(times) * 1000  # ms
            min_time = min(times) * 1000
            max_time = max(times) * 1000
            
            print(f"\nResults:")
            print(f"  Average latency: {avg_time:.2f} ms")
            print(f"  Min latency: {min_time:.2f} ms")
            print(f"  Max latency: {max_time:.2f} ms")
            print(f"  Throughput: {len(times) / sum(times):.2f} req/s")
            return True
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """メイン関数"""
    print("\n🚀 ML Inference API Test Suite")
    print(f"Testing against: {BASE_URL}")
    
    tests = [
        ("Gateway Health", test_gateway_health),
        ("Gateway Info", test_gateway_info),
        ("Worker Health", test_worker_health),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Load Balancing", test_load_balancing),
        ("Performance", test_performance),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except KeyboardInterrupt:
            print("\n\n⚠️  Tests interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Unexpected error in {name}: {e}")
            results.append((name, False))
    
    # サマリー
    print_section("Test Summary")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
