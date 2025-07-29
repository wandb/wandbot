# wandbot起動成功の手順

## 起動方法

### 前提条件
- Python 3.12以上
- uv がインストール済み
- `.env` ファイルが設定済み（API keys等）

### 起動コマンド
```bash
# 1. 環境変数を設定
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# 2. wandbotを起動
bash run.sh
```

### 動作確認
```bash
# 起動確認
curl http://localhost:8000/startup

# テストクエリ
curl -X POST http://localhost:8000/chat/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "How do I log a W&B artifact?"}'
```

### 停止方法
```bash
# プロセスを停止
pkill -f "uvicorn|wandbot|slack"
```

---

## 成功したこと

### タスク1: データベースとベクトルストアの設定 ✅

1. **データベース設定**
   - SQLiteデータベース（`./data/cache/app.db`）が既に存在し、正常に設定済み
   - 設定ファイル: `src/wandbot/configs/database_config.py`

2. **ベクトルストア設定**
   - `wandbot/wandbot-dev/chroma_index:v54` アーティファクトをダウンロード
   - `artifacts/vector_stores_512/` ディレクトリに配置
   - 実際のコレクション名 `vectorstore-chroma_index-v52-run_xwyhhej9` に設定を修正
   - 12,410個のドキュメントが含まれるベクトルストアを正常に設定

3. **設定修正内容**
   - `src/wandbot/configs/vector_store_config.py` で以下を修正:
     - `vector_store_mode` を `"local"` に変更
     - `vectordb_collection_name` を実際のコレクション名に修正
     - `vectordb_index_dir` を `"artifacts/vector_stores_512"` に設定
   - `.env` ファイルに必要な環境変数を追加:
     - `VECTORDB_INDEX_DIR=artifacts/vector_stores_512`
     - `VECTORDB_PERSIST_DIR=artifacts/vector_stores_512`

### タスク2: wandbotの実行 ✅

1. **起動方法**
   - `run.sh` を使用してwandbotを起動
   - APIサーバー（uvicorn）とSlack botが正常に起動

2. **動作確認**
   - APIエンドポイント: `http://localhost:8000/chat/query` が正常に応答
   - ベクトルストア検索: 200件のドキュメントを取得し、150件を検索、重複除去（79件）、15件にリランク
   - 回答生成: W&Bに関する質問に正確で詳細な回答を生成
   - ソース引用: 適切なソースが引用される

3. **利用可能な機能**
   - APIサーバー: `http://localhost:8000` で動作
   - Slack bot: 日本語対応（`-l ja`）で動作
   - ベクトルストア: 12,410個のドキュメントから検索可能

## 最終結果

wandbotは完全に動作しており、以下の機能が利用可能:
- ✅ ベクトルストア検索機能
- ✅ 回答生成機能  
- ✅ APIエンドポイント
- ✅ Slack bot連携
- ✅ 日本語対応
