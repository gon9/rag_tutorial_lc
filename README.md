# langchainでのRAGチュートリアル

## 事前準備
### .envファイルの作成

```bash
cp .env.example .env
```

OPENAI_API_KEY, LANGCHAIN_ENDPOINT, LANGCHAIN_API_KEY, LANGCHAIN_PROJECTを設定

### データの準備
以下のサイトを`data`フォルダにダウンロード
https://manual.toyota.jp/pdf/aqua/aqua_202005.pdf

## 実行(ローカル)

```bash
poetry install
poetry run python src/app.py
```

## 実行(docker compose)

```bash
docker compose up -d
```
