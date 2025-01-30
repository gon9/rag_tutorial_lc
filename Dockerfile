# ビルドステージ
FROM python:3.12-slim-bookworm as builder

WORKDIR /app

RUN pip install --no-cache-dir poetry
COPY pyproject.toml poetry.lock ./
# Poetry設定
RUN poetry config virtualenvs.create false
# 本番環境の依存パッケージのみをインストール
RUN poetry install --only main --no-root --no-interaction --no-ansi
# ソースコードをコピーしてプロジェクトをインストール
COPY . .
RUN poetry install --only main --no-interaction --no-ansi


# 実行ステージ
FROM python:3.12-slim-bookworm as runner

WORKDIR /app

# Poetryのインストール
RUN pip install --no-cache-dir poetry
COPY pyproject.toml poetry.lock ./
# Poetry設定
RUN poetry config virtualenvs.create false

# builderステージからインストール済みのパッケージをコピー
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
# アプリケーションコードのコピー
COPY . .

# 環境変数の設定
ENV PYTHONUNBUFFERED=1

# Poetry経由でアプリケーションを実行
CMD ["poetry", "run", "python", "src/app.py"]