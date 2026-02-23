FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train the production model during image build (so it exists on Render)
RUN mkdir -p models/prod && \
    python3 scripts/train_prod_model.py --csv data/dataset_40k_lignes.csv --out models/prod/premium_model_prod.joblib

ENV PORT=10000
EXPOSE 10000

CMD ["bash", "-lc", "streamlit run src/app/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true"]