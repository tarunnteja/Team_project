FROM python
WORKDIR /test_random_forest
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "train_random_forest.py"]