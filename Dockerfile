FROM python
WORKDIR /Train_linear
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "Train_linear.py"]