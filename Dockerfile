FROM python
WORKDIR /finalproject_v1_2_labelencoding
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "finalproject_v1_2_labelencoding.py"]