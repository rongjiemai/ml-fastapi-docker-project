# scikit-learn + FastAPI + Docker 示例

这是一个端到端的 Iris 分类项目：

1. 使用 scikit-learn 内置 Iris 数据集。
2. 用 `ColumnTransformer` + `Pipeline` 完成缺失值填充、标准化和逻辑回归训练。
3. 用 FastAPI 暴露健康检查、模型信息和预测接口。
4. 用 Docker 打包成可运行服务。

## 本地运行

```bash
cd ml-fastapi-docker
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.train
python -m uvicorn src.app:app --reload
```

打开 <http://127.0.0.1:8000/docs> 可以使用 Swagger UI。

## 预测示例

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

## Docker 运行

```bash
docker build -t iris-api .
docker run --rm -p 8000:8000 iris-api
```

容器启动时会先执行训练脚本，再启动 API。生产环境中通常会把训练和服务部署拆成两个独立阶段，模型文件通过对象存储或模型仓库传给服务。

## API

- `GET /health`：检查服务和模型是否就绪
- `GET /model-info`：查看模型及特征信息
- `POST /predict`：提交一条样本并获得预测结果

