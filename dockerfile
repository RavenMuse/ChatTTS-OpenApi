FROM image.cloudlayer.icu/python:3.12-slim

# RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN pip install uv -i https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /app

COPY . .

RUN --mount=type=cache,target=/root/.cache/uv,id=uv-repo \
    uv sync --extra cpu

EXPOSE 9000

CMD ["uv","run","--extra","cpu","api.py"]
