# SPLADE with Elasticsearch


Running SPLADE with Elasticsearch.

With Elasticsearch 8.15.0, `sparse_vector` query is added ([release note](https://www.elastic.co/guide/en/elasticsearch/reference/8.15/release-notes-8.15.0.html)).

In this Repository, running SPLADE using `sparse_vector` on BEIR benchmark.

## Running

First, startup Elasticsearch using docker:

```shell
docker compose up -d
```

Install dependencies:

```
uv sync
```

Running evaluation of SPLADE on a dataset:
```
./run EvaluateTask
```

You can configure which model and dataset is used in `conf/param.ini`.

If you want to use cuda for SPLADE inference, don't forget to set `device=cuda:0` to `SpladeEncodeTask` like following:
```ini
[SpladeEncodeTask]
corpus_load_batch_size=10000
device=cuda:0
```
