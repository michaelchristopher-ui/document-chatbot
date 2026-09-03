"""The capability ports, satisfied by binding a model name to an `LLMProvider`.

`EmbeddingModel`, `OcrModel` and `Reranker` share a package because they share
an implementation: each is a few lines holding a provider and the id of the
model to ask it for. Splitting them into three would be three files of one class
and no reader better off.
"""
