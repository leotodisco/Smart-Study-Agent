class LazyEmbeddings:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        self._emb = None
        self.model_name = model_name

    def _load(self):
        if self._emb is None:
            from langchain_huggingface import HuggingFaceEmbeddings
            self._emb = HuggingFaceEmbeddings(model_name=self.model_name)

    def embed_query(self, text):
        self._load()
        return self._emb.embed_query(text)

    def embed_documents(self, texts):
        self._load()
        return self._emb.embed_documents(texts)