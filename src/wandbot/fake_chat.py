class FakeChat:
    def __init__(
        self,
    ):
        pass

    def __call__(self, query):
        response = f"Hello! This is a test. Thanks for the query: {query}"
        return response + "\n\n"
