import llm
from llm.plugins import pm
import pytest


class AsyncMockModel(llm.AsyncModel):
    model_id = "async-mock"

    def __init__(self):
        self.history = []
        self._queue = []

    def enqueue(self, messages):
        assert isinstance(messages, list)
        self._queue.append(messages)

    async def execute(self, prompt, stream, response, conversation):
        self.history.append((prompt, stream, response, conversation))
        gathered = []
        while True:
            try:
                messages = self._queue.pop(0)
                for message in messages:
                    gathered.append(message)
                    yield message
                break
            except IndexError:
                break
        response.set_usage(input=len(prompt.prompt.split()), output=len(gathered))


@pytest.fixture(autouse=True)
def register_embed_demo_model():
    class MockModelsPlugin:
        __name__ = "MockModelsPlugin"

        @llm.hookimpl
        def register_models(self, register):
            # Registering async model as the first argument is invalid but
            # works for these tests since we never try to use a non-async model
            register(AsyncMockModel(), async_model=AsyncMockModel())

    pm.register(MockModelsPlugin(), name="undo-mock-models-plugin")
    try:
        yield
    finally:
        pm.unregister(name="undo-mock-models-plugin")
