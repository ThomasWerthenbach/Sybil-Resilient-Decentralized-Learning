from ml.executor.executor import Executor


class AverageExecutor(Executor):

    def execute(self, data):
        self._sum += data
        self._count += 1
        return self._sum / self._count