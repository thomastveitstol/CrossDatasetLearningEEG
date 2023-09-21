from torch.utils.data import Dataset


class SelfSupervisedDataGenerator(Dataset):  # type: ignore[type-arg]
    """
    (In the very early stage of development)

    Data generator for self supervised pretext tasks, where the targets are generated on the fly
    """

    def __init__(self, x, pretext_task):
        super().__init__()

        self._x = x
        self._pretext_task = pretext_task

    def __len__(self):
        return self._x.size()[0]

    def __getitem__(self, item):
        transformed, details = self._pretext_task.transform(self._x[item])
        return transformed, details
