import logging
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torchvision import transforms
from typing import Iterable, Optional, Sized
from utils.data import iCIFAR10, iCIFAR100, iImageNet100, iImageNet1000, iCIFAR224, iImageNetR,iImageNetA,CUB, objectnet, omnibenchmark, vtab


class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, args):
        self.args = args
        self.dataset_name = dataset_name
        self._task_builder = str(self.args.get("task_builder", "class_incremental")).lower()
        self._online_sampler = None
        self._online_task_indices = None
        self._online_test_task_indices = None
        self._setup_data(dataset_name, shuffle, seed)
        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)
        if self._task_builder == "online_sampler":
            self._setup_online_task_builder(seed)
            self._setup_online_test_task_builder()
            
    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    @property
    def nb_classes(self):
        return len(self._class_order)

    def get_dataset(
        self, indices, source, mode, appendent=None, ret_data=False, m_rate=None
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        task_id = self._get_task_id_from_indices(indices)
        if source == "train" and self._online_task_indices is not None and task_id is not None:
            if m_rate not in (None, 0):
                raise ValueError("m_rate is not supported together with task_builder=online_sampler.")
            task_indices = self._online_task_indices[task_id]
            data = x[task_indices]
            targets = y[task_indices]
            if appendent is not None and len(appendent) != 0:
                appendent_data, appendent_targets = appendent
                data = np.concatenate((data, appendent_data))
                targets = np.concatenate((targets, appendent_targets))
            if ret_data:
                return data, targets, DummyDataset(data, targets, trsf, self.use_path)
            return DummyDataset(data, targets, trsf, self.use_path)
        if source == "test" and self._online_test_task_indices is not None and task_id is not None:
            task_indices = self._online_test_task_indices[task_id]
            data = x[task_indices]
            targets = y[task_indices]
            if appendent is not None and len(appendent) != 0:
                appendent_data, appendent_targets = appendent
                data = np.concatenate((data, appendent_data))
                targets = np.concatenate((targets, appendent_targets))
            if ret_data:
                return data, targets, DummyDataset(data, targets, trsf, self.use_path)
            return DummyDataset(data, targets, trsf, self.use_path)

        data, targets = [], []
        for idx in indices:
            if m_rate is None:
                class_data, class_targets = self._select(
                    x, y, low_range=idx, high_range=idx + 1
                )
            else:
                class_data, class_targets = self._select_rmm(
                    x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, trsf, self.use_path)

    def get_dataset_with_split(
        self, indices, source, mode, appendent=None, val_samples_per_class=0
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(
                x, y, low_range=idx, high_range=idx + 1
            )
            val_indx = np.random.choice(
                len(class_data), val_samples_per_class, replace=False
            )
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets)) + 1):
                append_data, append_targets = self._select(
                    appendent_data, appendent_targets, low_range=idx, high_range=idx + 1
                )
                val_indx = np.random.choice(
                    len(append_data), val_samples_per_class, replace=False
                )
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(
            train_targets
        )
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        return DummyDataset(
            train_data, train_targets, trsf, self.use_path
        ), DummyDataset(val_data, val_targets, trsf, self.use_path)

    def _setup_data(self, dataset_name, shuffle, seed):
        idata = _get_idata(dataset_name, self.args)
        idata.download_data()

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        logging.info(self._class_order)

        self._train_targets = _map_new_class_index(
            self._train_targets, self._class_order
        )
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def _setup_online_task_builder(self, seed):
        n = int(self.args.get("online_n", self.args.get("n", 50)))
        m = int(self.args.get("online_m", self.args.get("m", 10)))
        varing_nm = bool(self.args.get("online_varing_NM", self.args.get("varing_NM", False)))

        self._online_sampler = OnlineSampler(
            classes=sorted(np.unique(self._train_targets).tolist()),
            targets=self._train_targets.tolist(),
            num_tasks=self.nb_tasks,
            m=m,
            n=n,
            rnd_seed=self.args.get("online_seed", seed),
            varing_NM=varing_nm,
        )

        self._online_task_indices = [
            np.asarray(task_indices, dtype=np.int64)
            for task_indices in self._online_sampler.indices
        ]

        logging.info(
            "OnlineSampler train task sample counts: %s",
            [len(x) for x in self._online_task_indices],
        )
        logging.info(
            "OnlineSampler disjoint classes: %s",
            self._online_sampler.disjoint_classes,
        )
        logging.info(
            "OnlineSampler blurry classes: %s",
            self._online_sampler.blurry_classes,
        )

    def _setup_online_test_task_builder(self):
        test_source = SamplerDataSource(
            classes=sorted(np.unique(self._test_targets).tolist()),
            targets=self._test_targets.tolist(),
        )

        self._online_test_task_indices = []
        for task_indices in self._online_task_indices:
            exposed_classes = np.unique(self._train_targets[task_indices]).tolist()
            sampler = OnlineTestSampler(test_source, exposed_classes)
            self._online_test_task_indices.append(
                np.asarray(list(sampler), dtype=np.int64)
            )

        logging.info(
            "OnlineSampler test task sample counts: %s",
            [len(x) for x in self._online_test_task_indices],
        )

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))

    def _get_task_id_from_indices(self, indices):
        if self._online_task_indices is None:
            return None

        indices = np.asarray(indices)
        if indices.ndim != 1 or len(indices) == 0:
            return None

        start = 0
        for task_id, task_size in enumerate(self._increments):
            end = start + task_size
            expected = np.arange(start, end)
            if np.array_equal(indices, expected):
                return task_id
            start = end

        return None

class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        return idx, image, label


class SamplerDataSource(object):
    def __init__(self, classes, targets):
        self.classes = list(classes)
        self.targets = list(targets)

    def __len__(self):
        return len(self.targets)


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name, args=None):
    name = dataset_name.lower()
    if name == "cifar10":
        return iCIFAR10()
    elif name == "cifar100":
        return iCIFAR100()
    elif name == "imagenet1000":
        return iImageNet1000()
    elif name == "imagenet100":
        return iImageNet100()
    elif name == "cifar224":
        return iCIFAR224(args)
    elif name == "imagenetr":
        return iImageNetR(args)
    elif name == "imageneta":
        return iImageNetA()
    elif name == "cub":
        return CUB()
    elif name == "objectnet":
        return objectnet()
    elif name == "omnibenchmark":
        return omnibenchmark()
    elif name == "vtab":
        return vtab()

    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
    accimage is available on conda-forge.
    """
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)

class OnlineSampler(Sampler):
    def __init__(
        self,
        data_source: Optional[Sized] = None,
        num_tasks: int = 1,
        m: int = 10,
        n: int = 50,
        rnd_seed: int = 0,
        cur_iter: int = 0,
        varing_NM: bool = False,
        num_replicas: int = None,
        rank: int = None,
        classes=None,
        targets=None,
    ) -> None:

        self.data_source = data_source
        if classes is None or targets is None:
            if data_source is None:
                raise ValueError("OnlineSampler requires either data_source or both classes and targets.")
            classes = data_source.classes
            targets = data_source.targets
        self.classes = list(classes)
        self.targets = list(targets)
        self.generator  = torch.Generator().manual_seed(rnd_seed)
        
        self.n  = n
        self.m  = m
        self.varing_NM = varing_NM
        self.task = cur_iter

        self.num_tasks = num_tasks

        if num_replicas is None and dist.is_available() and dist.is_initialized():
            num_replicas = dist.get_world_size()
        if rank is None and dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()

        self.distributed = num_replicas is not None and rank is not None
        self.num_replicas = num_replicas if num_replicas is not None else 1
        self.rank = rank if rank is not None else 0

        self.disjoint_num   = len(self.classes) * n // 100
        self.disjoint_num   = int(self.disjoint_num // num_tasks) * num_tasks
        self.blurry_num     = len(self.classes) - self.disjoint_num
        self.blurry_num     = int(self.blurry_num // num_tasks) * num_tasks

        if not self.varing_NM:
            class_order         = torch.randperm(len(self.classes), generator=self.generator)
            self.disjoint_classes   = class_order[:self.disjoint_num]
            self.disjoint_classes   = self.disjoint_classes.reshape(num_tasks, -1).tolist()
            self.blurry_classes     = class_order[self.disjoint_num:self.disjoint_num + self.blurry_num]
            self.blurry_classes     = self.blurry_classes.reshape(num_tasks, -1).tolist()

            print("disjoint classes: ", self.disjoint_classes)
            print("blurry classes: ", self.blurry_classes)
            self.disjoint_indices   = [[] for _ in range(num_tasks)]
            self.blurry_indices     = [[] for _ in range(num_tasks)]
            for i in range(len(self.targets)):
                for j in range(num_tasks):
                    if self.targets[i] in self.disjoint_classes[j]:
                        self.disjoint_indices[j].append(i)
                        break
                    elif self.targets[i] in self.blurry_classes[j]:
                        self.blurry_indices[j].append(i)
                        break

            blurred = []
            for i in range(num_tasks):
                blurred += self.blurry_indices[i][:len(self.blurry_indices[i]) * m // 100]
                self.blurry_indices[i] = self.blurry_indices[i][len(self.blurry_indices[i]) * m // 100:]
            blurred = torch.tensor(blurred)
            blurred = blurred[torch.randperm(len(blurred), generator=self.generator)].tolist()
            print("blurry indices: ", len(blurred))
            num_blurred = len(blurred) // num_tasks
            for i in range(num_tasks):
                self.blurry_indices[i] += blurred[:num_blurred]
                blurred = blurred[num_blurred:]

            self.indices = [[] for _ in range(num_tasks)]
            for i in range(num_tasks):
                print("task %d: disjoint %d, blurry %d" % (i, len(self.disjoint_indices[i]), len(self.blurry_indices[i])))
                self.indices[i] = self.disjoint_indices[i] + self.blurry_indices[i]
                self.indices[i] = torch.tensor(self.indices[i])[torch.randperm(len(self.indices[i]), generator=self.generator)].tolist()
        else:

            class_order = torch.randperm(len(self.classes), generator=self.generator)
            self.disjoint_classes = class_order[:self.disjoint_num].tolist()
            if self.disjoint_num > 0:

                self.disjoint_slice = [0] + torch.randint(0, self.disjoint_num, (num_tasks - 1,), generator=self.generator).sort().values.tolist() + [self.disjoint_num]
                self.disjoint_classes = [self.disjoint_classes[self.disjoint_slice[i]:self.disjoint_slice[i + 1]] for i in range(num_tasks)]
            else:
                self.disjoint_classes = [[] for _ in range(num_tasks)]

            if self.blurry_num > 0:

                self.blurry_slice = [0] + torch.randint(0, self.blurry_num, (num_tasks - 1,), generator=self.generator).sort().values.tolist() + [self.blurry_num]
                self.blurry_classes = [class_order[self.disjoint_num + self.blurry_slice[i]:self.disjoint_num + self.blurry_slice[i + 1]].tolist() for i in range(num_tasks)]
            else:
                self.blurry_classes = [[] for _ in range(num_tasks)]

            print("disjoint classes: ", self.disjoint_classes)
            print("blurry classes: ", self.blurry_classes)

            self.disjoint_indices   = [[] for _ in range(num_tasks)]
            self.blurry_indices     = [[] for _ in range(num_tasks)]
            num_blurred = 0
            for i in range(len(self.targets)):
                for j in range(num_tasks):
                    if self.targets[i] in self.disjoint_classes[j]:
                        self.disjoint_indices[j].append(i)
                        break
                    elif self.targets[i] in self.blurry_classes[j]:
                        self.blurry_indices[j].append(i)

                        num_blurred += 1
                        break

            blurred = []
            num_blurred = num_blurred * m // 100
            if num_blurred > 0:

                num_blurred = [0] + torch.randint(0, num_blurred, (num_tasks-1,), generator=self.generator).sort().values.tolist() + [num_blurred]

                for i in range(num_tasks):
                    blurred += self.blurry_indices[i][:num_blurred[i + 1] - num_blurred[i]]
                    self.blurry_indices[i] = self.blurry_indices[i][num_blurred[i + 1] - num_blurred[i]:]
                blurred = torch.tensor(blurred)
                blurred = blurred[torch.randperm(len(blurred), generator=self.generator)].tolist()
                print("blurry indices: ", len(blurred))
                for i in range(num_tasks):
                    self.blurry_indices[i] += blurred[:num_blurred[i + 1] - num_blurred[i]]
                    blurred = blurred[num_blurred[i + 1] - num_blurred[i]:]

            self.indices = [[] for _ in range(num_tasks)]
            for i in range(num_tasks):
                print("task %d: disjoint %d, blurry %d" % (i, len(self.disjoint_indices[i]), len(self.blurry_indices[i])))
                self.indices[i] = self.disjoint_indices[i] + self.blurry_indices[i]
                self.indices[i] = torch.tensor(self.indices[i])[torch.randperm(len(self.indices[i]), generator=self.generator)].tolist()

        if self.distributed:
            self.num_samples = int(len(self.indices[self.task]) // self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas  
            self.num_selected_samples = int(len(self.indices[self.task]) // self.num_replicas)
        else:
            self.num_samples = int(len(self.indices[self.task]))
            self.total_size = self.num_samples
            self.num_selected_samples = int(len(self.indices[self.task]))

    def __iter__(self) -> Iterable[int]:
        return iter(self.get_task(self.task))

    def __len__(self) -> int:
        return self.num_selected_samples

    def set_task(self, cur_iter: int)-> None:
        
        if cur_iter >= self.num_tasks or cur_iter < 0:
            raise ValueError("task out of range")
        self.task = cur_iter

        if self.distributed:
            self.num_samples = int(len(self.indices[self.task]) // self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
            self.num_selected_samples = int(len(self.indices[self.task]) // self.num_replicas)
        else:
            self.num_samples = int(len(self.indices[self.task]))
            self.total_size = self.num_samples
            self.num_selected_samples = int(len(self.indices[self.task]))

    def get_task(self, cur_iter: int)-> Iterable[int]:
        self.set_task(cur_iter)
        if self.distributed:
            indices = self.indices[self.task][self.rank:self.total_size:self.num_replicas]
            assert len(indices) == self.num_samples
            return indices[:self.num_selected_samples]
        return self.indices[self.task]



class OnlineTestSampler(Sampler):
    def __init__(self, data_source: Optional[Sized], exposed_class : Iterable[int], num_replicas: int=None, rank: int=None) -> None:
        self.data_source    = data_source
        self.classes    = self.data_source.classes
        self.targets    = self.data_source.targets
        self.exposed_class  = exposed_class
        self.indices    = [i for i in range(self.data_source.__len__()) if self.targets[i] in self.exposed_class]

        if num_replicas is not None:
            if not dist.is_available():
                raise RuntimeError("Distibuted package is not available, but you are trying to use it.")
            num_replicas = dist.get_world_size()
        if rank is not None:
            if not dist.is_available():
                raise RuntimeError("Distibuted package is not available, but you are trying to use it.")
            rank = dist.get_rank()

        self.distributed = num_replicas is not None and rank is not None
        self.num_replicas = num_replicas if num_replicas is not None else 1
        self.rank = rank if rank is not None else 0

        if self.distributed:
            self.num_samples = int(len(self.indices) // self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
            self.num_selected_samples = int(len(self.indices) // self.num_replicas)
        else:
            self.num_samples = int(len(self.indices))
            self.total_size = self.num_samples
            self.num_selected_samples = int(len(self.indices))

    def __iter__(self) -> Iterable[int]:
        if self.distributed:
            # subsample
            indices = self.indices[self.rank:self.total_size:self.num_replicas]
            assert len(indices) == self.num_samples
            return iter(indices[:self.num_selected_samples])
        else:
            return iter(self.indices)

    def __len__(self) -> int:
        return self.num_selected_samples
        if source == "train" and self._online_task_indices is not None and task_id is not None:
            if m_rate not in (None, 0):
                raise ValueError("m_rate is not supported together with task_builder=online_sampler.")
            task_indices = self._online_task_indices[task_id]
            data = x[task_indices]
            targets = y[task_indices]
            if appendent is not None and len(appendent) != 0:
                appendent_data, appendent_targets = appendent
                data = np.concatenate((data, appendent_data))
                targets = np.concatenate((targets, appendent_targets))
            if ret_data:
                return data, targets, DummyDataset(data, targets, trsf, self.use_path)
            return DummyDataset(data, targets, trsf, self.use_path)

        data, targets = [], []
        for idx in indices:
            if m_rate is None:
                class_data, class_targets = self._select(
                    x, y, low_range=idx, high_range=idx + 1
                )
            else:
                class_data, class_targets = self._select_rmm(
                    x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, trsf, self.use_path)

    def get_dataset_with_split(
        self, indices, source, mode, appendent=None, val_samples_per_class=0
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(
                x, y, low_range=idx, high_range=idx + 1
            )
            val_indx = np.random.choice(
                len(class_data), val_samples_per_class, replace=False
            )
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets)) + 1):
                append_data, append_targets = self._select(
                    appendent_data, appendent_targets, low_range=idx, high_range=idx + 1
                )
                val_indx = np.random.choice(
                    len(append_data), val_samples_per_class, replace=False
                )
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(
            train_targets
        )
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        return DummyDataset(
            train_data, train_targets, trsf, self.use_path
        ), DummyDataset(val_data, val_targets, trsf, self.use_path)

    def _setup_data(self, dataset_name, shuffle, seed):
        idata = _get_idata(dataset_name, self.args)
        idata.download_data()

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        logging.info(self._class_order)

        self._train_targets = _map_new_class_index(
            self._train_targets, self._class_order
        )
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def _setup_online_task_builder(self, seed):
        n = int(self.args.get("online_n", self.args.get("n", 50)))
        m = int(self.args.get("online_m", self.args.get("m", 10)))
        varing_nm = bool(self.args.get("online_varing_NM", self.args.get("varing_NM", False)))

        self._online_sampler = OnlineSampler(
            classes=sorted(np.unique(self._train_targets).tolist()),
            targets=self._train_targets.tolist(),
            num_tasks=self.nb_tasks,
            m=m,
            n=n,
            rnd_seed=self.args.get("online_seed", seed),
            varing_NM=varing_nm,
        )

        self._online_task_indices = [
            np.asarray(task_indices, dtype=np.int64)
            for task_indices in self._online_sampler.indices
        ]

        logging.info(
            "OnlineSampler train task sample counts: %s",
            [len(x) for x in self._online_task_indices],
        )
        logging.info(
            "OnlineSampler disjoint classes: %s",
            self._online_sampler.disjoint_classes,
        )
        logging.info(
            "OnlineSampler blurry classes: %s",
            self._online_sampler.blurry_classes,
        )

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))

    def _get_task_id_from_indices(self, indices):
        if self._online_task_indices is None:
            return None

        indices = np.asarray(indices)
        if indices.ndim != 1 or len(indices) == 0:
            return None

        start = 0
        for task_id, task_size in enumerate(self._increments):
            end = start + task_size
            expected = np.arange(start, end)
            if np.array_equal(indices, expected):
                return task_id
            start = end

        return None

class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        return idx, image, label


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name, args=None):
    name = dataset_name.lower()
    if name == "cifar10":
        return iCIFAR10()
    elif name == "cifar100":
        return iCIFAR100()
    elif name == "imagenet1000":
        return iImageNet1000()
    elif name == "imagenet100":
        return iImageNet100()
    elif name == "cifar224":
        return iCIFAR224(args)
    elif name == "imagenetr":
        return iImageNetR(args)
    elif name == "imageneta":
        return iImageNetA()
    elif name == "cub":
        return CUB()
    elif name == "objectnet":
        return objectnet()
    elif name == "omnibenchmark":
        return omnibenchmark()
    elif name == "vtab":
        return vtab()

    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
    accimage is available on conda-forge.
    """
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)

class OnlineSampler(Sampler):
    def __init__(
        self,
        data_source: Optional[Sized] = None,
        num_tasks: int = 1,
        m: int = 10,
        n: int = 50,
        rnd_seed: int = 0,
        cur_iter: int = 0,
        varing_NM: bool = False,
        num_replicas: int = None,
        rank: int = None,
        classes=None,
        targets=None,
    ) -> None:

        self.data_source = data_source
        if classes is None or targets is None:
            if data_source is None:
                raise ValueError("OnlineSampler requires either data_source or both classes and targets.")
            classes = data_source.classes
            targets = data_source.targets
        self.classes = list(classes)
        self.targets = list(targets)
        self.generator  = torch.Generator().manual_seed(rnd_seed)
        
        self.n  = n
        self.m  = m
        self.varing_NM = varing_NM
        self.task = cur_iter

        self.num_tasks = num_tasks

        if num_replicas is None and dist.is_available() and dist.is_initialized():
            num_replicas = dist.get_world_size()
        if rank is None and dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()

        self.distributed = num_replicas is not None and rank is not None
        self.num_replicas = num_replicas if num_replicas is not None else 1
        self.rank = rank if rank is not None else 0

        self.disjoint_num   = len(self.classes) * n // 100
        self.disjoint_num   = int(self.disjoint_num // num_tasks) * num_tasks
        self.blurry_num     = len(self.classes) - self.disjoint_num
        self.blurry_num     = int(self.blurry_num // num_tasks) * num_tasks

        if not self.varing_NM:
            class_order         = torch.randperm(len(self.classes), generator=self.generator)
            self.disjoint_classes   = class_order[:self.disjoint_num]
            self.disjoint_classes   = self.disjoint_classes.reshape(num_tasks, -1).tolist()
            self.blurry_classes     = class_order[self.disjoint_num:self.disjoint_num + self.blurry_num]
            self.blurry_classes     = self.blurry_classes.reshape(num_tasks, -1).tolist()

            print("disjoint classes: ", self.disjoint_classes)
            print("blurry classes: ", self.blurry_classes)
            self.disjoint_indices   = [[] for _ in range(num_tasks)]
            self.blurry_indices     = [[] for _ in range(num_tasks)]
            for i in range(len(self.targets)):
                for j in range(num_tasks):
                    if self.targets[i] in self.disjoint_classes[j]:
                        self.disjoint_indices[j].append(i)
                        break
                    elif self.targets[i] in self.blurry_classes[j]:
                        self.blurry_indices[j].append(i)
                        break

            blurred = []
            for i in range(num_tasks):
                blurred += self.blurry_indices[i][:len(self.blurry_indices[i]) * m // 100]
                self.blurry_indices[i] = self.blurry_indices[i][len(self.blurry_indices[i]) * m // 100:]
            blurred = torch.tensor(blurred)
            blurred = blurred[torch.randperm(len(blurred), generator=self.generator)].tolist()
            print("blurry indices: ", len(blurred))
            num_blurred = len(blurred) // num_tasks
            for i in range(num_tasks):
                self.blurry_indices[i] += blurred[:num_blurred]
                blurred = blurred[num_blurred:]

            self.indices = [[] for _ in range(num_tasks)]
            for i in range(num_tasks):
                print("task %d: disjoint %d, blurry %d" % (i, len(self.disjoint_indices[i]), len(self.blurry_indices[i])))
                self.indices[i] = self.disjoint_indices[i] + self.blurry_indices[i]
                self.indices[i] = torch.tensor(self.indices[i])[torch.randperm(len(self.indices[i]), generator=self.generator)].tolist()
        else:

            class_order = torch.randperm(len(self.classes), generator=self.generator)
            self.disjoint_classes = class_order[:self.disjoint_num].tolist()
            if self.disjoint_num > 0:

                self.disjoint_slice = [0] + torch.randint(0, self.disjoint_num, (num_tasks - 1,), generator=self.generator).sort().values.tolist() + [self.disjoint_num]
                self.disjoint_classes = [self.disjoint_classes[self.disjoint_slice[i]:self.disjoint_slice[i + 1]] for i in range(num_tasks)]
            else:
                self.disjoint_classes = [[] for _ in range(num_tasks)]

            if self.blurry_num > 0:

                self.blurry_slice = [0] + torch.randint(0, self.blurry_num, (num_tasks - 1,), generator=self.generator).sort().values.tolist() + [self.blurry_num]
                self.blurry_classes = [class_order[self.disjoint_num + self.blurry_slice[i]:self.disjoint_num + self.blurry_slice[i + 1]].tolist() for i in range(num_tasks)]
            else:
                self.blurry_classes = [[] for _ in range(num_tasks)]

            print("disjoint classes: ", self.disjoint_classes)
            print("blurry classes: ", self.blurry_classes)

            self.disjoint_indices   = [[] for _ in range(num_tasks)]
            self.blurry_indices     = [[] for _ in range(num_tasks)]
            num_blurred = 0
            for i in range(len(self.targets)):
                for j in range(num_tasks):
                    if self.targets[i] in self.disjoint_classes[j]:
                        self.disjoint_indices[j].append(i)
                        break
                    elif self.targets[i] in self.blurry_classes[j]:
                        self.blurry_indices[j].append(i)

                        num_blurred += 1
                        break

            blurred = []
            num_blurred = num_blurred * m // 100
            if num_blurred > 0:

                num_blurred = [0] + torch.randint(0, num_blurred, (num_tasks-1,), generator=self.generator).sort().values.tolist() + [num_blurred]

                for i in range(num_tasks):
                    blurred += self.blurry_indices[i][:num_blurred[i + 1] - num_blurred[i]]
                    self.blurry_indices[i] = self.blurry_indices[i][num_blurred[i + 1] - num_blurred[i]:]
                blurred = torch.tensor(blurred)
                blurred = blurred[torch.randperm(len(blurred), generator=self.generator)].tolist()
                print("blurry indices: ", len(blurred))
                for i in range(num_tasks):
                    self.blurry_indices[i] += blurred[:num_blurred[i + 1] - num_blurred[i]]
                    blurred = blurred[num_blurred[i + 1] - num_blurred[i]:]

            self.indices = [[] for _ in range(num_tasks)]
            for i in range(num_tasks):
                print("task %d: disjoint %d, blurry %d" % (i, len(self.disjoint_indices[i]), len(self.blurry_indices[i])))
                self.indices[i] = self.disjoint_indices[i] + self.blurry_indices[i]
                self.indices[i] = torch.tensor(self.indices[i])[torch.randperm(len(self.indices[i]), generator=self.generator)].tolist()

        if self.distributed:
            self.num_samples = int(len(self.indices[self.task]) // self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas  
            self.num_selected_samples = int(len(self.indices[self.task]) // self.num_replicas)
        else:
            self.num_samples = int(len(self.indices[self.task]))
            self.total_size = self.num_samples
            self.num_selected_samples = int(len(self.indices[self.task]))

    def __iter__(self) -> Iterable[int]:
        return iter(self.get_task(self.task))

    def __len__(self) -> int:
        return self.num_selected_samples

    def set_task(self, cur_iter: int)-> None:
        
        if cur_iter >= self.num_tasks or cur_iter < 0:
            raise ValueError("task out of range")
        self.task = cur_iter

        if self.distributed:
            self.num_samples = int(len(self.indices[self.task]) // self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
            self.num_selected_samples = int(len(self.indices[self.task]) // self.num_replicas)
        else:
            self.num_samples = int(len(self.indices[self.task]))
            self.total_size = self.num_samples
            self.num_selected_samples = int(len(self.indices[self.task]))

    def get_task(self, cur_iter: int)-> Iterable[int]:
        self.set_task(cur_iter)
        if self.distributed:
            indices = self.indices[self.task][self.rank:self.total_size:self.num_replicas]
            assert len(indices) == self.num_samples
            return indices[:self.num_selected_samples]
        return self.indices[self.task]
