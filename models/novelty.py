import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

num_workers = 8


class NoveltyMixin:
    """Mixin for class mean tracking and drift computation.

    Add to Learner via multiple inheritance. Expects the host class to have:
      - self._network, self._device, self.batch_size
      - self.data_manager, self._cur_task
      - self._known_classes, self._total_classes
      - self.feature_dim
    """

    def _compute_class_means(self):
        """Compute class means for ALL seen classes using the current backbone,
        then log Euclidean drift from the previous task's snapshot.

        Called from after_task() in sema.py after each task finishes training.

        Why re-extract all classes (not just current task)?
        The backbone changes after training on each task. Old class means were
        computed with the old backbone. To measure pure representation drift
        caused by distribution change, we re-extract all old classes' features
        through the new backbone and compare with the previous snapshot.
        """

        # --- Step 1: One-time init ---
        # _class_mean_history: {class_id: {task_id: mean_vector}}
        #   Stores a fresh mean snapshot per class per task, always extracted
        #   with that task's backbone. This lets us compare how the same class
        #   is represented by different versions of the backbone.
        if not hasattr(self, '_class_mean_history'):
            self._class_mean_history = {}

        self._network.eval()

        # --- Step 2: Extract features for ALL seen classes ---
        # In online sampler mode, get_dataset only works with single-task index ranges
        # (e.g. [0,10), [10,20)). Passing [0,20) won't match any task and misses
        # blurry classes. So we load each task separately and combine.
        # mode="test" gives data without augmentation for clean, comparable features.
        network = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
        all_vectors = []
        all_labels = []
        task_size = self.data_manager.get_task_size(0)
        for t in range(self._cur_task + 1):
            start = t * task_size
            end = start + task_size
            task_dataset = self.data_manager.get_dataset(
                np.arange(start, end),
                source="train",
                mode="test",
            )
            task_loader = DataLoader(task_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
            with torch.no_grad():
                for _, inputs, targets in task_loader:
                    out = network.extract_vector(inputs.to(self._device))
                    all_vectors.append(out["features"].cpu().numpy())
                    all_labels.append(targets.numpy())

        vectors = np.concatenate(all_vectors)  # (N_all, 768)
        labels  = np.concatenate(all_labels)   # (N_all,)

        # --- Step 3: Compute mean for each class using current backbone ---
        # This is NOT incremental — it's a fresh mean from all samples of each class
        # passed through the current backbone. This way old and new snapshots are
        # directly comparable (same backbone version).
        for c in np.unique(labels):
            mask = labels == c
            vecs = vectors[mask]
            class_mean = np.mean(vecs, axis=0)

            # Store snapshot for this class at this task
            if c not in self._class_mean_history:
                self._class_mean_history[c] = {}
            self._class_mean_history[c][self._cur_task] = class_mean.copy()

        # Log which classes were computed
        seen_classes = np.unique(labels).tolist()
        logging.info("Task {}: class means computed for {} classes".format(self._cur_task, len(seen_classes)))
        for c in seen_classes:
            mean = self._class_mean_history[c][self._cur_task]
            n_samples = int((labels == c).sum())
            logging.info("  Class {:3d}: norm={:.4f}, samples={}, mean[:5]={}".format(
                c, np.linalg.norm(mean), n_samples,
                np.round(mean[:5], 4)
            ))

        # --- Step 4: Compute and log representation drift ---
        # For classes that existed in the previous task too, compute Euclidean distance
        # between old snapshot (old backbone) and new snapshot (new backbone).
        # This measures pure representation drift — how much the backbone's view
        # of a class changed due to training on new data.
        if self._cur_task > 0:
            logging.info("Task {}: representation drift (backbone change)".format(self._cur_task))
            for c in sorted(self._class_mean_history.keys()):
                task_ids = sorted(self._class_mean_history[c].keys())
                if len(task_ids) < 2:
                    continue
                old_task = task_ids[-2]
                new_task = task_ids[-1]
                old_mean = self._class_mean_history[c][old_task]
                new_mean = self._class_mean_history[c][new_task]
                # Euclidean distance between same class represented by two different backbones
                drift = np.linalg.norm(new_mean - old_mean)
                logging.info("  Class {}: task {} -> {}, drift = {:.6f}".format(c, old_task, new_task, drift))

        self._network.train()
