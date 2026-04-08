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

        # --- Step 5: Update baseline distance stats for expansion checks ---
        self._update_distance_stats(vectors, labels)

        self._network.train()

    def _min_distances_to_means(self, features, mean_matrix):
        """Compute min distance from each sample to nearest class mean.

        Processes in batches of 1000 to avoid creating a huge (N, C, D) array.

        Args:
            features: (N, D) numpy array of sample features.
            mean_matrix: (C, D) numpy array of class means.

        Returns:
            (N,) numpy array of min distances.
        """
        batch_size = 1000
        all_min_dists = []
        for i in range(0, len(features), batch_size):
            batch = features[i:i + batch_size]  # (B, D)
            # (B, C) — distance from each sample to each class mean
            dists = np.linalg.norm(
                batch[:, None, :] - mean_matrix[None, :, :], axis=2
            )
            all_min_dists.append(dists.min(axis=1))  # (B,)
        return np.concatenate(all_min_dists)

    def _get_mean_matrix(self):
        """Get (C, D) matrix of latest class means."""
        mean_list = []
        for c in sorted(self._class_mean_history.keys()):
            latest_task = max(self._class_mean_history[c].keys())
            mean_list.append(self._class_mean_history[c][latest_task])
        return np.stack(mean_list)

    def _update_distance_stats(self, vectors, labels=None):  # labels kept for future use
        """Compute baseline distance stats from current data.

        After each task, we know how far samples sit from their nearest class
        mean under the current backbone. This becomes the baseline for
        detecting novelty in the next task via z-scores.

        Args:
            vectors: (N, D) feature array for all seen samples.
            labels:  (N,) class labels.
        """
        mean_matrix = self._get_mean_matrix()  # (C, D)
        min_dists = self._min_distances_to_means(vectors, mean_matrix)  # (N,)

        self._dist_mean = min_dists.mean()
        self._dist_std = min_dists.std() + 1e-8

        logging.info(
            "NoveltyMixin: distance stats — "
            "mean={:.4f}, std={:.4f}, n_classes={}".format(
                self._dist_mean, self._dist_std, len(mean_matrix)
            )
        )

    def check_expansion(self, task_data_loader, diff_threshold=3.5):
        """Check whether new task data is novel enough to trigger adapter expansion.

        Compares the mean distance of new task samples to nearest known class
        mean against the baseline mean distance from previous task. If new data
        sits significantly farther from known means, it's novel.

        Args:
            task_data_loader: DataLoader for the incoming task's training data.
            diff_threshold: if (new_mean_dist - baseline_mean_dist) > this, expand.

        Returns:
            bool: True if expansion should be triggered.
        """
        # First task — no history, must expand
        if not hasattr(self, '_class_mean_history') or len(self._class_mean_history) == 0:
            logging.info("NoveltyMixin: no class means yet -> expand=True")
            return True

        # No baseline stats yet
        if not hasattr(self, '_dist_mean'):
            logging.info("NoveltyMixin: no distance stats yet -> expand=True")
            return True

        network = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
        network.eval()

        mean_matrix = self._get_mean_matrix()  # (C, D)

        # Extract features for new task data
        all_feats = []
        with torch.no_grad():
            for _, inputs, _ in task_data_loader:
                out = network.extract_vector(inputs.to(self._device))
                all_feats.append(out["features"].cpu().numpy())
        features = np.concatenate(all_feats)  # (N, D)

        # Mean distance of new data to nearest known class mean
        min_dists = self._min_distances_to_means(features, mean_matrix)
        new_mean_dist = min_dists.mean()

        # Raw difference: how much farther is new data from known means
        diff = new_mean_dist - self._dist_mean
        should_expand = diff > diff_threshold

        logging.info(
            "NoveltyMixin: expansion check — "
            "new_mean_dist={:.4f}, baseline_mean_dist={:.4f}, "
            "diff={:.4f}, threshold={:.4f}, expand={}".format(
                new_mean_dist, self._dist_mean,
                diff, diff_threshold, should_expand
            )
        )

        return should_expand
