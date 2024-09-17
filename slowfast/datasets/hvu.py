import os
import random
import torch
import torch.utils.data
from iopath.common.file_io import g_pathmgr
from torchvision import transforms

import slowfast.utils.logging as logging

from . import decoder as decoder
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY
from .random_erasing import RandomErasing
from .transform import create_random_augment

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Hvu(torch.utils.data.Dataset):
    """
    Multilabel Video Loader. This class loads videos and supports multilabel classification.
    For each video, multiple labels are loaded and returned.
    """

    def __init__(self, cfg, mode, num_retries=10):
        """
        Args:
            cfg (CfgNode): Configuration node.
            mode (str): One of `train`, `val`, or `test`. Determines which split to use.
            num_retries (int): Number of retries for loading videos.
        """
        assert mode in ["train", "val", "test"], "Split '{}' not supported".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._num_retries = num_retries

        # Determine the number of clips to sample based on mode
        if self.mode in ["train", "val"]:
            self._num_clips = 1
            cfg.TEST.NUM_ENSEMBLE_VIEWS = 1
            cfg.TEST.NUM_SPATIAL_CROPS = 1
        elif self.mode in ["test"]:
            self._num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS

        logger.info(f"Constructing MultilabelVideoDataset {mode}...")
        self._construct_loader()
        self.aug = False
        self.rand_erase = False

        if self.mode == "train" and self.cfg.AUG.ENABLE:
            self.aug = True
            if self.cfg.AUG.RE_PROB > 0:
                self.rand_erase = True

    def _construct_loader(self):
        """
        Construct the dataset by loading the CSV file and processing video paths and labels.
        """
        path_to_file = os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, f"{self.mode}.csv")
        assert g_pathmgr.exists(path_to_file), f"{path_to_file} not found"

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        with g_pathmgr.open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                path, label_str = path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR)
                labels = list(map(int, label_str.split('|')))  # Convert comma-separated labels to a list of ints
                # one_hot_labels = [0] * int(self.cfg.MODEL.NUM_CLASSES)
                one_hot_labels = torch.zeros(self.cfg.MODEL.NUM_CLASSES, dtype=torch.float32)

                # Set 1's at the positions specified in labels
                for label in labels:
                    one_hot_labels[label] = 1
                
                for idx in range(self._num_clips):
                    self._path_to_videos.append(os.path.join(self.cfg.DATA.PATH_PREFIX, path))
                    self._labels.append(one_hot_labels)
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}

        assert len(self._path_to_videos) > 0, f"Failed to load dataset split {self.mode} from {path_to_file}"
        logger.info(f"Constructed dataloader with {len(self._path_to_videos)} videos from {path_to_file}")

    def __getitem__(self, index):
        """
        Retrieve video frames and corresponding multilabels.
        Args:
            index (int): Index of the video sample.
        Returns:
            frames (tensor): Video frames (C x T x H x W).
            labels (list of int): Multilabels for the video.
            index (int): The index of the current video.
        """
        short_cycle_idx = None
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode in ["train"]:
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(round(self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx] * self.cfg.MULTIGRID.DEFAULT_S))
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                min_scale = int(round(float(min_scale) * crop_size / self.cfg.MULTIGRID.DEFAULT_S))
        elif self.mode in ["val", "test"]:
            temporal_sample_index = self._spatial_temporal_idx[index] // self.cfg.TEST.NUM_SPATIAL_CROPS
            spatial_sample_index = (self._spatial_temporal_idx[index] % self.cfg.TEST.NUM_SPATIAL_CROPS
                                    if self.cfg.TEST.NUM_SPATIAL_CROPS > 1 else 1)
            min_scale, max_scale, crop_size = ([self.cfg.DATA.TEST_CROP_SIZE] * 3)
        else:
            raise NotImplementedError(f"Does not support {self.mode} mode")

        sampling_rate = utils.get_random_sampling_rate(self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE, self.cfg.DATA.SAMPLING_RATE)

        for i_try in range(self._num_retries):
            video_container = None
            try:
                video_container = container.get_video_container(self._path_to_videos[index], 
                                                                self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                                                                self.cfg.DATA.DECODING_BACKEND)
            except Exception as e:
                logger.warning(f"Failed to load video {self._path_to_videos[index]}: {e}")

            if video_container is None:
                logger.warning(
                    "Failed to load video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                elif self.mode in ["test"] and i_try > self._num_retries // 2:
                    # BUG: should not repeat video
                    logger.info(
                        "Failed to load video idx {} from {}; use idx {}".format(
                            index, self._path_to_videos[index], index - 1
                        )
                    )
                    index = index - 1
                continue

            frames = decoder.decode(
                video_container,
                sampling_rate,
                self.cfg.DATA.NUM_FRAMES,
                temporal_sample_index,
                self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                video_meta=self._video_meta[index],
                target_fps=self.cfg.DATA.TARGET_FPS,
                backend=self.cfg.DATA.DECODING_BACKEND,
                max_spatial_scale=min_scale,
                use_offset=self.cfg.DATA.USE_OFFSET_SAMPLING,
                sparse=True
            )

            if frames is None:
                logger.warning(f"Failed to decode video {self._path_to_videos[index]} on trial {i_try}")
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            if self.aug:
                if self.cfg.AUG.NUM_SAMPLE > 1:
                    frame_list, label_list, index_list = [], [], []
                    for _ in range(self.cfg.AUG.NUM_SAMPLE):
                        new_frames = self._aug_frame(
                            frames,
                            spatial_sample_index,
                            min_scale,
                            max_scale,
                            crop_size,
                        )
                        labels = self._labels[index]
                        new_frames = utils.pack_pathway_output(self.cfg, new_frames)
                        frame_list.append(new_frames)
                        label_list.append(labels)
                        index_list.append(index)
                    return frame_list, label_list, index_list, {}

                else:
                    frames = self._aug_frame(frames, spatial_sample_index, min_scale, max_scale, crop_size)

            else:
                frames = utils.tensor_normalize(frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD)
                frames = frames.permute(3, 0, 1, 2)
                frames = utils.spatial_sampling(
                    frames,
                    spatial_idx=spatial_sample_index,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    crop_size=crop_size,
                    random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                    inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE
                )

            labels = self._labels[index]
            frames = utils.pack_pathway_output(self.cfg, frames)
            return frames, labels, index, {}

        raise RuntimeError(f"Failed to load video {self._path_to_videos[index]} after {self._num_retries} retries")
    
    def _aug_frame(
        self,
        frames,
        spatial_sample_index,
        min_scale,
        max_scale,
        crop_size,
    ):
        aug_transform = create_random_augment(
            input_size=(frames.size(1), frames.size(2)),
            auto_augment=self.cfg.AUG.AA_TYPE,
            interpolation=self.cfg.AUG.INTERPOLATION,
        )
        # T H W C -> T C H W.
        frames = frames.permute(0, 3, 1, 2)
        list_img = self._frame_to_list_img(frames)
        list_img = aug_transform(list_img)
        frames = self._list_img_to_frames(list_img)
        frames = frames.permute(0, 2, 3, 1)

        frames = utils.tensor_normalize(
            frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
        )
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            self.cfg.DATA.TRAIN_JITTER_SCALES_RELATIVE,
            self.cfg.DATA.TRAIN_JITTER_ASPECT_RELATIVE,
        )
        relative_scales = (
            None if (self.mode not in ["train"] or len(scl) == 0) else scl
        )
        relative_aspect = (
            None if (self.mode not in ["train"] or len(asp) == 0) else asp
        )
        frames = utils.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            aspect_ratio=relative_aspect,
            scale=relative_scales,
            motion_shift=self.cfg.DATA.TRAIN_JITTER_MOTION_SHIFT
            if self.mode in ["train"]
            else False,
        )

        if self.rand_erase:
            erase_transform = RandomErasing(
                self.cfg.AUG.RE_PROB,
                mode=self.cfg.AUG.RE_MODE,
                max_count=self.cfg.AUG.RE_COUNT,
                num_splits=self.cfg.AUG.RE_COUNT,
                device="cpu",
            )
            frames = frames.permute(1, 0, 2, 3)
            frames = erase_transform(frames)
            frames = frames.permute(1, 0, 2, 3)

        return frames

    def _frame_to_list_img(self, frames):
        img_list = [
            transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))
        ]
        return img_list

    def _list_img_to_frames(self, img_list):
        img_list = [transforms.ToTensor()(img) for img in img_list]
        return torch.stack(img_list)

    def __len__(self):
        return len(self._path_to_videos)

    @property
    def num_videos(self):
        return len(self._path_to_videos)
