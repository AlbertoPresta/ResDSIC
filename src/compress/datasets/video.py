import random

from pathlib import Path
import os
import numpy as np
import torch
from compressai.ops import compute_padding
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from compressai.registry import register_dataset


@register_dataset("VideoFolder")
class VideoFolder(Dataset):
    """Load a video folder database. Training and testing video clips
    are stored in a directorie containing mnay sub-directorie like Vimeo90K Dataset:

    .. code-block::

        - rootdir/
            train.list
            test.list
            - sequences/
                - 00010/
                    ...
                    -0932/
                    -0933/
                    ...
                - 00011/
                    ...
                - 00012/
                    ...

    training and testing (valid) clips are withdrew from sub-directory navigated by
    corresponding input files listing relevant folders.

    This class returns a set of three video frames in a tuple.
    Random interval can be applied to if subfolders includes more than 6 frames.

    Args:
        root (string): root directory of the dataset
        rnd_interval (bool): enable random interval [1,2,3] when drawing sample frames
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'test')
    """

    def __init__(
        self,
        root,
        rnd_interval=False,
        rnd_temp_order=False,
        transform=None,
        max_frames = 3, #hard coding for now 
        split="train",
    ):
        if transform is None:
            raise RuntimeError("Transform must be applied")

        
        
        
        if split != "valid":
            splitfile = Path(f"{root}/{split}.list")
            splitdir = Path(f"{root}/sequences")
        else:
            splitfile = Path(f"{root}/test.list")
            splitdir = Path(f"{root}/sequences")          

        if not splitfile.is_file():
            raise RuntimeError(f'Missing file "{splitfile}"')

        if not splitdir.is_dir():
            raise RuntimeError(f'Missing directory "{splitdir}"')

        with open(splitfile, "r") as f_in:
            self.sample_folders = [Path(f"{splitdir}/{f.strip()}") for f in f_in]

        self.max_frames = max_frames  # hard coding for now
        self.rnd_interval = rnd_interval
        self.rnd_temp_order = rnd_temp_order
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """

        sample_folder = self.sample_folders[index]
        samples = sorted(f for f in sample_folder.iterdir() if f.is_file())

        max_interval = (len(samples) + 2) // self.max_frames
        interval = random.randint(1, max_interval) if self.rnd_interval else 1
        frame_paths = (samples[::interval])[: self.max_frames]

        frames = np.concatenate([np.asarray(Image.open(p).convert("RGB")) for p in frame_paths], axis=-1)
        frames = torch.chunk(self.transform(frames), self.max_frames)

        if self.rnd_temp_order:
            if random.random() < 0.5:
                return frames[::-1]

        return frames

    def __len__(self):
        return len(self.sample_folders)
    

def pad(x, p: int = 2 ** (4 + 3)):
    h, w = x.size(2), x.size(3)
    padding, _ = compute_padding(h, w, min_div=p)
    x = F.pad(x, padding, mode="constant", value=0)
    return x, padding


def crop(x, padding):
    return F.pad(x, tuple(-p for p in padding))

class Vimeo90kDataset(Dataset):

    def __init__(self, root, 
                        transform, 
                        split="train", 
                        tuplet=7,
                        max_frames = 7,
                        rnd_interval = False,
                        rnd_temp_order = False,
                        pad = False):
        
        list_path = Path(root) / self._list_filename(split, tuplet)
        self.max_frames = max_frames
        self.rnd_temp_order = rnd_temp_order
        self.rnd_interval = rnd_interval
        self.samples_folder = []
        self.pad = pad

        with open(list_path) as f:

            for line in f: 
                if line.strip() != "":
                    #print("-->",line.strip()," ",line)
                    self.samples_folder.append(f"{root}/sequences/{line.strip()}")
                    #for idx in range(1,tuplet + 1):
                    #    self.samples = [f"{root}/sequences/{line.rstrip()}/im{idx}.png"]

        self.transform = transform



    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """

        sample_folder = self.samples_folder[index]
        samples = [os.path.join(sample_folder,f) for f in os.listdir(sample_folder)] #sorted(f for f in sample_folder.iterdir() if f.is_file())

        max_interval = (len(samples) + 2) // self.max_frames
        interval = random.randint(1, max_interval) if self.rnd_interval else 1
        frame_paths = (samples[::interval])[: self.max_frames]

        
        if self.pad is False:
            frames = np.concatenate([np.asarray(Image.open(p).convert("RGB")) for p in frame_paths], axis=-1)
            frames = torch.chunk(self.transform(frames), self.max_frames)
        else:
            fr = []
            pd = []
            for p in frame_paths:
                f,p = pad(Image.open(p).convert("RGB"))
                fr.append(f)
                pd.append(p)

            frames = np.concatenate(fr, axis=-1)
            frames = torch.chunk(self.transform(frames), self.max_frames)           

        if self.rnd_temp_order:
            if random.random() < 0.5: #dddd
                return frames[::-1]

        if self.pad:
            return frames,pd
        else:
            return frames





    def __len__(self):
        return len(self.samples_folder)

    def _list_filename(self, split: str, tuplet: int) -> str:
        tuplet_prefix = {3: "tri", 7: "sep"}[tuplet]
        list_suffix = {"train": "trainlist", "valid": "validlist","test":"testlist"}[split]
        return f"{tuplet_prefix}_{list_suffix}.txt"