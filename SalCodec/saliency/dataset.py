from skimage import io, transform
import torch
from torch.utils.data import Dataset, DataLoader
import glob



class DHF1KDataset(Dataset):
    def __init__(self, path_data="/data/lqh/Saliency/DHF1K/dataset/train"):
        self.path_data = path_data
        self.train_pair =[]
        imgs = glob.glob(path_data + "/*/images/*.jpg")
        maps = list(map(lambda x: x.replace("images", "maps").replace("jpg", "png"), imgs))
        self.data = list(zip(imgs, maps))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        img = io.imread(data[0])
        sal = io.imread(data[1])
        img = transform.resize(img, (224, 384))
        sal = transform.resize(sal, (224, 384))
        img = torch.from_numpy(img.copy()).contiguous().float()
        sal = torch.from_numpy(sal.copy()).contiguous().float()
        img = img.permute(2, 0, 1)
        sal = sal[None, :, :]      
        return img, sal

# from gist.github.com/MFreidank/821cc87b012c53fade03b0c7aba13958 
class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch

if __name__ == "__main__":
    dataset = DHF1KDataset()
    data = dataset.__getitem__(5)