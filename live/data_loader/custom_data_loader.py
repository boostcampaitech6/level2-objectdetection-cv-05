from torch.utils.data import DataLoader

class CustomDataLoader(DataLoader):
    def __init__(
        self, dataset, batch_size, shuffle, num_workers):
        
        self.shuffle = shuffle
        
        # 생성자에서 데이터 로더 생성하기 위한 파라미터 선언
        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': num_workers
        }
        """ 
        like
        mnist_train_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
            )
        """
        super().__init__(**self.init_kwargs)
        
    def collate_fn(self, batch):
        return tuple(zip(*batch))