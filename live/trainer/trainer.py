import torch
import os
from tqdm import tqdm
from utils.util import Averager


class CustomTrainer:
    def __init__(
        self, epochs, save_path, train_data_loader, optimizer, model, device
    ) -> None:
        self.num_epochs = epochs
        self.train_data_loader = train_data_loader
        self.optimizer = optimizer
        self.model = model
        self.device = device
        self.save_path = save_path

    def train_fn(self):
        best_loss = 1000
        loss_hist = Averager()
        for epoch in range(self.num_epochs):
            loss_hist.reset()

            for images, targets, image_ids in tqdm(self.train_data_loader):
                # gpu 계산을 위해 image.to(device)
                images = list(image.float().to(self.device) for image in images)
                targets = [
                    {k: v.to(self.device) for k, v in t.items()} for t in targets
                ]

                # calculate loss
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()

                loss_hist.send(loss_value)

                # backward
                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()

            print(f"Epoch #{epoch+1} loss: {loss_hist.value}")
            if loss_hist.value < best_loss:
                save_dir = os.path.dirname(self.save_path)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                torch.save(self.model.state_dict(), self.save_path)
                best_loss = loss_hist.value
