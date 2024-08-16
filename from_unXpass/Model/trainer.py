import pandas as pd
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from from_unXpass.dataset.dataloader import SoccerDataset
from from_unXpass.dataset.Into_Soccermap_tensor import ToSoccerMapTensor, convert_row_to_sample
from from_unXpass.Model.Soccermap import PytorchSoccerMapModel
import logging
from pytorch_lightning.callbacks import Callback

class ModelTrainer:
    def __init__(self, data_path, transform, lr=1e-4, batch_size=32, train_split=0.6, val_split=0.2, max_epochs=10):
        self.data_path = data_path
        self.transform = transform
        self.lr = lr
        self.batch_size = batch_size
        self.train_split = train_split
        self.val_split = val_split
        self.max_epochs = max_epochs

        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_data(self):
        self.logger.info("데이터 로딩 중...")
        df = pd.read_csv(self.data_path, index_col=0)
        samples = df.apply(convert_row_to_sample, axis=1).tolist()

        dataset = SoccerDataset(samples, transform=self.transform)
        train_size = int(self.train_split * len(dataset))
        val_size = int(self.val_split * len(dataset))
        test_size = len(dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size])

    def train(self):
        self.logger.info("학습 시작...")
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size)

        model = PytorchSoccerMapModel(lr=self.lr)
        checkpoint_callback = ModelCheckpoint(monitor='val/loss', save_top_k=1, mode='min')
        early_stop_callback = EarlyStopping(monitor='val/loss', patience=3, mode='min')

        trainer = Trainer(
            callbacks=[checkpoint_callback, early_stop_callback],
            max_epochs=self.max_epochs,
            logger=True,
            log_every_n_steps=10
        )

        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        best_model_path = checkpoint_callback.best_model_path
        self.model = PytorchSoccerMapModel.load_from_checkpoint(best_model_path)
        self.trainer = trainer
        self.logger.info(f"학습 완료. 최적의 모델이 {best_model_path}에 저장되었습니다.")

    def evaluate(self):
        self.logger.info("평가 시작...")
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size)
        results = self.trainer.test(self.model, test_loader)
        print(f"테스트 결과: {results}")

if __name__ == "__main__":
    data_path = '../dataset/total_data_with_state_label_mask.csv'
    transform = ToSoccerMapTensor(dim=(68, 104))

    # 트레이너 초기화 및 모델 학습
    trainer = ModelTrainer(data_path=data_path, transform=transform, lr=1e-5, max_epochs=1)
    trainer.load_data()
    trainer.train()

    # 모델 평가
    trainer.evaluate()
