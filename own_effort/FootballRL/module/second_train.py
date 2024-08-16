import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from network import SoccerMap
from dataset import SoccerDataset


def train(num_epochs, save_path):
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for state_inputs, end_x, end_y, success_labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            state_inputs, end_x, end_y, success_labels = state_inputs.cuda(), end_x.cuda(), end_y.cuda(), success_labels.cuda()
            optimizer.zero_grad()
            success_predictions = model(state_inputs)
            print("[before success_predictions", success_predictions)
            batch_size = state_inputs.size(0)
            end_x = end_x.long().clamp(0, 79)
            end_y = end_y.long().clamp(0, 119)
            indices = end_x * 120 + end_y
            success_predictions = success_predictions.view(batch_size, -1)
            print("after success_predictions", success_predictions)
            predicted_probabilities = success_predictions.gather(1, indices.unsqueeze(1)).squeeze(1)
            print("predicted_probabilities", predicted_probabilities)
            loss = criterion(predicted_probabilities, success_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        val_loss = evaluate(val_loader)
        print(f"Epoch {epoch + 1}, Training Loss: {avg_loss}, Validation Loss: {val_loss}")

        # 마지막 에폭에서 모델 저장
        if epoch == num_epochs - 1:
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")


def evaluate(loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for state_inputs, end_x, end_y, success_labels in loader:
            state_inputs, end_x, end_y, success_labels = state_inputs.cuda(), end_x.cuda(), end_y.cuda(), success_labels.cuda()
            success_predictions = model(state_inputs)
            batch_size = state_inputs.size(0)
            end_x = end_x.long().clamp(0, 79)
            end_y = end_y.long().clamp(0, 119)
            indices = end_x * 120 + end_y
            success_predictions = success_predictions.view(batch_size, -1)
            predicted_probabilities = success_predictions.gather(1, indices.unsqueeze(1)).squeeze(1)
            loss = criterion(predicted_probabilities, success_labels)
            total_loss += loss.item()
    return total_loss / len(loader)


if __name__ == "__main__":
    data = pd.read_csv("../dataset/total_state_data_with_paths.csv", index_col=0)
    # 데이터 로드 및 분할
    train_data, test_data = train_test_split(data, test_size=0.2, stratify=data['outcome_name'])
    train_data, val_data = train_test_split(train_data, test_size=0.25, stratify=train_data['outcome_name'])

    train_dataset = SoccerDataset(train_data)
    val_dataset = SoccerDataset(val_data)
    test_dataset = SoccerDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 모델, 손실 함수, 최적화 설정
    model = SoccerMap(input_channels=11).cuda()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 모델 저장 경로


    # 학습 및 평가
    num_epochs = 30
    save_path = f"final_model_{num_epochs}.pth"
    train(num_epochs, save_path)

    test_loss = evaluate(test_loader)
    print(f"Test Loss: {test_loss}")
