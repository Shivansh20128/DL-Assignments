import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class CustomDataset:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.data = self.load_dataset()

    def load_dataset(self):
        dataset = []
        with open(self.csv_file, 'r') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)
            for row in csv_reader:
                label = int(row[0])
                pixel_values = np.array([int(value) for value in row[1:]], dtype=np.uint8)
                image = pixel_values.reshape((28, 28))
                dataset.append((image, label))
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        return image, label

csv_train_path = 'extracted_files/mnist_train.csv'
csv_test_path = 'extracted_files/test.csv'
custom_dataset = CustomDataset(csv_train_path)
test_data_custom = CustomDataset(csv_test_path)

# Splitting the train dataset into train and validation datasets

from sklearn.model_selection import train_test_split
train_data_custom, val_data_custom = train_test_split(custom_dataset, test_size=0.1, random_state=42)

class CustomDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        self.current_index = 0

        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index >= len(self.indices):
            raise StopIteration

        batch_indices = self.indices[self.current_index:self.current_index + self.batch_size]
        batch_data = [self.dataset[i] for i in batch_indices]

        self.current_index += self.batch_size

        images, labels = zip(*batch_data)
        return np.array(images), np.array(labels)

criterion = nn.CrossEntropyLoss()

import numpy as np

class CustomFeedForwardNN:
    def __init__(self):
        self.weights = {
            'fc1': np.random.randn(28 * 28, 32),
            'fc2': np.random.randn(32, 32),
            'fc3': np.random.randn(32, 32),
            'fc4': np.random.randn(32, 32),
            'fc5': np.random.randn(32, 10)
        }

        self.biases = {
            'fc1': np.zeros((1, 32)),
            'fc2': np.zeros((1, 32)),
            'fc3': np.zeros((1, 32)),
            'fc4': np.zeros((1, 32)),
            'fc5': np.zeros((1, 10))
        }

        self.layer_outputs = {}

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        self.layer_outputs['fc0'] = x
        x = self.relu(np.dot(x, self.weights['fc1']) + self.biases['fc1'])
        self.layer_outputs['fc1'] = x
        x = self.relu(np.dot(x, self.weights['fc2']) + self.biases['fc2'])
        self.layer_outputs['fc2'] = x
        x = self.relu(np.dot(x, self.weights['fc3']) + self.biases['fc3'])
        self.layer_outputs['fc3'] = x
        x = self.relu(np.dot(x, self.weights['fc4']) + self.biases['fc4'])
        self.layer_outputs['fc4'] = x
        x = np.dot(x, self.weights['fc5']) + self.biases['fc5']
        self.layer_outputs['fc5'] = x
        return x

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_pred, y_true, epsilon=1e-15):
        m = y_true.shape[0]
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.sum(y_true * np.log(y_pred + epsilon)) / m
        return loss

    def backward(self, x, y_true, learning_rate):
        m = x.shape[0]

        # Calculate loss
        loss = self.cross_entropy_loss(self.softmax(self.layer_outputs['fc5']), y_true)

        # Backpropagation
        d_fc5 = self.softmax(self.layer_outputs['fc5'])
        d_fc5[range(m), y_true.argmax(axis=1)] -= 1
        d_fc5 /= m

        d_fc4 = np.dot(d_fc5, self.weights['fc5'].T)
        d_fc4[self.layer_outputs['fc4'] <= 0] = 0

        d_fc3 = np.dot(d_fc4, self.weights['fc4'].T)
        d_fc3[self.layer_outputs['fc3'] <= 0] = 0

        d_fc2 = np.dot(d_fc3, self.weights['fc3'].T)
        d_fc2[self.layer_outputs['fc2'] <= 0] = 0

        d_fc1 = np.dot(d_fc2, self.weights['fc2'].T)
        d_fc1[self.layer_outputs['fc1'] <= 0] = 0

        # Update weights and biases
        self.weights['fc5'] -= learning_rate * np.dot(self.layer_outputs['fc4'].T, d_fc5)
        self.biases['fc5'] -= learning_rate * np.sum(d_fc5, axis=0, keepdims=True)

        self.weights['fc4'] -= learning_rate * np.dot(self.layer_outputs['fc3'].T, d_fc4)
        self.biases['fc4'] -= learning_rate * np.sum(d_fc4, axis=0, keepdims=True)

        self.weights['fc3'] -= learning_rate * np.dot(self.layer_outputs['fc2'].T, d_fc3)
        self.biases['fc3'] -= learning_rate * np.sum(d_fc3, axis=0, keepdims=True)

        self.weights['fc2'] -= learning_rate * np.dot(self.layer_outputs['fc1'].T, d_fc2)
        self.biases['fc2'] -= learning_rate * np.sum(d_fc2, axis=0, keepdims=True)

        self.weights['fc1'] -= learning_rate * np.dot(self.layer_outputs['fc0'].T, d_fc1)
        self.biases['fc1'] -= learning_rate * np.sum(d_fc1, axis=0, keepdims=True)

        return loss

custom_model = CustomFeedForwardNN()
for epoch in range(60):
    total_correct = 0
    total_samples = 0
    custom_dataloader = CustomDataLoader(test_data_custom, 512)
    for inputs, labels in custom_dataloader:

        outputs = custom_model.forward(inputs)
        one_labels = np.eye(10)[labels]
        loss = custom_model.backward(inputs, one_labels, 1)

        acc = np.mean(np.argmax(custom_model.softmax(outputs), axis=1) == labels)
        print("Accuracy:",acc)

    
    print(f'Epoch [{epoch + 1}/60], Loss: {loss.item()}')
