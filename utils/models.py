import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class FaceCNN(nn.Module):
    def __init__(self, num_classes, input_shape=(3, 128, 128)):
        super(FaceCNN, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)

        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self._forward_features(dummy_input)
            self.flattened_size = dummy_output.view(1, -1).shape[1]

        self.linear_1 = nn.Linear(self.flattened_size, 256)
        self.linear_2 = nn.Linear(256, num_classes)

    def _forward_features(self, x):
        x = self.pool(F.relu(self.conv_1(x)))
        x = self.pool(F.relu(self.conv_2(x)))
        x = self.pool(F.relu(self.conv_3(x)))
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.linear_1(x))
        x = self.linear_2(x)
        return x

def model_functions(model):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    return loss_fn, optimizer

def train_model(model, train, validation, loss_fn, optimizer, device, epochs):
    for epoch in range(epochs):
        model.train()

        total = 0
        correct = 0
        running_loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(train):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if batch_idx % 10 == 0:
                if batch_idx % 10 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx}/{len(train)}], 'f'Loss: {loss.item():.4f}, Accuracy: {100 * correct / total:.2f}%')

    epoch_loss = running_loss / len(train)
    epoch_acc = 100 * correct / total
    print(f'==> Epoch {epoch+1} complete: Avg Loss: {epoch_loss:.4f}, Avg Accuracy: {epoch_acc:.2f}%')
    validate_model(model=model, validation=validation, loss_fn=loss_fn, device=device)
    return model

def validate_model(model, validation, loss_fn, device):
    model.eval()
    
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in validation:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_loss /= len(validation)
    val_acc = 100 * val_correct / val_total
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')
    print("---------------------------------------------------------------\n")

def save_model(model):
    torch.save(model.state_dict(), 'face_recognition_model.pth')
    print("Model saved !!!")
    print("---------------------------------------------------------------\n")

def train_eval_save_model(num_classes, train, validation, device, epochs):
    model = FaceCNN(num_classes=num_classes, input_shape=(3, 160, 160)).to(device)
    loss_fn, optimizer = model_functions(model)
    model = train_model(model=model, train=train, validation=validation, loss_fn=loss_fn, optimizer=optimizer, device=device, epochs=epochs)
    save_model(model=model)