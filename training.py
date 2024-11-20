import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import FullyConnectedNN



def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Train the model for one epoch
    
    Args:
    model: nn.Module, the neural network model
    loader: torch DataLoader, the training data loader
    criterion: loss function
    optimizer: optimizer
    device: str, 'cpu' or 'cuda'
    
    Returns:
    epoch_loss: float, the average loss for the epoch
    epoch_accuracy: float, the accuracy for the epoch
    """
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        # Move data to the specified device (CPU or GPU)
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update running loss and accuracy
        running_loss += loss.item() * images.size(0)  # Multiply by batch size
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total
    return epoch_loss, epoch_accuracy


def validate(model, loader, criterion, device):
    """
    Validate the model

    Args:
    model: nn.Module, the neural network model
    loader: torch DataLoader, the validation data loader
    criterion: loss function
    device: str, 'cpu' or 'cuda'

    Returns:
    epoch_loss: float, the average loss for the epoch
    epoch_accuracy: float, the accuracy for the epoch
    """
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradient computation
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Update running loss and accuracy
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total
    return epoch_loss, epoch_accuracy, all_preds, all_labels



def train_and_evaluate(lr, num_hidden_layers, hidden_size, batch_size, activation, dropout_rate, decay_factor, train_input, train_labels, val_input, val_labels, test_input, test_labels, optimiser = 'adam', no_epochs=3):
    """
    Master function to train and evaluate the model
    
    Args:
    lr: float, learning rate
    num_hidden_layers: int, number of hidden layers
    hidden_size: int, number of hidden units
    batch_size: int, batch size
    activation: str, activation function
    train_input: list of torch tensors, training input data
    train_labels: list of ints, training labels
    val_input: list of torch tensors, validation input data
    val_labels: list of ints, validation labels
    test_input: list of torch tensors, test input data
    test_labels: list of ints, test labels
    no_epochs: int, number of epochs
    optimiser: str, 'adam', 'sgd', or 'rmsprop'
    
    Returns:
    val_loss: float, validation loss
    val_acc: float, validation accuracy
    test_loss: float, test loss
    test_acc: float, test accuracy
    train_loss: float, train loss
    train_acc: float, train accuracy
    train_loss_list: list of floats, training loss for each epoch
    val_loss_list: list of floats, validation loss for each epoch
    model: nn.Module, trained model
    """
    # Initialize the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FullyConnectedNN(
        input_size=1568,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        activation=activation,
        dropout_rate=dropout_rate,
        decay_factor=decay_factor,
        num_classes=20,
    ).to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    if optimiser == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimiser == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimiser == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    # Stack the train and label sets
    train_data_stacked = torch.stack(train_input)
    train_labels_stacked = torch.tensor(train_labels)

    val_data_stacked = torch.stack(val_input)
    val_labels_stacked = torch.tensor(val_labels)

    test_data_stacked = torch.stack(test_input)
    test_labels_stacked = torch.tensor(test_labels)

    # Combine data and labels into TensorDataset
    train_tensor = TensorDataset(train_data_stacked, train_labels_stacked)
    val_tensor = TensorDataset(val_data_stacked, val_labels_stacked)
    test_tensor = TensorDataset(test_data_stacked, test_labels_stacked)

    # Create data loaders
    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tensor, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_tensor, batch_size=batch_size, shuffle=False)


    # Train the model
    train_loss_list = []
    val_loss_list = []

    train_acc_list = []
    val_acc_list = []
    for epoch in range(no_epochs):  # Keep epochs small for quick tuning
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _,_ = validate(model, val_loader, criterion, device)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        print(f"Epoch {epoch+1}/{no_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

    test_loss, test_acc, all_preds, all_labels = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    return val_loss, val_acc, test_loss, test_acc, train_loss, train_acc, train_loss_list, val_loss_list, train_acc_list, val_acc_list, all_preds, all_labels, model




def objective(trial, combined_train_images, combined_train_labels, val_images, val_labels):
    """
    Objective function for Optuna to optimize
    
    Args:
    trial: optuna.trial.Trial, a trial object
    
    Returns:
    val_acc: float, validation accuracy
    """
    lr = trial.suggest_float('lr', 0.0001, 0.001, log=True)
    num_hidden_layers = trial.suggest_int('num_hidden_layers', 1, 10)
    hidden_size = trial.suggest_int('hidden_size', 128, 512)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    activation_class = trial.suggest_categorical('activation_class', ['ReLU', 'Sigmoid', 'Tanh', 'LeakyReLU'])
    activation = getattr(nn, activation_class)()
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    decay_factor = trial.suggest_float('decay_factor', 0, 0.9)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FullyConnectedNN(
        input_size=1568,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        activation=activation,
        dropout_rate=dropout_rate,
        decay_factor=decay_factor,
        num_classes=20,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Data loaders
    train_data_stacked = torch.stack(combined_train_images)
    train_labels_stacked = torch.tensor(combined_train_labels)

    val_data_stacked = torch.stack(val_images)
    val_labels_stacked = torch.tensor(val_labels)

    train_tensor = TensorDataset(train_data_stacked, train_labels_stacked)
    val_tensor = TensorDataset(val_data_stacked, val_labels_stacked)

    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tensor, batch_size=batch_size, shuffle=False)

    for epoch in range(3):
        _, _ = train_one_epoch(model, train_loader, criterion, optimizer, device)
        _, val_acc, _ ,_ = validate(model, val_loader, criterion, device)
    
    return val_acc



def FCNN_optimised(combined_train_images, combined_train_labels, val_images, val_labels, test_images, test_labels, n_trials = 10, no_epochs = 10):
    """
    Optuna optimisation of the FullyConnectedNN model
    
    Args:
    n_trials: int, number of trials
    no_epochs: int, number of epochs
    combined_train_images: list of torch tensors, training input data
    combined_train_labels: list of ints, training labels
    val_images: list of torch tensors, validation input data
    val_labels: list of ints, validation labels
    test_images: list of torch tensors, test input data
    test_labels: list of ints, test labels
    
    Returns:
    val_loss: float, validation loss
    val_acc: float, validation accuracy
    test_loss: float, test loss
    test_acc: float, test accuracy
    train_loss: float, train loss
    train_acc: float, train accuracy
    train_loss_list: list of floats, training loss for each epoch
    val_loss_list: list of floats, validation loss for each epoch
    model: nn.Module, trained model
    """
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials)

    print(f'Best hyperparameters: {study.best_params}')
    print(f'Best validation accuracy{study.best_value}')

    best_lr = study.best_params['lr']
    best_num_hidden_layers = study.best_params['num_hidden_layers']
    best_hidden_size = study.best_params['hidden_size']
    best_batch_size = study.best_params['batch_size']
    best_activation_class = study.best_params['activation_class']
    best_activation = getattr(nn, best_activation_class)()
    best_dropout_rate = study.best_params['dropout_rate']
    best_decay_factor = study.best_params['decay_factor']

    val_loss, val_acc, test_loss, test_acc, train_loss, train_acc, train_list, val_list, train_acc_list, val_acc_list, all_preds, all_labels, model = train_and_evaluate(best_lr, best_num_hidden_layers, best_hidden_size, best_batch_size, best_activation, best_dropout_rate, best_decay_factor, combined_train_images, combined_train_labels, val_images, val_labels, test_images, test_labels, 'adam', no_epochs)
    torch.save(model.state_dict(), 'best_model.pth')

    return val_loss, val_acc, test_loss, test_acc, train_loss, train_acc, train_list, val_list, train_acc_list, val_acc_list, all_preds, all_labels, model



# Realised that i should have the optimiser outside the optimisation function


def objective_fixed_opt(trial, optimizer_name, combined_train_images, combined_train_labels, val_images, val_labels):
    """
    Objective function for Optuna to optimize, for the 5 main hyperparameters.

    Sets batch size to 64 and optimizer and the activation function to ReLU
    
    Args:
    trial: optuna.trial.Trial, a trial object
    
    Returns:
    val_acc: float, validation accuracy
    """
    if optimizer_name == 'adam' or optimizer_name == 'rmsprop':
        lr = trial.suggest_float('lr', 0.0005, 0.001, log=True)
    elif optimizer_name == 'sgd':
        lr = trial.suggest_float('lr', 0.01, 0.1, log=True)
    num_hidden_layers = trial.suggest_int('num_hidden_layers', 2, 20)
    hidden_size = trial.suggest_int('hidden_size', 500, 1000)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
    decay_factor = trial.suggest_float('decay_factor', 0.2, 0.8)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FullyConnectedNN(
        input_size=1568,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        activation=nn.ReLU(),
        dropout_rate=dropout_rate,
        decay_factor=decay_factor,
        num_classes=20,
    ).to(device)

    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    criterion = nn.CrossEntropyLoss()

    # Data loaders
    train_data_stacked = torch.stack(combined_train_images)
    train_labels_stacked = torch.tensor(combined_train_labels)

    val_data_stacked = torch.stack(val_images)
    val_labels_stacked = torch.tensor(val_labels)

    train_tensor = TensorDataset(train_data_stacked, train_labels_stacked)
    val_tensor = TensorDataset(val_data_stacked, val_labels_stacked)

    train_loader = DataLoader(train_tensor, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_tensor, batch_size=64, shuffle=False)

    best_val_acc = 0.0

    for epoch in range(10):
        _, _ = train_one_epoch(model, train_loader, criterion, optimizer, device)
        _, val_acc, _ ,_ = validate(model, val_loader, criterion, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    return best_val_acc
    


def FCNN_optimised_5_hyp(combined_train_images, combined_train_labels, val_images, val_labels, test_images, test_labels, optimizer, n_trials = 10, no_epochs = 10, optimisation_required = False):
    """
    Function to optimise the main 5 hyperparameters of the Fully Connected Neural Network for a given optimiser

    Sets the batch size to 64 and the activation function to ReLU

    Args:
    n_trials: int, number of trials for the hyperparameter optimisation
    no_epochs: int, number of epochs for training
    optimiser: str, 'adam', 'sgd', or 'rmsprop'
    optimisation_required: bool, whether to perform hyperparameter optimisation

    Returns:
    val_loss: float, validation loss
    val_acc: float, validation accuracy
    test_loss: float, test loss
    test_acc: float, test accuracy
    train_loss: float, train loss
    train_acc: float, train accuracy
    train_loss_list: list of floats, training loss for each epoch
    val_loss_list: list of floats, validation loss for each epoch
    model: nn.Module, trained model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if optimisation_required == True:
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective_fixed_opt(trial, optimizer), n_trials=n_trials)
        # joblib.dump(study, 'study.pkl')

        print(f'Best hyperparameters: {study.best_params}')
        print(f'Best validation accuracy{study.best_value}')

        best_lr = study.best_params['lr']
        best_num_hidden_layers = study.best_params['num_hidden_layers']
        best_hidden_size = study.best_params['hidden_size']
        best_dropout_rate = study.best_params['dropout_rate']
        best_decay_factor = study.best_params['decay_factor']
        best_optimizer = optimizer
    

        model = FullyConnectedNN(
            input_size=1568,
            hidden_size=best_hidden_size,
            num_hidden_layers=best_num_hidden_layers,
            activation=nn.ReLU(),
            dropout_rate=best_dropout_rate,
            decay_factor=best_decay_factor,
            num_classes=20,
        ).to(device)

        if optimizer == 'adam':
            torch.save({
                'model_state_dict': model.state_dict(),
                'hyperparameters': study.best_params,  
                'input_size': 1568  
            }, "best_model_with_params_adam.pth")
        
        elif optimizer == 'sgd':
            torch.save({
                'model_state_dict': model.state_dict(),
                'hyperparameters': study.best_params,  
                'input_size': 1568  
            }, "best_model_with_params_sgd.pth")
        
        elif optimizer == 'rmsprop':
            torch.save({
                'model_state_dict': model.state_dict(),
                'hyperparameters': study.best_params,  
                'input_size': 1568  
            }, "best_model_with_params_rmsprop.pth")
            
    else:
        try:
            checkpoint = torch.load('best_model_with_params.pth')
        except FileNotFoundError:
            print('No checkpoint found. Please run the function with optimisation_required=True first')
        except KeyError:
            print('No hyperparameters found in the checkpoint. Please run the function with optimisation_required=True first')
        
        best_params = checkpoint['hyperparameters']

        best_lr = best_params['lr']
        best_num_hidden_layers = best_params['num_hidden_layers']
        best_hidden_size = best_params['hidden_size']
        best_dropout_rate = best_params['dropout_rate']
        best_decay_factor = best_params['decay_factor']
        best_optimizer = optimizer

        model = FullyConnectedNN(
            input_size=1568,
            hidden_size=best_hidden_size,
            num_hidden_layers=best_num_hidden_layers,
            activation=nn.ReLU(),
            dropout_rate=best_dropout_rate,
            decay_factor=best_decay_factor,
            num_classes=20,
        ).to(device)

    val_loss, val_acc, test_loss, test_acc, train_loss, train_acc, train_list, val_list, train_acc_list, val_acc_list, all_preds, all_labels, model = train_and_evaluate(best_lr, best_num_hidden_layers, best_hidden_size, 64, nn.ReLU(), best_dropout_rate, best_decay_factor, combined_train_images, combined_train_labels, val_images, val_labels, test_images, test_labels, best_optimizer, no_epochs)
 
    return val_loss, val_acc, test_loss, test_acc, train_loss, train_acc, train_list, val_list, train_acc_list, val_acc_list, all_preds, all_labels, model



