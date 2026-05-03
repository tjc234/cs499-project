# =========================
# 1. Setup / Configuration
# =========================

# import dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
from medmnist import PathMNIST
from torchvision import transforms, models
from torch.utils.data import DataLoader

# select device work is being done on
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

# handle getting configuration
def get_config():
    return {}


# =========================
# 2. Dataset + Loaders
# =========================

# handle loading dataset
def load_dataset():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train = PathMNIST( split = "train", transform = transform, download = True)
    val = PathMNIST( split = "val", transform = transform, download = True)
    test = PathMNIST( split = "test", transform = transform, download = True)

    return train, val, test

# create loaders for each segment of data
def create_loaders( train_data, val_data, test_data ):
    train_loader = DataLoader( train_data, batch_size = 64, shuffle = True )
    val_loader = DataLoader( val_data, batch_size = 64, shuffle = False )
    test_loader = DataLoader( test_data, batch_size = 64, shuffle = False )

    return train_loader, val_loader, test_loader


# =========================
# 3. Model Definitions
# =========================

# simple CNN
def build_simple_cnn():

    class SimpleCNN( nn.Module ):
        def __init__( self ):
            super().__init__()
            #  first conv layer, low level features
            self.conv1 = nn.Conv2d( 1, 16, 3, padding = 1 )

            # second conv layer, more complex features
            self.conv2 = nn.Conv2d( 16, 32, 3, padding = 1 ) 

            # fully connected layer
            self.fc1 = nn.Linear( 32 * 7 * 7, 128 )

            # output layer (9 tissue classes)
            self.fc2 = nn.Linear( 128, 9 ) 

        def forward(self, x):

            x = F.relu( self.conv1( x ) ) # conv + relu activation
            x = F.max_pool2d ( x, 2 ) # downsample dimensions while retaining features

            # second feature extraction
            x = F.relu( self.conv2( x )) 
            x = F.max_pool2d( x, 2 )

            x = x.view(x.size( 0 ), -1 ) # flatten feature maps into vector for fully connected layer
            x = F.relu(self.fc1( x ) ) # dense feature learning

            # return logits for each class
            return self.fc2( x )

    return SimpleCNN()

# resnet18
def build_resnet18( pretrained = False ):

    # load resnet18 from torchvision
    model = models.resnet18( pretrained = pretrained )

    # replace final classification layer with 9 tissue class output
    model.fc = nn.Linear( model.fc.in_features, 9 )

    return model

# resnet50
def build_resnet50( pretrained = False ):

    # load resnet50 from torchvision
    model = models.resnet50( pretrained = pretrained )

    # replace final classification layer with 9 tissue class output
    model.fc = nn.Linear( model.fc.in_features, 9 )
    
    return model

# resnet101
def build_resnet101( pretrained = False ):

    # load resnet101 from torchvision
    model = models.resnet101( pretrained = pretrained )

    # replace final classification layer with 9 tissue class output
    model.fc = nn.Linear( model.fc.in_features, 9 )
    
    return model


# =========================
# 4. Training Utilities
# =========================

# handle loss data
def get_loss_function():
    return nn.CrossEntropyLoss()

# handle optimization
def get_optimizer( model ):
    return torch.optim.Adam( model.parameters(), lr = 1e-3 )

# handle accuracy
def compute_accuracy( outputs, labels ):
    preds = torch.argmax(outputs, dim=1)
    return ( preds == labels ).float().mean().item()


# =========================
# 5. Train Model Function
# =========================

# handle epoch
def run_one_epoch( model, loader, optimizer = None ):

    # select computation device
    device = get_device()
    model.to( device )

    # loss functionality for 9 tissue class classification
    loss_fn = get_loss_function()

    total_loss = 0
    total_acc = 0

    # if no optimizer is given then default to train mode, otherwise eval mode
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    # iterate over images and labels
    for x, y in loader:

        # move data to computation device
        x = x.to(device)

        # remove label dimensions and convert integer labels
        y = y.squeeze().long().to(device)

        # forward pass
        outputs = model(x)

        # compute loss
        loss = loss_fn(outputs, y)


        # backporpagation and paramter update (for training only)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # counter for batch loss
        total_loss += loss.item()

        # compute batch accuracy
        total_acc += compute_accuracy( outputs, y )

    # return average loss and accuracy acrosss the full dataset
    return total_loss / len( loader ), total_acc / len( loader )

# handle training
def train_model( model, train_loader, val_loader ):

    config = get_config()
    device = get_device()

    model.to( device )

    optimizer = get_optimizer( model )

    num_epochs = 5

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = 0

    # iterate over epochs
    for epoch in range( num_epochs ):

        # training phase
        train_loss, train_acc = run_one_epoch( model, train_loader, optimizer )

        # validation phase
        with torch.no_grad():
            val_loss, val_acc = run_one_epoch( model, val_loader )

        # store metrics
        history[ "train_loss" ].append( train_loss )
        history[ "train_acc" ].append( train_acc )
        history[ "val_loss" ].append( val_loss )
        history[ "val_acc" ].append( val_acc )

        # track best model performance
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        # print progress
        print( f"Epoch {epoch+1}/{num_epochs}" )
        print( f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}" )
        print( f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}" )
        print( "-" * 40 )

    return model, history


# =========================
# 6. Evaluate Function
# =========================

# evaluate given model based on test data
def evaluate_model( model, test_loader ):

    device = get_device()
    model.to( device )

    # evaluation mode
    model.eval()

    total_loss = 0
    total_acc = 0

    loss_fn = get_loss_function()

    # disable gradient computation
    with torch.no_grad():

        for x, y in test_loader:

            x = x.to( device )
            y = y.squeeze().long().to( device )

            outputs = model(x)

            loss = loss_fn( outputs, y )

            total_loss += loss.item()
            total_acc += compute_accuracy( outputs, y )

    return {
        "test_loss": total_loss / len( test_loader ),
        "test_acc": total_acc / len( test_loader )
    }


# =========================
# 7. Run Experiments
# =========================

# handle experiments
def run_experiments():

    # load dataset
    train_data, val_data, test_data = load_dataset()

    # create loaders
    train_loader, val_loader, test_loader = create_loaders(
        train_data, val_data, test_data
    )

    results = {}

    # dictionary of models to test
    model_builders = {
        "SimpleCNN": build_simple_cnn,
        "ResNet18": build_resnet18,
        "ResNet50": build_resnet50
    }

    # iterate through models
    for name, builder in model_builders.items():

        print( f"\nRunning experiment: {name}" )
        print( "=" * 50 )

        model = builder()

        # train model
        model, history = train_model( model, train_loader, val_loader )

        # evaluate model
        test_metrics = evaluate_model( model, test_loader )

        # store results
        results[name] = {
            "history": history,
            "test_metrics": test_metrics
        }

    return results


# =========================
# 8. Compare Results
# =========================

# compare metrics
def compare_results( results ):
    
    print( "\nFinal Model Comparison" )
    print( "=" * 50 )

    for name, data in results.items():

        test_loss = data[ "test_metrics" ][ "test_loss" ]
        test_acc = data[ "test_metrics" ][ "test_acc" ]

        print( f"{name}" )
        print( f"Test Loss: {test_loss:.4f}" )
        print( f"Test Acc:  {test_acc:.4f}" )
        print( "-" * 40 )



# =========================
# Main
# =========================

def main():
    results = run_experiments()
    compare_results(results)





if __name__ == "__main__":
    main()