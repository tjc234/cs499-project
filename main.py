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

# handle random seed generation
def set_seed():
    pass

# select device work is being done on
def get_device():
    return "cpu"

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
            self.conv1 = nn.Conv2d( 1, 16, 3, padding = 1 )
            self.conv2 = nn.Conv2d( 16, 32, 3, padding = 1 )
            self.fc1 = nn.Linear( 32 * 7 * 7, 128 )
            self.fc2 = nn.Linear( 128, 9 )  # MedMNIST has 9 tissue classes

        def forward(self, x):
            x = F.relu( self.conv1( x ) )
            x = F.max_pool2d (x, 2 )
            x = F.relu( self.conv2( x ))
            x = F.max_pool2d( x, 2 )
            x = x.view(x.size( 0 ), -1 )
            x = F.relu(self.fc1( x ) )
            return self.fc2( x )

    return SimpleCNN()

# resnet18
def build_resnet18( pretrained = False ):
    return None

# resnet50
def build_resnet50( pretrained = False ):
    return None

# resnet101
def build_resnet101( pretrained = False ):
    return None


# =========================
# 4. Training Utilities
# =========================

# handle loss data
def get_loss_function():
    return None

# handle optimization
def get_optimizer( model ):
    return None

# handle accuracy
def compute_accuracy( outputs, labels ):
    return 0.0


# =========================
# 5. Train Model Function
# =========================

# handle epoch
def run_one_epoch( model, loader, optimizer = None ):
    return 0.0, 0.0

# handle training
def train_model( model, train_loader, val_loader ):
    return model, {}


# =========================
# 6. Evaluate Function
# =========================

# evaluate given model based on test data
def evaluate_model( model, test_loader ):
    return {}


# =========================
# 7. Run Experiments
# =========================

# handle experiments
def run_experiments():
    return {}


# =========================
# 8. Compare Results
# =========================

# compare metrics
def compare_results(results):
    pass



# =========================
# Main
# =========================

def main():
    results = run_experiments()
    compare_results(results)





if __name__ == "__main__":
    main()