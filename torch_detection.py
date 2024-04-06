import torch
from torch import nn
from pathlib import Path
from torchvision import transforms

class CrashCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(
            num_features=3
        )
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=43264,
                out_features=512
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=512,
                out_features=2,
            )
        )

    def forward(self,x):
        x = self.bn(x)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.classifier(x)

        return x
    
class ReturnLoadedCNN():
    class_names = ['Accident', 'Non Accident']
    img_height = 250
    img_width = 250

    def __init__(self, path_to_weights):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = CrashCNN()
        self.model.load_state_dict(torch.load(Path(path_to_weights)))
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(self.img_height,self.img_width), antialias=True),
            transforms.ToTensor()
        ])

    def predict_accident(self, img):

        img = self.transforms(img)

        self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            img = img.unsqueeze(dim=0)
            img_pred_logits = self.model(img.to(self.device)) # Make sure the target image is on the right device

        img_pred_probs = torch.softmax(img_pred_logits,dim=1).max().cpu()
        img_pred_labels = torch.argmax(img_pred_probs, dim=0).cpu()

        return self.class_names[img_pred_labels], img_pred_probs
