import torch
from torch import nn


class Reorder(nn.Module):
    def forward(self, input):
        return input.permute((0, 2, 1))


class ThreeInputsNet(nn.Module):
    def __init__(self, n_tokens, n_cat_features, concat_number_of_features, hid_size=64):
        super(ThreeInputsNet, self).__init__()
        self.title_encoder = nn.Sequential(
            nn.Embedding(
                n_tokens,
                embedding_dim=hid_size
            ),
            Reorder(),
            nn.Conv1d(
                in_channels=hid_size,
                out_channels=(hid_size * 2),
                kernel_size=2
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(
                output_size=4
            )
        )
        self.desc_encoder = self.title_encoder
        self.cat_encoder = nn.Sequential(
            nn.Linear(
                in_features=n_cat_features,
                out_features=(n_cat_features // 2)
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=(n_cat_features // 2),
                out_features=hid_size
            )
        )

        self.out_layers = nn.Sequential(
            nn.Linear(
                in_features=concat_number_of_features,
                out_features=(hid_size // 2)
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=(hid_size // 2),
                out_features=1
            )
        )

    def forward(self, whole_input):
        input1, input2, input3 = whole_input

        title = self.title_encoder(input1)
        desc = self.desc_encoder(input2)
        category = self.cat_encoder(input3)

        concatenated = torch.cat(
            (
                title.view(title.size(0), -1),
                desc.view(desc.size(0), -1),
                category.view(category.size(0), -1)
            ),
            dim=1
        )

        out = self.out_layers(concatenated)

        return out
