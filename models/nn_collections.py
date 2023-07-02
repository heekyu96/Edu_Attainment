import torch
import torch.nn as nn

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device('cpu')

class Net(nn.Module):
    # Constructor
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        # hidden layer
        self.linear1 = nn.Linear(D_in, H).to(device)
        self.dp1 = nn.Dropout(0.2).to(device)
        self.linear2 = nn.Linear(H, 64).to(device)
        self.linear3 = nn.Linear(64, D_out).to(device)

    # Prediction
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.dp1(x)
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)

        return x


# class TF_EC_Net(nn.Module):
#     # Constructor
#     def __init__(self, D_in, H, D_out):
#         super(TF_EC_Net, self).__init__()
#         # hidden layer
#         self.linear1 = nn.Linear(D_in, H)
#         self.tf_ec = nn.TransformerEncoderLayer(d_model=H, nhead=8,batch_first=True)
#         self.linear2 = nn.Linear(1024, 512)
#         self.dp1 = nn.Dropout(0.2)
#         self.linear3 = nn.Linear(512, 64)
#         self.linear4 = nn.Linear(64, D_out)

#     # Prediction
#     def forward(self, x):
#         x = torch.relu(self.linear1(x))
#         x = self.tf_ec(x)
#         x = self.dp1(x)
#         x = torch.relu(self.linear2(x))
#         x = torch.relu(self.linear3(x))
#         x = self.linear4(x)

#         return x

# class TF_Net(nn.Module):
#     """
#     Text classifier based on a pytorch TransformerEncoder.
#     """

#     def __init__(
#         self,
#         D_in = 66,
#         # embeddings,
#         d_model = 512,
#         nhead=8,
#         dim_feedforward=2048,
#         num_layers=6,
#         dropout=0.1,
#         activation="relu",
#         classifier_dropout=0.1,
#     ):

#         super().__init__()
#         self.linear1 = nn.Linear(D_in, d_model)
#         # vocab_size, d_model = embeddings.size()
#         assert d_model % nhead == 0, "nheads must divide evenly into d_model"

#         # self.emb = nn.Embedding.from_pretrained(embeddings, freeze=False)

#         # self.pos_encoder = PositionalEncoding(
#         #     d_model=d_model,
#         #     dropout=dropout,
#         #     vocab_size=vocab_size,
#         # )

#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#         )
#         self.transformer_encoder = nn.TransformerEncoder(
#             encoder_layer,
#             num_layers=num_layers,
#         )
#         self.classifier = nn.Linear(d_model, 2)
#         self.d_model = d_model

#     def forward(self, x):
#         x = self.linear1(x)
#         # x = self.pos_encoder(x)
#         x = self.transformer_encoder(x)
#         x = x.mean(dim=1)
#         x = self.classifier(x)

#         return x