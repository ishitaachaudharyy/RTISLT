import torch

from bilstm import ISLBiLSTM


# ---------------------------------------
# Create model
# ---------------------------------------

model = ISLBiLSTM()

print(model)


# ---------------------------------------
# Fake batch
# ---------------------------------------

batch_size = 4
sequence_length = 90
feature_size = 225


x = torch.randn(
    batch_size,
    sequence_length,
    feature_size
)


# ---------------------------------------
# Forward pass
# ---------------------------------------

output = model(x)


print()
print("Input shape :", x.shape)
print("Output shape:", output.shape)