import requests
import pandas as pd

# Set the base URL for ChEMBL API
base_url = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"

# Set the parameters for fetching the data (10,000 molecules in chunks)
params = {
    'format': 'json',  # Ensure data is returned in JSON format
    'limit': 100,  # Number of results per query
    'offset': 0,  # Starting point for results
}

# Set the number of molecules to fetch
num_molecules = 10000
molecules_list = []

# Loop through pages until we've fetched the desired number of molecules
while len(molecules_list) < num_molecules:
    # Make the API request
    response = requests.get(base_url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()

        # Check if the results are available
        if 'molecules' in data:
            molecules_list.extend(data['molecules'])
        else:
            print("No more molecules available.")
            break
    else:
        print(f"Error: {response.status_code}")
        break

    # Increment the offset for the next batch of data
    params['offset'] += params['limit']

    # Check if we have enough molecules
    if len(molecules_list) >= num_molecules:
        break

# Now you have a list of molecules with SMILES strings
print(f"Retrieved {len(molecules_list)} molecules")

# Convert to DataFrame for easy handling
df = pd.DataFrame(molecules_list)

# Show the first few rows of the DataFrame
df.head()

!pip install torch torchvision rdkit pandas scikit-learn
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load data from the provided DataFrame
smiles_data = df['molecule_structures'].apply(lambda x: x['canonical_smiles'] if x else None).dropna()

# Tokenize SMILES strings
def tokenize_smiles(smiles):
    tokens = []
    i = 0
    while i < len(smiles):
        if i < len(smiles) - 1 and smiles[i:i+2] in ('Cl', 'Br'):  # Handle two-character tokens
            tokens.append(smiles[i:i+2])
            i += 2
        else:
            tokens.append(smiles[i])
            i += 1
    return tokens

# Create a vocabulary from SMILES
all_tokens = [tokenize_smiles(sm) for sm in smiles_data]
vocab = sorted(set(token for tokens in all_tokens for token in tokens))
token_to_idx = {token: i for i, token in enumerate(vocab)}
idx_to_token = {i: token for token, i in token_to_idx.items()}

# Convert SMILES to numerical sequences
def smiles_to_seq(smiles):
    tokens = tokenize_smiles(smiles)
    return [token_to_idx[token] for token in tokens]

smiles_sequences = [smiles_to_seq(sm) for sm in smiles_data]
max_length = max(len(seq) for seq in smiles_sequences)

# Pad sequences to the same length
def pad_sequence(seq, max_length, pad_value=0):
    return seq + [pad_value] * (max_length - len(seq))

padded_sequences = [pad_sequence(seq, max_length) for seq in smiles_sequences]

# Prepare training and testing data
X = np.array(padded_sequences)
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
import torch
import torch.nn as nn
import torch.optim as optim

# VAE Architecture
class VAE(nn.Module):
    def __init__(self, vocab_size, embedding_dim, latent_dim, hidden_dim):
        super(VAE, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def encode(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.encoder(embedded)
        hidden = hidden.squeeze(0)
        mu = self.mu(hidden)
        log_var = self.log_var(hidden)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_len):
        z = z.unsqueeze(1).repeat(1, seq_len, 1)
        outputs, _ = self.decoder(z)
        return self.output_layer(outputs)

    def forward(self, x, seq_len):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        decoded = self.decode(z, seq_len)
        return decoded, mu, log_var
# Hyperparameters
embedding_dim = 50
hidden_dim = 128
latent_dim = 64
vocab_size = len(vocab)
seq_len = max_length
batch_size = 64
epochs = 20
learning_rate = 1e-3

# Prepare DataLoader
train_data = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.long))
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Initialize the VAE model
vae = VAE(vocab_size, embedding_dim, latent_dim, hidden_dim)
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# KL divergence loss
def kl_divergence_loss(mu, log_var):
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

# Training Loop
vae.train()
for epoch in range(epochs):
    total_loss = 0
    for batch in train_loader:
        x = batch[0]
        optimizer.zero_grad()
        output, mu, log_var = vae(x, seq_len)
        recon_loss = loss_fn(output.view(-1, vocab_size), x.view(-1))
        kl_loss = kl_divergence_loss(mu, log_var)
        loss = recon_loss + kl_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")
vae.eval()
with torch.no_grad():
    z = torch.randn(1, latent_dim)
    generated = vae.decode(z, seq_len)
    generated_tokens = torch.argmax(generated, dim=-1).squeeze().tolist()
    generated_smiles = ''.join([idx_to_token[idx] for idx in generated_tokens if idx in idx_to_token])
    print(f"Generated SMILES: {generated_smiles}")
    import torch

# Function to save the trained model
def save_vae_model(model, filepath):
    """
    Save the trained VAE model to a file.
    """
    torch.save(model.state_dict(), filepath)
    print(f"VAE model saved to {filepath}")

# Example usage:
# Save the VAE model to the file "vae_model.pth"
save_vae_model(vae, "vae_model.pth")
