class ConvolutionalCVAE(nn.Module):
    def __init__(self, intermediate_dims, latent_dim, input_shape):
        super().__init__()

        input_dim = np.prod(input_shape)
        print('Shape', input_shape, 'Input dim', input_dim)
        self.encoder = nn.Sequential(
            Flatten(),
            nn.Linear(input_dim, intermediate_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(intermediate_dims[0]),
            nn.Dropout(0.3),
            nn.Linear(intermediate_dims[0], intermediate_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(intermediate_dims[1]),
            nn.Dropout(0.3)
        )

        self.mu_repr = nn.Linear(intermediate_dims[1], latent_dim)
        self.log_sigma_repr = nn.Linear(intermediate_dims[1], latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dims[1]),
            nn.LeakyReLU(),
            nn.BatchNorm1d(intermediate_dims[1]),
            nn.Dropout(0.3),
            nn.Linear(intermediate_dims[1], intermediate_dims[0]),
            nn.LeakyReLU(),
            nn.BatchNorm1d(intermediate_dims[0]),
            nn.Dropout(0.3),
            nn.Linear(intermediate_dims[0], input_dim),
            nn.Sigmoid(),
            RestoreShape(input_shape)
        )

    def _encode(self, x):
        latent_repr = self.encoder(x)
        mu_values = self.mu_repr(latent_repr)
        log_sigma_values = self.log_sigma_repr(latent_repr)
        return mu_values, log_sigma_values, latent_repr

    def _reparametrize(self, sample, mu_values, log_sigma_values):
        return sample * torch.exp(log_sigma_values) + mu_values

    def forward(self, x, raw_sample=None):
        mu_values, log_sigma_values, latent_repr = self._encode(x)

        if raw_sample is None:
            raw_sample = torch.randn_like(mu_values)

        latent_sample = self._reparametrize(raw_sample, mu_values, log_sigma_values)

        reconstructed_repr = self.decoder(latent_sample)

        return reconstructed_repr, latent_sample, mu_values, log_sigma_values