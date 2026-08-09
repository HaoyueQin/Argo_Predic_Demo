import torch
import torch.nn as nn
import random

class LSTMTrajectoryModel(nn.Module):
    def __init__(self, input_size: int = 2, hidden_size: int = 256, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.hidden_size = hidden_size

        # Encoder: Encodes the history of relative displacements
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # Decoder: Generates future relative displacements
        self.decoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # Output layer: Predicts (dx, dy)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x, target=None, steps=30, teacher_forcing_ratio=0.5):
        """
        x: History [batch, obs_len, 2]
        target: Future Ground Truth [batch, pred_len, 2] (Optional, for teacher forcing)
        steps: Number of future steps to predict
        teacher_forcing_ratio: Probability of using teacher forcing
        """
        batch_size = x.shape[0]
        device = x.device

        # 1. Preprocess: Calculate velocities (displacements) from history
        if x.shape[1] > 1:
            v = x[:, 1:] - x[:, :-1]
        else:
            v = torch.zeros(batch_size, 1, 2, device=device)

        # 2. Encoder
        _, (h, c) = self.encoder(v)

        # 3. Decoder
        last_vel = v[:, -1:, :]
        last_pos = x[:, -1, :]

        preds = []
        decoder_input = last_vel

        # Prepare target velocities if teacher forcing is used
        if target is not None:
            # target_vels: [batch, steps, 2]
            # First step velocity: target[:, 0] - last_pos
            # Subsequent steps: target[:, i] - target[:, i-1]
            target_vels = torch.zeros(batch_size, steps, 2, device=device)
            target_vels[:, 0, :] = target[:, 0, :] - last_pos
            if steps > 1:
                target_vels[:, 1:, :] = target[:, 1:, :] - target[:, :-1, :]

        for i in range(steps):
            o, (h, c) = self.decoder(decoder_input, (h, c))
            pred_vel = self.fc(o) # [batch, 1, 2]

            # Update position
            current_pos = last_pos + pred_vel.squeeze(1)
            preds.append(current_pos)

            # Determine next input
            if target is not None and random.random() < teacher_forcing_ratio:
                # Use ground truth velocity
                decoder_input = target_vels[:, i:i+1, :]
            else:
                # Use predicted velocity
                decoder_input = pred_vel

            last_pos = current_pos

        return torch.stack(preds, dim=1)
