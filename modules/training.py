import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from modules.graph_conversion import convert_to_pyg_data
from modules.ggnn_model import GGNN


def train_ggnn(
    model: GGNN,
    pyg_data,
    warehouse_graph,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    scheduler_step: int = 300,
    scheduler_gamma: float = 0.5,
    grad_clip: float = 2.0,
    device: str = 'cpu'
) -> GGNN:
    """
    Train the GGNN model on pyg_data with specified hyperparameters.

    Returns the trained model.
    """
    model.to(device)
    model.train()

    # Convert data to device
    pyg_data = pyg_data.to(device)
    if not hasattr(pyg_data, 'y') or pyg_data.y is None:
        raise ValueError("Training data must include target tensor 'y'.")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    loss_fn = nn.MSELoss()
    best_loss = float('inf')
    patience = 10
    no_improv_epochs = 0

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        output = model(pyg_data.x, pyg_data.edge_index)
        target = pyg_data.y.view(-1, output.size(1))
        loss = loss_fn(output, target)
        loss.backward()
        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()

        # Early stopping check
        if loss.item() < best_loss:
            best_loss = loss.item()
            no_improv_epochs = 0
        else:
            no_improv_epochs += 1

        if epoch % 100 == 0 or no_improv_epochs >= patience:
            print(f"Epoch {epoch}/{epochs}  Loss: {loss.item():.4f}  (best: {best_loss:.4f})")

        if no_improv_epochs >= patience:
            print(f"Early stopping after {epoch} epochs.")
            break

    print("Training completed. Best loss:", best_loss)
    return model


def validate_ggnn(model: GGNN, pyg_data) -> torch.Tensor:
    """
    Validate the GGNN model by performing a forward pass on pyg_data.

    Returns the output predictions tensor.
    """
    model.eval()
    with torch.no_grad():
        output = model(pyg_data.x, pyg_data.edge_index)
    if output.nelement() == 0:
        print("Validation Warning: No predictions generated.")
    else:
        print("Validation: sample predictions ->", output[:5])
    return output


# Example integration
if __name__ == '__main__':
    # Build synthetic graph and convert to PyG data
    G = create_warehouse_graph(seed=42)
    pyg_data = convert_to_pyg_data(G)

    # Initialize & train GGNN
    model = GGNN(in_dim=pyg_data.x.size(1), hidden_dim=128, num_layers=4)
    trained = train_ggnn(
        model,
        pyg_data,
        G,
        epochs=100,
        learning_rate=1e-3,
        weight_decay=1e-5,
        scheduler_step=300,
        scheduler_gamma=0.5,
        grad_clip=2.0,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Validate
    validate_ggnn(trained, pyg_data)
