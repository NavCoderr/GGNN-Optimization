import torch
import torch.nn as nn
import torch.optim as optim
from modules.ggnn_model import GGNN
from modules.graph_conversion import convert_to_pyg_data

def train_model(model, pyg_data, loss_fn, optimizer, epochs=100):
     
    model.train()
    
    if not hasattr(pyg_data, "y") or pyg_data.y is None:
        raise ValueError("The training data does not contain target values (y). Ensure that convert_to_pyg_data() provides y values.")

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(pyg_data.x, pyg_data.edge_index)
        
        
        loss = loss_fn(output, pyg_data.y.view(-1, output.shape[1]))  
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f" Epoch {epoch}: Loss {loss.item():.4f}")

    print(" Training completed. Running validation...")
    validate_model(model, pyg_data)

def validate_model(model, pyg_data):
     
    model.eval()
    with torch.no_grad():
        predictions = model(pyg_data.x, pyg_data.edge_index)

    if predictions.nelement() == 0:
        print("Validation Warning: No predictions generated.")
        return
    
    print(f" GGNN Validation: Predictions Sample -> {predictions[:5]}")
    return model
