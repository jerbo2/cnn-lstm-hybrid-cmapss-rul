import torch, logging, os, json
from . import test_model

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler("train_model.log")
logger.addHandler(console_handler)
logger.addHandler(file_handler)


# Function for saving best result dicts for future use
def save_best_config(best_config_per_engine, engine_number):
    with open(f"configs/best_config_engine_{engine_number}.json", "w") as f:
        # have to convert from numpy types to python types for json serialization
        for field in best_config_per_engine:
            best_config_per_engine[field] = float(
                best_config_per_engine[field]
            )
        json.dump(best_config_per_engine, f)


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    engine_number,
    engine_test_data,
    engine_test_df,
    engine_test_unit_indices,
    current_config,
    device,
    target_scaler,
    patience=3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_loss = float("inf")
    early_stop_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for sequences, labels in train_loader:
            # Forward pass
            sequences = sequences.float().to(device)
            labels = labels.float().to(device)
            outputs = model(sequences).squeeze()
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.float().to(device)
                labels = labels.float().to(device)
                outputs = model(sequences).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        # Test the model's performance
        test_rmse, _, _, _ = test_model.test_model_performance(
            model,
            engine_test_data,
            engine_test_df,
            engine_test_unit_indices,
            device,
            target_scaler,
        )

        # Check the RMSE value that the previous best model had for this engine in the best_saved_models folder
        try:
            current_state = [
                state
                for state in os.listdir("./best_saved_models")
                if f"engine_{engine_number}" in state
            ][0]
            current_best_rmse = float(
                current_state.split(f"{engine_number}_")[1].split(".pt")[0]
            )
        except Exception as e:
            logger.error(e)
            current_best_rmse = float("inf")

        if test_rmse < current_best_rmse:
            save_best_config(current_config, engine_number)
            logger.info(
                f"New best model found for engine {engine_number} w/ RMSE: {test_rmse}"
            )
            try:
                os.remove(f"./best_saved_models/{current_state}")
            except:
                pass
            torch.save(
                model.state_dict(),
                f"./best_saved_models/best_model_engine_{engine_number}_{round(test_rmse,2)}.pt",
            )

        # Log epoch results
        logger.info(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Validation Loss: {val_loss/len(val_loader):.4f}, Test RMSE: {test_rmse:.4f}",
        )

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                logger.info("Early stopping triggered!")
                break
