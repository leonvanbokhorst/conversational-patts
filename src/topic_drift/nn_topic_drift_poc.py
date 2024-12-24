import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, List
from topic_drift.data_loader import load_from_huggingface
from topic_drift.data_prep import prepare_training_data, DataSplit
from tqdm import tqdm
import torchmetrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Linguistic markers for topic transitions
TRANSITION_MARKERS = [
    "so",           # Topic shift
    "anyway",       # Topic shift
    "by the way",   # Topic introduction
    "speaking of",  # Related topic
    "oh",          # Sudden realization/topic change
    "well",        # Hesitation/transition
    "actually",     # Contradiction/shift
    "but",         # Contrast/shift
    "however",      # Contrast/shift
    "meanwhile",    # Parallel topic
]

class EnhancedTopicDriftDetector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512):
        """Initialize the topic drift detection model.

        Args:
            input_dim: Dimension of a single turn embedding
            hidden_dim: Dimension of hidden layers
        """
        super().__init__()
        self.input_dim = input_dim

        # Embedding processor
        self.embedding_processor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Multi-level attention
        self.local_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softmax(dim=1)
        )
        
        # Global context attention
        self.global_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),  # Better for long-range dependencies
            nn.Linear(hidden_dim // 2, 1),
            nn.Softmax(dim=1)
        )
        
        # Linguistic marker detection
        self.marker_detector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, len(TRANSITION_MARKERS)),
            nn.Sigmoid()
        )

        # Pattern detection
        self.pattern_detector = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            bidirectional=True
        )

        # Final regression layer
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Enhanced forward pass with multi-level attention."""
        batch_size = x.shape[0]
        window_size = 8
        
        # Reshape input
        x = x.view(batch_size, window_size, self.input_dim)
        
        # Process embeddings
        processed = self.embedding_processor(x)
        
        # Multi-level attention
        local_weights = self.local_attention(processed)
        global_weights = self.global_attention(processed)
        
        # Combine attention weights based on pattern detection
        pattern_scores = self.pattern_detector(processed)[0]
        attention_weights = (
            0.7 * local_weights +  # Local context
            0.3 * global_weights   # Global context
        ) * pattern_scores
        
        # Detect semantic bridges
        transitions = self.semantic_bridge_detection(processed)
        
        # Final drift score computation
        context = torch.sum(attention_weights * processed, dim=1)
        base_score = self.regressor(context)
        
        # Adjust score based on transition patterns
        final_score = base_score * (1 + transitions.mean(dim=1, keepdim=True))
        
        return torch.clamp(final_score, 0, 1)

    def semantic_bridge_detection(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Detect semantic bridges between turns."""
        batch_size, seq_len, _ = embeddings.shape
        
        # Pairwise semantic similarity
        similarities = torch.bmm(
            embeddings, embeddings.transpose(1, 2)
        )
        
        # Detect transition patterns
        transitions = torch.zeros(batch_size, seq_len - 1, device=embeddings.device)
        
        # Create transition scores based on similarity thresholds
        high_sim = (similarities[:, range(seq_len-1), range(1, seq_len)] > 0.8).float()
        med_sim = (similarities[:, range(seq_len-1), range(1, seq_len)] > 0.6).float()
        low_sim = (similarities[:, range(seq_len-1), range(1, seq_len)] > 0.4).float()
        
        # Apply transition scores using logical operations on tensors
        transitions = (
            0.11 * high_sim +  # topic_maintenance
            0.15 * (med_sim * (1 - high_sim)) +  # smooth_transition
            0.18 * (low_sim * (1 - med_sim)) +  # topic_shift
            0.21 * (1 - low_sim)  # abrupt_change
        )
            
        return transitions


def train_model(
    data: DataSplit,
    batch_size: int = 32,
    epochs: int = 10,
    learning_rate: float = 0.001,
    early_stopping_patience: int = 3,
) -> Tuple[EnhancedTopicDriftDetector, Dict[str, list]]:
    """Train the topic drift detection model.

    Args:
        data: DataSplit object containing all data splits
        batch_size: Batch size for training
        epochs: Number of epochs to train
        learning_rate: Learning rate for optimizer
        early_stopping_patience: Number of epochs to wait for improvement

    Returns:
        Tuple[EnhancedTopicDriftDetector, dict]: Trained model and training metrics dictionary
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Move data to device
    train_embeddings = data.train_embeddings.to(device)
    train_labels = data.train_labels.to(device)
    val_embeddings = data.val_embeddings.to(device)
    val_labels = data.val_labels.to(device)

    # Create data loaders
    train_dataset = TensorDataset(train_embeddings, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(val_embeddings, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model and training components
    embedding_dim = data.train_embeddings.shape[1] // 8  # Using window_size=8
    model = EnhancedTopicDriftDetector(embedding_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize metrics
    rmse = torchmetrics.MeanSquaredError(squared=False).to(device)
    r2score = torchmetrics.R2Score().to(device)

    # Initialize metrics tracking
    metrics = {
        "train_losses": [],
        "train_rmse": [],
        "train_r2": [],
        "val_losses": [],
        "val_rmse": [],
        "val_r2": [],
    }

    # Early stopping setup
    best_val_rmse = float("inf")
    patience_counter = 0
    best_model_state = None

    # Training loop with tqdm
    epoch_pbar = tqdm(range(epochs), desc="Training Progress")
    for epoch in epoch_pbar:
        # Training phase
        model.train()
        train_loss = 0.0
        rmse.reset()
        r2score.reset()

        batch_pbar = tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1}")
        for batch_x, batch_y in batch_pbar:
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.unsqueeze(1))

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update metrics
            train_loss += loss.item()
            rmse.update(outputs.squeeze(), batch_y)
            r2score.update(outputs.squeeze(), batch_y)
            
            batch_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        # Calculate training metrics
        train_results = {
            "loss": train_loss / len(train_loader),
            "rmse": rmse.compute().item(),
            "r2": r2score.compute().item(),
        }

        # Validation phase
        model.eval()
        val_loss = 0.0
        rmse.reset()
        r2score.reset()

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                val_loss += loss.item()
                
                rmse.update(outputs.squeeze(), batch_y)
                r2score.update(outputs.squeeze(), batch_y)

        # Calculate validation metrics
        val_results = {
            "loss": val_loss / len(val_loader),
            "rmse": rmse.compute().item(),
            "r2": r2score.compute().item(),
        }

        # Update metrics
        metrics["train_losses"].append(train_results["loss"])
        metrics["train_rmse"].append(train_results["rmse"])
        metrics["train_r2"].append(train_results["r2"])
        metrics["val_losses"].append(val_results["loss"])
        metrics["val_rmse"].append(val_results["rmse"])
        metrics["val_r2"].append(val_results["r2"])

        # Early stopping check
        if val_results["rmse"] < best_val_rmse:
            best_val_rmse = val_results["rmse"]
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        # Update progress bar
        epoch_pbar.set_postfix(
            {
                "train_loss": f"{train_results['loss']:.4f}",
                "train_rmse": f"{train_results['rmse']:.4f}",
                "val_loss": f"{val_results['loss']:.4f}",
                "val_rmse": f"{val_results['rmse']:.4f}",
            }
        )

        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, metrics


def evaluate_model(
    model: EnhancedTopicDriftDetector,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on a dataset.

    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    # Initialize metrics
    rmse = torchmetrics.MeanSquaredError(squared=False).to(device)
    r2score = torchmetrics.R2Score().to(device)
    total_loss = 0.0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            total_loss += loss.item()

            rmse.update(outputs.squeeze(), batch_y)
            r2score.update(outputs.squeeze(), batch_y)

    return {
        "loss": total_loss / len(dataloader),
        "rmse": rmse.compute().item(),
        "r2": r2score.compute().item(),
    }


def plot_training_curves(metrics: Dict[str, List[float]], save_path: str = None):
    """Plot training and validation metrics over epochs.
    
    Args:
        metrics: Dictionary containing training history
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(131)
    plt.plot(metrics['train_losses'], label='Train Loss')
    plt.plot(metrics['val_losses'], label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot RMSE
    plt.subplot(132)
    plt.plot(metrics['train_rmse'], label='Train RMSE')
    plt.plot(metrics['val_rmse'], label='Val RMSE')
    plt.title('RMSE Curves')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
    
    # Plot R²
    plt.subplot(133)
    plt.plot(metrics['train_r2'], label='Train R²')
    plt.plot(metrics['val_r2'], label='Val R²')
    plt.title('R² Score Curves')
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def visualize_attention(
    model: EnhancedTopicDriftDetector,
    sample_windows: torch.Tensor,
    sample_texts: List[List[str]],
    sample_scores: torch.Tensor,
    device: torch.device,
    save_path: str = None
):
    """Visualize attention weights for sample windows."""
    model.eval()
    with torch.no_grad():
        batch_size = sample_windows.shape[0]
        window_size = 8
        
        # Get model attention weights and predictions
        x = sample_windows.to(device)
        x = x.view(batch_size, window_size, model.input_dim)
        processed = model.embedding_processor(x)
        
        # Get multi-level attention weights
        local_weights = model.local_attention(processed)
        global_weights = model.global_attention(processed)
        pattern_scores = model.pattern_detector(processed)[0]
        
        # Combine attention weights
        attention_weights = (
            0.7 * local_weights +  # Local context
            0.3 * global_weights   # Global context
        ) * pattern_scores
        
        attention_weights = attention_weights.cpu().numpy()
        predictions = model(sample_windows.to(device)).squeeze().cpu().numpy()
        
        # Plot attention heatmaps
        plt.figure(figsize=(15, 4 * batch_size))
        for i in range(batch_size):
            plt.subplot(batch_size, 1, i + 1)
            
            # Create heatmap
            sns.heatmap(
                attention_weights[i].T,
                cmap='YlOrRd',
                xticklabels=[f"Turn {j+1}\n{sample_texts[i][j][:50]}..." for j in range(window_size)],
                yticklabels=False,
                cbar_kws={'label': 'Attention Weight'}
            )
            
            # Rotate x-labels for better readability
            plt.xticks(rotation=45, ha='right')
            
            # Add title with scores
            plt.title(
                f'Window {i+1} - Predicted Drift: {predictions[i]:.4f}, '
                f'Actual Drift: {sample_scores[i].item():.4f}\n'
                f'Attention Distribution Across Conversation Turns',
                pad=20
            )
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()


def plot_prediction_distribution(
    model: EnhancedTopicDriftDetector,
    test_loader: DataLoader,
    device: torch.device,
    save_path: str = None
):
    """Plot distribution of predictions vs actual values.
    
    Args:
        model: Trained EnhancedTopicDriftDetector model
        test_loader: DataLoader for test data
        device: Device to run model on
        save_path: Optional path to save the plot
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            all_preds.extend(outputs.squeeze().cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    plt.figure(figsize=(15, 5))
    
    # Scatter plot
    plt.subplot(121)
    plt.scatter(all_targets, all_preds, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')  # Perfect prediction line
    plt.xlabel('Actual Drift Score')
    plt.ylabel('Predicted Drift Score')
    plt.title('Predictions vs Actual Values')
    plt.grid(True)
    
    # Error distribution
    plt.subplot(122)
    errors = np.array(all_preds) - np.array(all_targets)
    plt.hist(errors, bins=50, density=True)
    plt.xlabel('Prediction Error')
    plt.ylabel('Density')
    plt.title('Error Distribution')
    plt.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


class TransitionPatternModule(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        # Pattern weights based on our analysis
        self.pattern_weights = {
            "gentle_wave": (0.10, 0.16),    # Low drift range
            "single_peak": (0.13, 0.19),    # Medium drift range
            "ascending_stairs": (0.16, 0.22) # High drift range
        }
        
        # Pattern detection layers
        self.pattern_detector = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            bidirectional=True
        )


def train_with_patterns(
    model: EnhancedTopicDriftDetector,
    data: DataSplit,
    batch_size: int = 32,
    epochs: int = 20,
) -> None:
    """Enhanced training with pattern recognition."""
    # Additional loss components
    pattern_loss = nn.MSELoss()  # For pattern detection
    transition_loss = nn.L1Loss() # For transition smoothness
    
    # Combined loss function
    def combined_loss(pred, target, patterns, transitions):
        base_loss = nn.MSELoss()(pred, target)
        pattern_penalty = pattern_loss(patterns, expected_patterns)
        transition_smoothness = transition_loss(transitions, expected_transitions)
        
        return (
            0.6 * base_loss +          # Base drift prediction
            0.2 * pattern_penalty +     # Pattern adherence
            0.2 * transition_smoothness # Transition smoothness
        )


def main():
    """Load data, prepare it, and train the model."""
    # Load conversation data from Hugging Face
    conversation_data = load_from_huggingface()

    # Prepare training data with splits
    data = prepare_training_data(
        conversation_data,
        window_size=8,  # Use 8 turns for context
        batch_size=32,  # Reduced batch size for larger windows
        max_workers=8,
        force_recompute=True,  # Force recompute since we changed the data format
    )

    # Train model with enhanced metrics
    model, metrics = train_model(
        data,
        batch_size=16,  # Reduced batch size for training due to larger windows
        epochs=20,  # Increased epochs for better convergence
        learning_rate=0.0005,  # Reduced learning rate for stability
        early_stopping_patience=5,  # Increased patience for better convergence
    )

    # Get device
    device = next(model.parameters()).device

    # Plot training curves
    plot_training_curves(metrics)

    # Evaluate on test set
    test_dataset = TensorDataset(data.test_embeddings.to(device), data.test_labels.to(device))
    test_loader = DataLoader(test_dataset, batch_size=16)  # Reduced batch size
    test_results = evaluate_model(model, test_loader, device)

    # Plot prediction distribution
    plot_prediction_distribution(model, test_loader, device)

    # Print final metrics
    print("\nTraining Results:")
    print(f"Best Validation RMSE: {min(metrics['val_rmse']):.4f}")
    print(f"Best Validation R²: {max(metrics['val_r2']):.4f}")
    print("\nTest Set Results:")
    print(f"Loss: {test_results['loss']:.4f}")
    print(f"RMSE: {test_results['rmse']:.4f}")
    print(f"R²: {test_results['r2']:.4f}")

    # Get sample windows with their texts for attention visualization
    print("\nAnalyzing attention patterns on sample conversations...")
    
    # Get a few interesting examples (high drift, low drift, medium drift)
    test_preds = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            test_preds.extend(model(batch_x).squeeze().cpu().numpy())
    test_preds = np.array(test_preds)
    
    # Get indices for interesting examples
    high_drift_idx = np.argmax(data.test_labels.numpy())
    low_drift_idx = np.argmin(data.test_labels.numpy())
    med_drift_idx = np.argmin(np.abs(data.test_labels.numpy() - np.median(data.test_labels.numpy())))
    
    sample_indices = [low_drift_idx, med_drift_idx, high_drift_idx]
    sample_windows = data.test_embeddings[sample_indices]
    sample_scores = data.test_labels[sample_indices]
    
    # Get original texts from conversation data
    sample_texts = []
    for conv in conversation_data.conversations:
        turns = conv["turns"]
        if len(turns) >= 8:  # window_size = 8
            for i in range(len(turns) - 8 + 1):
                window_turns = turns[i : i + 8]
                window_text = [turn[:100] + "..." if len(turn) > 100 else turn for turn in window_turns]
                sample_texts.append(window_text)
    
    # Get texts for our selected windows
    selected_texts = [sample_texts[idx] for idx in sample_indices]
    
    # Print example conversations
    print("\nAnalyzing conversations with different drift levels:")
    for i, (texts, score) in enumerate(zip(selected_texts, sample_scores)):
        drift_level = "Low" if i == 0 else "Medium" if i == 1 else "High"
        print(f"\n{drift_level} Drift (score: {score.item():.4f}):")
        for j, text in enumerate(texts):
            print(f"Turn {j+1}: {text}")
    
    # Visualize attention patterns
    visualize_attention(model, sample_windows, selected_texts, sample_scores, device)


if __name__ == "__main__":
    main()
