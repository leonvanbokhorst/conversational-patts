"""Module for preparing conversation data for training."""

import torch
import numpy as np
from typing import Tuple, List, Dict, Optional, NamedTuple
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer

from topic_drift.data_types import ConversationData


class DataSplit(NamedTuple):
    """Container for train/val/test split tensors."""

    train_embeddings: torch.Tensor
    train_labels: torch.Tensor
    val_embeddings: torch.Tensor
    val_labels: torch.Tensor
    test_embeddings: torch.Tensor
    test_labels: torch.Tensor


def split_data(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> DataSplit:
    """Split data into train, validation, and test sets.

    Args:
        embeddings: Full embeddings tensor
        labels: Full labels tensor
        val_size: Fraction of data to use for validation
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        DataSplit object containing train/val/test tensors
    """
    # Convert labels to discrete bins for stratification
    n_bins = 10  # Number of bins for stratification
    binned_labels = np.floor(labels.numpy() * n_bins).astype(int)
    
    # Count samples in each bin
    unique_bins, bin_counts = np.unique(binned_labels, return_counts=True)
    min_samples = np.min(bin_counts)
    
    # Determine if stratification is possible
    use_stratify = min_samples >= 2
    stratify = binned_labels if use_stratify else None
    
    # First split off test set
    train_val_emb, test_emb, train_val_labels, test_labels = train_test_split(
        embeddings.numpy(),
        labels.numpy(),
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    # Then split remaining data into train and validation
    if use_stratify:
        # Recompute bins for the remaining data
        binned_train_val = np.floor(train_val_labels * n_bins).astype(int)
        stratify_remaining = binned_train_val
    else:
        stratify_remaining = None

    train_emb, val_emb, train_labels, val_labels = train_test_split(
        train_val_emb,
        train_val_labels,
        test_size=val_size / (1 - test_size),  # Adjust for remaining data
        random_state=random_state,
        stratify=stratify_remaining,
    )

    return DataSplit(
        train_embeddings=torch.from_numpy(train_emb).float(),
        train_labels=torch.from_numpy(train_labels).float(),
        val_embeddings=torch.from_numpy(val_emb).float(),
        val_labels=torch.from_numpy(val_labels).float(),
        test_embeddings=torch.from_numpy(test_emb).float(),
        test_labels=torch.from_numpy(test_labels).float(),
    )


def save_to_cache(
    data_split: DataSplit,
    cache_key: str,
) -> None:
    """Save prepared data splits to cache.

    Args:
        data_split: DataSplit object containing all splits
        cache_key: Cache key for the data
    """
    cache_dir = get_cache_path()
    np.savez(
        cache_dir / f"{cache_key}.npz",
        train_embeddings=data_split.train_embeddings.numpy(),
        train_labels=data_split.train_labels.numpy(),
        val_embeddings=data_split.val_embeddings.numpy(),
        val_labels=data_split.val_labels.numpy(),
        test_embeddings=data_split.test_embeddings.numpy(),
        test_labels=data_split.test_labels.numpy(),
    )


def load_from_cache(
    cache_key: str,
) -> Optional[DataSplit]:
    """Load prepared data splits from cache.

    Args:
        cache_key: Cache key for the data

    Returns:
        DataSplit object if cache exists, None otherwise
    """
    cache_path = get_cache_path() / f"{cache_key}.npz"
    if not cache_path.exists():
        return None

    print(f"Loading prepared data from cache: {cache_path}")
    data = np.load(cache_path)
    return DataSplit(
        train_embeddings=torch.from_numpy(data["train_embeddings"]).float(),
        train_labels=torch.from_numpy(data["train_labels"]).float(),
        val_embeddings=torch.from_numpy(data["val_embeddings"]).float(),
        val_labels=torch.from_numpy(data["val_labels"]).float(),
        test_embeddings=torch.from_numpy(data["test_embeddings"]).float(),
        test_labels=torch.from_numpy(data["test_labels"]).float(),
    )


@dataclass
class TurnWindow:
    """Container for a window of turns and their embeddings."""

    turns: List[str]  # List of consecutive turns in the window
    embeddings: List[np.ndarray] = None  # Embeddings for each turn
    drift_score: float = None  # Continuous drift score between 0 and 1
    window_similarity: float = None  # Average similarity within window
    original_texts: List[str] = None  # Original text for each turn


def get_cache_path() -> Path:
    """Get the path to the cache directory."""
    cache_dir = Path.home() / ".cache" / "topic_drift" / "embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cache_key(
    conversation_data: ConversationData,
    window_size: int,
) -> str:
    """Generate a cache key for the prepared data.

    Args:
        conversation_data: The conversation data
        window_size: Size of the sliding window

    Returns:
        A unique cache key based on the data content and parameters
    """
    data_str = json.dumps(
        [conv["turns"] for conv in conversation_data.conversations], sort_keys=True
    )
    param_str = f"window_{window_size}"
    return hashlib.md5(f"{data_str}{param_str}".encode()).hexdigest()


async def process_batch(
    windows: List[TurnWindow],
    model: SentenceTransformer,
    executor: ThreadPoolExecutor,
) -> List[TurnWindow]:
    """Process a batch of windows in parallel.

    Args:
        windows: List of TurnWindow objects
        model: SentenceTransformer model
        executor: ThreadPoolExecutor for parallel processing

    Returns:
        List of processed TurnWindow objects
    """
    loop = asyncio.get_event_loop()

    # Flatten all turns for batch processing
    all_turns = [turn for window in windows for turn in window.turns]
    
    # Get embeddings for all turns in one batch
    async def get_batch_embeddings(texts: List[str]) -> np.ndarray:
        # Wrap encode call with kwargs in a lambda
        return await loop.run_in_executor(
            executor,
            lambda: model.encode(texts, show_progress_bar=False)
        )
    
    all_embeddings = await get_batch_embeddings(all_turns)
    
    # Distribute embeddings back to windows
    window_size = len(windows[0].turns)
    for i, window in enumerate(windows):
        start_idx = i * window_size
        window.embeddings = list(all_embeddings[start_idx:start_idx + window_size])
        
        # Calculate similarities efficiently using vectorized operations
        embeddings_array = np.stack(window.embeddings)
        similarities = cosine_similarity(embeddings_array)
        
        # Get upper triangle indices for unique pairs
        upper_tri_idx = np.triu_indices(len(window.embeddings), k=1)
        similarities = similarities[upper_tri_idx]
        
        # Calculate window metrics
        window.window_similarity = np.mean(similarities)
        window.drift_score = 1 - window.window_similarity
    
    return windows


def prepare_windows(
    conversation_data: ConversationData,
    window_size: int = 3,
) -> List[TurnWindow]:
    """Prepare sliding windows from conversation data.

    Args:
        conversation_data: ConversationData object
        window_size: Size of each sliding window

    Returns:
        List of TurnWindow objects
    """
    windows = []
    for conv in conversation_data.conversations:
        turns = conv["turns"]
        if len(turns) >= window_size:
            # Create sliding windows
            for i in range(len(turns) - window_size + 1):
                window_turns = turns[i : i + window_size]
                windows.append(TurnWindow(
                    turns=window_turns,
                    original_texts=window_turns.copy()
                ))
    return windows


async def prepare_training_data_async(
    conversation_data: ConversationData,
    window_size: int = 8,
    batch_size: int = 128,  # Increased batch size
    max_workers: int = 16,  # Increased worker count
    use_cache: bool = True,
    force_recompute: bool = False,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    model_name: str = "BAAI/bge-small-en-v1.5",
) -> DataSplit:
    """Prepare training data asynchronously using sliding windows.

    Args:
        conversation_data: ConversationData object containing conversations
        window_size: Size of the sliding window (default: 8)
        batch_size: Number of windows to process in parallel
        max_workers: Maximum number of worker threads
        use_cache: Whether to use cached data
        force_recompute: Whether to force recomputation even if cache exists
        val_size: Fraction of data to use for validation
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        model_name: Name of the sentence-transformer model to use

    Returns:
        DataSplit object containing train/val/test tensors
    """
    # Check cache first
    if use_cache and not force_recompute:
        cache_key = get_cache_key(conversation_data, window_size)
        cached_data = load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

    # Initialize sentence transformer model
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare all windows
    windows = prepare_windows(conversation_data, window_size)
    print(f"Created {len(windows)} windows of size {window_size}")

    # Process in larger batches
    processed_windows = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in tqdm(range(0, len(windows), batch_size), desc="Processing batches"):
            batch = windows[i : i + batch_size]
            processed_batch = await process_batch(batch, model, executor)
            processed_windows.extend(processed_batch)

    # Preallocate arrays for better memory efficiency
    n_windows = len(processed_windows)
    embedding_dim = len(processed_windows[0].embeddings[0])
    total_dim = embedding_dim * window_size
    
    # Create arrays directly
    window_embeddings_array = np.empty((n_windows, total_dim), dtype=np.float32)
    drift_scores_array = np.empty(n_windows, dtype=np.float32)
    
    # Fill arrays efficiently
    for i, window in enumerate(processed_windows):
        window_embeddings_array[i] = np.concatenate(window.embeddings)
        drift_scores_array[i] = window.drift_score
    
    # Convert to tensors in one go
    window_embeddings = torch.from_numpy(window_embeddings_array)
    drift_scores = torch.from_numpy(drift_scores_array)

    # Split data
    data_split = split_data(
        window_embeddings,
        drift_scores,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
    )

    # Save to cache if enabled
    if use_cache:
        cache_key = get_cache_key(conversation_data, window_size)
        print(f"Saving prepared data to cache: {get_cache_path() / f'{cache_key}.npz'}")
        save_to_cache(data_split, cache_key)

    return data_split


def prepare_training_data(
    conversation_data: ConversationData,
    window_size: int = 8,
    batch_size: int = 128,  # Increased default batch size
    max_workers: int = 16,  # Increased default worker count
    use_cache: bool = True,
    force_recompute: bool = False,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    model_name: str = "BAAI/bge-m3",
) -> DataSplit:
    """Synchronous wrapper for async data preparation.

    Args:
        conversation_data: ConversationData object containing conversations
        window_size: Size of the sliding window (default: 8)
        batch_size: Number of windows to process in parallel
        max_workers: Maximum number of worker threads
        use_cache: Whether to use cached data
        force_recompute: Whether to force recomputation even if cache exists
        val_size: Fraction of data to use for validation
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        model_name: Name of the sentence-transformer model to use

    Returns:
        DataSplit object containing train/val/test tensors
    """
    return asyncio.run(
        prepare_training_data_async(
            conversation_data,
            window_size,
            batch_size,
            max_workers,
            use_cache,
            force_recompute,
            val_size,
            test_size,
            random_state,
            model_name,
        )
    )
