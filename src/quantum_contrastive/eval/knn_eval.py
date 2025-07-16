import torch
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def knn_evaluate(encoder, train_loader, test_loader, device, k=5):
    encoder.eval()
    with torch.no_grad():
        # Get embeddings + labels for training data
        train_features, train_labels = [], []
        for x, labels in tqdm(train_loader, desc="Encoding train set"):
            x = x.to(device)
            feats = encoder(x)
            feats = F.adaptive_avg_pool2d(feats, (1, 1)).squeeze()  # (N, D)
            train_features.append(feats.cpu())
            train_labels.append(labels)

        X_train = torch.cat(train_features).numpy()
        y_train = torch.cat(train_labels).numpy()

        # Get embeddings + labels for test data
        test_features, test_labels = [], []
        for x, labels in tqdm(test_loader, desc="Encoding test set"):
            x = x.to(device)
            feats = encoder(x)
            feats = F.adaptive_avg_pool2d(feats, (1, 1)).squeeze()  # (N, D)
            test_features.append(feats.cpu())
            test_labels.append(labels)

        X_test = torch.cat(test_features).numpy()
        y_test = torch.cat(test_labels).numpy()

    # Fit and evaluate kNN
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"KNN Accuracy (k={k}): {acc:.4f}")
    return acc
