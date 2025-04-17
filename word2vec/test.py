import torch
from datasets import load_dataset
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load the regressor model
regressor_model = Regressor()
regressor_model.load_state_dict(torch.load('regressor_model.pth'))
regressor_model.eval()  # Set to evaluation mode

# Load test data (example from Hugging Face)
test_dataset = load_dataset("loredanagaspar/hn_title_modeling_dataset", split='test')

# Assuming 'title' and 'score' columns in the test set
test_titles = test_dataset['title']
test_scores = test_dataset['score']

# Tokenize test titles
tokenized_test_titles = [tokenize_title(title, word2id) for title in test_titles]

# Create a dataset for testing
test_dataset = RegressorDataset(tokenized_test_titles, test_scores, embeddings)
test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False)

# Initialize lists to hold predictions and true scores
predictions = []
true_scores = []

# Run the model on the test data
with torch.no_grad():  # Disable gradient tracking during inference
    for embeddings, scores in test_dataloader:
        embeddings = embeddings.to(device)
        true_scores.extend(scores.numpy())  # Add true scores to the list

        # Get predicted scores from the model
        predicted_scores = regressor_model(embeddings).squeeze().cpu().numpy()  # Remove extra dimension
        predictions.extend(predicted_scores)  # Add predicted scores to the list

# Calculate MSE
mse = mean_squared_error(true_scores, predictions)
print(f'Mean Squared Error (MSE): {mse:.4f}')

# Calculate R² score
r2 = r2_score(true_scores, predictions)
print(f'R² Score: {r2:.4f}')

# Calculate MAE
mae = mean_absolute_error(true_scores, predictions)
print(f'Mean Absolute Error (MAE): {mae:.4f}')
