"""
Test Model Script for Sign Language Recognition

This script loads a trained sign language recognition model and tests it on a directory of images.
It provides accuracy statistics and displays sample predictions with images.
"""
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import matplotlib.pyplot as plt

def load_and_preprocess_image(image_path, target_size=(400, 400)):
    """Load and preprocess a single image for prediction"""
    # Load image and resize
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_sign(model, image_path):
    """Predict the sign language letter from an image"""
    # Load and preprocess the image
    img_array = load_and_preprocess_image(image_path)
    
    # Get prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    # Map class index to letter (assuming A-Z plus additional classes)
    # You might need to adjust this mapping based on your training data
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    predicted_letter = letters[predicted_class]
    
    return predicted_letter, confidence

def test_model_on_directory(model, test_dir, num_samples=5):
    """Test the model on a few random images from each class"""
    results = []
    
    for letter in os.listdir(test_dir):
        letter_path = os.path.join(test_dir, letter)
        if os.path.isdir(letter_path):
            # Get all images for this letter
            images = [f for f in os.listdir(letter_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            if not images:
                continue
                
            # Test on random samples
            test_images = np.random.choice(images, min(num_samples, len(images)), replace=False)
            
            for img_name in test_images:
                img_path = os.path.join(letter_path, img_name)
                predicted_letter, confidence = predict_sign(model, img_path)
                
                results.append({
                    'image': img_path,
                    'true_letter': letter,
                    'predicted_letter': predicted_letter,
                    'confidence': confidence
                })
    
    return results

def display_results(results):
    """Display the test results with images"""
    plt.figure(figsize=(15, 15))
    for i, (img, pred, true) in enumerate(results):
        if i < 25:  # Ensure we don't exceed the number of subplots
            plt.subplot(5, 5, i+1)
            plt.imshow(img)
            plt.title(f'Pred: {pred}, True: {true}')
            plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    # Load the trained model
    model_path = 'checkpoint_sign_converter_alkadya.keras'  # Changed to use checkpoint file
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return

    # Test the model
    test_dir = 'data/validation'  # Using validation set for testing
    results = test_model_on_directory(model, test_dir)
    
    # Calculate accuracy
    correct_predictions = sum(1 for r in results if r['true_letter'] == r['predicted_letter'])
    accuracy = correct_predictions / len(results)
    
    print(f"\nTest Results:")
    print(f"Total samples tested: {len(results)}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2%}")
    
    # Display results with images
    display_results(results)

if __name__ == "__main__":
    main() 