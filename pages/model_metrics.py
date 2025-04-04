import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import librosa
import librosa.display
import io
import base64
from PIL import Image
import os

# Function to load model and metrics
@st.cache_resource
def load_model_and_metrics():
    try:
        # Load the model
        st.info("Loading model from raga_model5.keras...")
        model = keras.models.load_model("raga_model5.keras")
        
        # Try to load the chromagrams data if available
        try:
            data = np.load("chromagrams.npz", allow_pickle=True)
        except FileNotFoundError:
            # Create a placeholder for the data if the file doesn't exist
            st.warning("chromagrams.npz not found. Using sample data instead.")
            # Create some dummy data for demonstration
            data = {
                'chromagrams': np.random.random((100, 12, 30)),  # Sample chromagrams
                'labels': np.array(['Bhairav', 'Yaman', 'Bhairavi'] * 33 + ['Bhairav'])  # Sample labels
            }
            data = type('obj', (object,), data)  # Convert to namespace
        
        return model, data
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Function to generate confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names, top_n=10):
    # Get the top N classes by frequency
    classes, counts = np.unique(y_true, return_counts=True)
    top_classes_idx = np.argsort(counts)[-top_n:]
    top_classes = classes[top_classes_idx]
    
    # Filter predictions for only these classes
    mask = np.isin(y_true, top_classes)
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    
    # Create confusion matrix
    cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=top_classes)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create class labels for display
    display_class_names = [class_names[i] for i in top_classes]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=display_class_names, yticklabels=display_class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix (Top {top_n} Ragas)')
    plt.tight_layout()
    
    return fig

# Function to generate ROC curves
def plot_roc_curves(y_true, y_score, class_names, top_n=5):
    # Convert to one-hot encoding
    y_true_binary = tf.keras.utils.to_categorical(y_true)
    
    # Get the top N classes by frequency
    classes, counts = np.unique(y_true, return_counts=True)
    top_classes_idx = np.argsort(counts)[-top_n:]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Compute ROC curve and ROC area for each class
    for i, class_idx in enumerate(top_classes_idx):
        fpr, tpr, _ = roc_curve(y_true_binary[:, class_idx], y_score[:, class_idx])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, 
                 label=f'{class_names[class_idx]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves (Top {top_n} Ragas)')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    return fig

# Function to plot learning curves
def plot_learning_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Plot training & validation accuracy
    axes[0].plot(history['accuracy'], label='Training Accuracy')
    axes[0].plot(history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot training & validation loss
    axes[1].plot(history['loss'], label='Training Loss')
    axes[1].plot(history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_ylabel('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

# Function to generate classification report as DataFrame
def get_classification_report(y_true, y_pred, class_names):
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Convert to DataFrame for better visualization
    df = pd.DataFrame(report).transpose()
    
    # Replace class indices with class names
    df.index = [class_names[int(idx)] if idx.isdigit() else idx for idx in df.index]
    
    return df

# Function to predict on a sample
def predict_and_visualize(model, audio_file, class_names):
    try:
        # Load audio and generate chromagram
        y, sr = librosa.load(audio_file, sr=22050)
        
        # Take a 30-second segment if longer
        if len(y) > 30 * sr:
            y = y[:30 * sr]
        
        # Generate chromagram
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12, n_fft=4096)
        
        # Reshape for model input
        chroma = np.expand_dims(chroma, axis=-1)
        
        # Make prediction
        prediction = model.predict(np.array([chroma]))
        predicted_class = np.argmax(prediction, axis=1)[0]
        
        # Get top 5 predictions
        top_indices = np.argsort(prediction[0])[-5:][::-1]
        top_ragas = [class_names[i] for i in top_indices]
        top_probs = [prediction[0][i] for i in top_indices]
        
        # Create visualizations
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot chromagram
        img = librosa.display.specshow(chroma[:,:,0], y_axis='chroma', x_axis='time', ax=axes[0])
        axes[0].set_title('Chromagram')
        fig.colorbar(img, ax=axes[0])
        
        # Plot prediction probabilities
        bars = axes[1].bar(top_ragas, top_probs, color='skyblue')
        axes[1].set_title('Top 5 Predictions')
        axes[1].set_ylabel('Probability')
        axes[1].set_ylim([0, 1])
        
        # Add percentage labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{top_probs[i]:.1%}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig, top_ragas, top_probs, predicted_class
        
    except Exception as e:
        st.error(f"Error analyzing audio: {e}")
        return None, [], [], None

# Function to load training history if available
def load_training_history():
    try:
        if os.path.exists("training_history.npy"):
            return np.load("training_history.npy", allow_pickle=True).item()
        else:
            st.warning("Training history file not found. Using sample data.")
            # Generate sample training history
            return {
                'accuracy': [0.45, 0.58, 0.67, 0.72, 0.76, 0.79, 0.82, 0.84, 0.86, 0.87],
                'val_accuracy': [0.41, 0.52, 0.60, 0.64, 0.67, 0.69, 0.71, 0.72, 0.73, 0.74],
                'loss': [1.72, 1.35, 1.10, 0.92, 0.78, 0.67, 0.58, 0.52, 0.47, 0.43],
                'val_loss': [1.85, 1.51, 1.28, 1.15, 1.05, 0.98, 0.92, 0.88, 0.85, 0.83]
            }
    except Exception as e:
        st.error(f"Error loading training history: {e}")
        # Return a fallback training history
        return {
            'accuracy': [0.4, 0.6, 0.8],
            'val_accuracy': [0.35, 0.5, 0.65],
            'loss': [1.5, 1.0, 0.5],
            'val_loss': [1.6, 1.2, 0.7]
        }

# Page configuration
st.set_page_config(
    page_title="Raga Classification - Model Metrics",
    page_icon="📊",
    layout="wide"
)

# Header section
st.markdown("<h1 style='text-align: center;'>Raga Classification Model Metrics</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #636EFA;'>Detailed Performance Analysis</h3>", unsafe_allow_html=True)

# Load model and data
placeholder = st.empty()

with placeholder.container():
    with st.spinner("Loading model and data..."):
        model, data = load_model_and_metrics()

placeholder.empty()  # This removes the spinner

st.success("Model and data loaded successfully!")


if model is not None:
    # Extract data if available
    if isinstance(data, dict) or isinstance(data, np.lib.npyio.NpzFile):
        X = None
        y = None
        if "chromagrams" in data and "labels" in data:
            X = data["chromagrams"]
            y = data["labels"]
        
        # Create label encoder
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        class_names = le.classes_
        
        # Generate test predictions (using a portion of the data for demonstration)
        test_size = min(500, len(X))  # Limit test size for performance
        test_indices = np.random.choice(len(X), test_size, replace=False)
        X_test = np.array(X[test_indices])
        y_test = y_encoded[test_indices]
        
        # Add a channel dimension for the CNN if needed
        if len(X_test.shape) == 3:  # If shape is (samples, features, time)
            X_test = np.expand_dims(X_test, axis=-1)  # Make it (samples, features, time, channels)
        
        # Get predictions
        with st.spinner("Generating predictions..."):
            y_pred_proba = model.predict(X_test)
            y_pred = np.argmax(y_pred_proba, axis=1)
    else:
        # If we don't have the chromagrams, extract class names from the model
        # Get number of output classes from the model's last layer
        num_classes = model.output_shape[1]
        # Create generic class names
        class_names = [f"Raga_{i}" for i in range(num_classes)]
        st.warning("No test data available. Using generic class names and limited metrics.")
        
        # Create sample data for demonstration
        y_test = np.random.randint(0, num_classes, 100)
        y_pred = np.random.randint(0, num_classes, 100)
        y_pred_proba = np.random.random((100, num_classes))
        for i in range(100):
            y_pred_proba[i] = y_pred_proba[i] / np.sum(y_pred_proba[i])
    
    # Create tabs for different metrics
    tab1, tab2, tab3, tab4 = st.tabs([
        "Model Overview", 
        "Performance Metrics", 
        "Confusion Matrix", 
        "Learning Curves",
    ])
    
    with tab1:
        st.markdown("## Model Architecture and Information")
                
        # Model summary
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        model_summary = "\n".join(stringlist)
        st.code(model_summary, language="bash")
            
        # Model visualization
        st.markdown("### Model Layers")
        
        # Count layers by type
        layer_types = {}
        for layer in model.layers:
            layer_type = layer.__class__.__name__
            if layer_type in layer_types:
                layer_types[layer_type] += 1
            else:
                layer_types[layer_type] = 1
        
        # Create a bar chart
        fig = px.bar(
            x=list(layer_types.keys()),
            y=list(layer_types.values()),
            labels={"x": "Layer Type", "y": "Count"},
            title="Model Layer Distribution",
            color=list(layer_types.keys()),
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Model complexity
        st.markdown("### Model Complexity")
        trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])
        non_trainable_params = np.sum([np.prod(v.shape) for v in model.non_trainable_weights])
        
        complexity_data = pd.DataFrame({
            "Metric": ["Total Parameters", "Trainable Parameters", "Non-trainable Parameters"],
            "Value": [f"{trainable_params + non_trainable_params:,}", 
                        f"{trainable_params:,}", 
                        f"{non_trainable_params:,}"]
        })
        st.table(complexity_data)
            
    with tab2:
        st.markdown("## Performance Metrics")
                
        # Overall accuracy and loss
        accuracy = np.mean(y_pred == y_test)
        
        # Top-K accuracy 
        top3_accuracy = np.mean([y_test[i] in np.argsort(y_pred_proba[i])[-3:] for i in range(len(y_test))])
        top5_accuracy = np.mean([y_test[i] in np.argsort(y_pred_proba[i])[-5:] for i in range(len(y_test))])
        
        metrics_data = pd.DataFrame({
            "Metric": ["Test Accuracy", "Top-3 Accuracy", "Top-5 Accuracy"],
            "Value": [f"{accuracy:.2%}", f"{top3_accuracy:.2%}", f"{top5_accuracy:.2%}"]
        })
        
        st.table(metrics_data)
        
        # Classification report for top classes
        st.markdown("### Detailed Metrics by Class")
        top_classes = 10
        report_df = get_classification_report(y_test, y_pred, class_names)
        report_df = report_df.sort_values(by='support', ascending=False).head(top_classes)
        
        # Format the report for better readability
        report_df['precision'] = report_df['precision'].map('{:.2%}'.format)
        report_df['recall'] = report_df['recall'].map('{:.2%}'.format)
        report_df['f1-score'] = report_df['f1-score'].map('{:.2%}'.format)
        report_df['support'] = report_df['support'].astype(int)
        
        st.dataframe(report_df)
        
        # ROC curves
        st.markdown("### ROC Curves")
        roc_fig = plot_roc_curves(y_test, y_pred_proba, class_names)
        st.pyplot(roc_fig)
        
        # Performance by raga category
        st.markdown("### Performance by Raga Category")
        
        # Define actual raga categories based on available class names
        # This is an example - adjust based on your actual raga categories
        raga_categories = {}
        
        # Try to categorize based on common raga categories
        morning_ragas = ["Bhairav", "Ahir Bhairav", "Todi", "Lalit", "Gunakri"]
        afternoon_ragas = ["Shuddh Sarang", "Bhimpalasi", "Multani", "Madhuvanti", "Poorvi"]
        evening_ragas = ["Yaman", "Bageshri", "Puriya Dhanashree", "Marwa", "Shree"]
        night_ragas = ["Darbari", "Malkauns", "Chandrakauns", "Jog", "Bihag"]
        
        # Add ragas that exist in your class_names
        for raga in morning_ragas:
            if raga in class_names:
                if "Morning" not in raga_categories:
                    raga_categories["Morning"] = []
                raga_categories["Morning"].append(raga)
        
        for raga in afternoon_ragas:
            if raga in class_names:
                if "Afternoon" not in raga_categories:
                    raga_categories["Afternoon"] = []
                raga_categories["Afternoon"].append(raga)
                
        for raga in evening_ragas:
            if raga in class_names:
                if "Evening" not in raga_categories:
                    raga_categories["Evening"] = []
                raga_categories["Evening"].append(raga)
                
        for raga in night_ragas:
            if raga in class_names:
                if "Night" not in raga_categories:
                    raga_categories["Night"] = []
                raga_categories["Night"].append(raga)
        
        # If no categories were found, create sample ones
        if not raga_categories:
            # Split ragas into 4 random groups for demonstration
            available_ragas = list(class_names)
            chunk_size = max(1, len(available_ragas) // 4)
            
            raga_categories = {
                "Group 1": available_ragas[:chunk_size],
                "Group 2": available_ragas[chunk_size:2*chunk_size],
                "Group 3": available_ragas[2*chunk_size:3*chunk_size],
                "Group 4": available_ragas[3*chunk_size:]
            }
        
        # Calculate accuracy for each category
        category_acc = {}
        for category, ragas in raga_categories.items():
            # Convert raga names to indices
            raga_indices = [list(class_names).index(raga) if raga in class_names else -1 for raga in ragas]
            raga_indices = [idx for idx in raga_indices if idx >= 0]
            
            if raga_indices:
                # Get samples belonging to this category
                cat_mask = np.isin(y_test, raga_indices)
                if np.any(cat_mask):
                    cat_acc = np.mean(y_pred[cat_mask] == y_test[cat_mask])
                    category_acc[category] = cat_acc
        
        # Create bar chart for category performance
        if category_acc:
            fig = px.bar(
                x=list(category_acc.keys()),
                y=list(category_acc.values()),
                labels={"x": "Raga Category", "y": "Accuracy"},
                title="Accuracy by Raga Category",
                color=list(category_acc.keys()),
                color_discrete_sequence=px.colors.qualitative.Set2,
                text_auto='.2%'
            )
            fig.update_yaxes(range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("## Confusion Matrix Analysis")
                
        # Show confusion matrix
        cm_top_n = st.slider("Number of classes to display", 5, min(20, len(class_names)), 10)
        cm_fig = plot_confusion_matrix(y_test, y_pred, class_names, top_n=cm_top_n)
        st.pyplot(cm_fig)
            
        st.markdown("### Confusion Matrix Insights")
        
        # Calculate most confused pairs
        cm = confusion_matrix(y_test, y_pred)
        np.fill_diagonal(cm, 0)  # Remove diagonal elements
        
        # Get top confused pairs
        confused_pairs = []
        for i in range(len(cm)):
            for j in range(len(cm)):
                if i != j and cm[i, j] > 0:
                    confused_pairs.append((i, j, cm[i, j]))
        
        # Sort by confusion count
        confused_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Display top confused pairs
        st.markdown("#### Most Confused Raga Pairs")
        
        confused_data = []
        for i, j, count in confused_pairs[:10]:  # Top 10 confused pairs
            # Avoid index errors
            if i < len(class_names) and j < len(class_names):
                test_count = np.sum(y_test == i)
                confusion_rate = count/test_count if test_count > 0 else 0
                confused_data.append({
                    "True Raga": class_names[i],
                    "Predicted As": class_names[j],
                    "Count": int(count),
                    "Confusion Rate": f"{confusion_rate:.1%}"
                })
        
        confused_df = pd.DataFrame(confused_data)
        st.dataframe(confused_df)
        
        # Calculate error distribution
        error_mask = y_pred != y_test
        error_counts = np.bincount(y_test[error_mask], minlength=len(class_names))
        
        # Get ragas with highest error rates
        class_counts = np.bincount(y_test, minlength=len(class_names))
        error_rates = np.zeros_like(error_counts, dtype=float)
        for i in range(len(class_counts)):
            if class_counts[i] > 0:
                error_rates[i] = error_counts[i] / class_counts[i]
        
        top_error_indices = np.argsort(error_rates)[-5:][::-1]
        
        # Display ragas with highest error rates
        st.markdown("#### Ragas with Highest Error Rates")
        
        error_data = []
        for idx in top_error_indices:
            if idx < len(class_names) and np.sum(y_test == idx) > 0:  # Avoid index errors
                error_data.append({
                    "Raga": class_names[idx],
                    "Error Rate": f"{error_rates[idx]:.1%}",
                    "Sample Count": int(np.sum(y_test == idx))
                })
        
        error_df = pd.DataFrame(error_data)
        st.dataframe(error_df)
    
    with tab4:
        st.markdown("## Learning Curves")
        
        # Load training history
        training_history = load_training_history()
                
        # Plot learning curves
        learning_fig = plot_learning_curves(training_history)
        st.pyplot(learning_fig)
            
        st.markdown("### Training Analysis")
        
        # Calculate metrics
        final_train_acc = training_history['accuracy'][-1]
        final_val_acc = training_history['val_accuracy'][-1]
        final_train_loss = training_history['loss'][-1]
        final_val_loss = training_history['val_loss'][-1]
        
        overfitting = final_val_loss / final_train_loss
        
        # Display metrics
        st.markdown(f"**Final Training Accuracy:** {final_train_acc:.1%}")
        st.markdown(f"**Final Validation Accuracy:** {final_val_acc:.1%}")
        st.markdown(f"**Final Training Loss:** {final_train_loss:.3f}")
        st.markdown(f"**Final Validation Loss:** {final_val_loss:.3f}")
        
        # Overfitting assessment
        st.markdown("### Overfitting Assessment")
        
        if overfitting > 1.3:
            st.warning(f"Potential overfitting (Loss ratio: {overfitting:.2f})")
            st.markdown("""
            **Recommendations:**
            - Increase dropout rate
            - Add regularization
            - Collect more training data
            - Use data augmentation
            """)
        elif overfitting > 1.1:
            st.info(f"Slight overfitting (Loss ratio: {overfitting:.2f})")
            st.markdown("""
            **Recommendations:**
            - Consider early stopping
            - Mild regularization
            """)
        else:
            st.success(f"No significant overfitting (Loss ratio: {overfitting:.2f})")
        
        # Convergence analysis
        st.markdown("### Convergence Analysis")
        
        # Calculate if training has converged
        last_5_val_loss = training_history['val_loss'][-5:]
        loss_diff = np.abs(np.diff(last_5_val_loss))
        avg_change = np.mean(loss_diff)
        
        if avg_change < 0.01:
            st.success(f"Training has converged (Avg change: {avg_change:.4f})")
        else:
            st.info(f"Training still improving (Avg change: {avg_change:.4f})")
            st.markdown("**Consider training for more epochs**")
        
        
else:
    st.error("Failed to load the model. Please check if 'raga_model5.keras' exists in the root directory.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center;'>
    <p>Raga Classification System | Advanced Model Metrics</p>
    <p>© 2025 Indian Classical Music Research Project</p>
</div>
""", unsafe_allow_html=True)