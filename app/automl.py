from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots


iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Best pipeline from TPOT
best_pipeline = Pipeline(steps=[
    ('normalizer', Normalizer()),
    ('gradientboostingclassifier', GradientBoostingClassifier(
        max_depth=8, max_features=0.1, min_samples_leaf=14, 
        min_samples_split=10, subsample=0.8))
])

# Train the best pipeline
best_pipeline.fit(X_train, y_train)

# Evaluate and store the best result
accuracy = best_pipeline.score(X_test, y_test)
predictions = best_pipeline.predict(X_test)

# Create a scatter plot for visualization
def create_plot(X_test, predictions):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Sepal Length vs Sepal Width", "Petal Length vs Petal Width"))

    # First subplot
    fig.add_trace(go.Scatter(
        x=X_test[:, 0],
        y=X_test[:, 1],
        mode='markers',
        marker=dict(
            color=predictions,
            colorscale='Viridis',
            colorbar=dict(title="Classes of IRIS")
        ),
        text=[iris.target_names[pred] for pred in predictions]
    ), row=1, col=1)

    # Second subplot
    fig.add_trace(go.Scatter(
        x=X_test[:, 2],
        y=X_test[:, 3],
        mode='markers',
        marker=dict(
            color=predictions,
            colorscale='Viridis'
        ),
        text=[iris.target_names[pred] for pred in predictions]
    ), row=1, col=2)

    # Update x and y axis labels
    fig.update_xaxes(title_text="Sepal Length (cm)", row=1, col=1)
    fig.update_yaxes(title_text="Sepal Width (cm)", row=1, col=1)
    fig.update_xaxes(title_text="Petal Length (cm)", row=1, col=2)
    fig.update_yaxes(title_text="Petal Width (cm)", row=1, col=2)

    # Convert plot to HTML
    plot_html = pio.to_html(fig, full_html=False)
    return plot_html


plot_html = create_plot(X_test, predictions)

best_result = {
    "accuracy": accuracy,
    "pipeline": str(best_pipeline),
    "plot_html": plot_html
}

def train_automl():
    return best_result

def predict_new_data(new_data):
    new_prediction = best_pipeline.predict(new_data)
    updated_X_test = np.vstack([X_test, new_data])
    updated_predictions = np.append(predictions, new_prediction)
    updated_plot_html = create_plot(updated_X_test, updated_predictions)
    
    updated_result = {
        "accuracy": accuracy,
        "pipeline": str(best_pipeline),
        "plot_html": updated_plot_html
    }
    return updated_result
