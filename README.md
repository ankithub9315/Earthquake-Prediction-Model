# 🌍 Earthquake Prediction and Analysis App

An interactive Streamlit application for earthquake prediction and analysis using machine learning. This app provides real-time earthquake predictions, historical data visualization, and trend analysis using a neural network model.

## 🌟 Features

- **Real-time Earthquake Prediction**: Input coordinates and time to predict earthquake magnitude and depth
- **Historical Data Analysis**: View and filter historical earthquake data with interactive maps
- **Trend Analysis**: Visualize earthquake patterns, frequencies, and distributions
- **Model Performance Tracking**: Monitor the neural network's training performance
- **Interactive Maps**: Explore earthquake locations and hotspots with dynamic mapping

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/earthquake-prediction-app.git
cd earthquake-prediction-app
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the earthquake database and place it in the project root directory as `database.csv`

## 📊 Data Requirements

The application expects a CSV file named `database.csv` with the following columns:
- Date (MM/DD/YYYY format)
- Time (HH:MM:SS format)
- Latitude
- Longitude
- Depth
- Magnitude

## 🔧 Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Navigate to the different sections using the sidebar:
   - 🔮 Prediction: Input coordinates and time for earthquake predictions
   - 📜 Historical Data: Explore past earthquake data
   - 📈 Trends: Analyze patterns and distributions
   - 📊 Model Performance: Review model training metrics

## 🧠 Model Architecture

The application uses a Sequential Neural Network with the following architecture:
- Input layer: 3 features (Timestamp, Latitude, Longitude)
- Hidden layers:
  - Dense(64) with ReLU activation
  - Dropout(0.2)
  - Dense(32) with ReLU activation
  - Dropout(0.2)
  - Dense(16) with ReLU activation
- Output layer: Dense(2) for Magnitude and Depth prediction

## 🛠️ Technical Details

- **Framework**: Streamlit
- **Machine Learning**: TensorFlow, Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Maps**: Plotly Mapbox
- **Caching**: Streamlit cache for improved performance

## 🔑 Key Features Explained

### Prediction System
- Uses historical earthquake data for training
- Implements early stopping to prevent overfitting
- Provides real-time predictions with uncertainty estimates

### Data Visualization
- Interactive maps showing earthquake locations
- Time series analysis of earthquake frequency
- Magnitude and depth distribution visualizations
- Heatmaps of earthquake hotspots

### Data Processing
- Automatic data cleaning and preprocessing
- Timestamp conversion and standardization
- Feature scaling for improved model performance

## 📝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- *Ankit* - [YourGitHub](https://github.com/ankithub9315)

## 🙏 Acknowledgments

- Thanks to all contributors who help improve this project
- Earthquake data providers
- Streamlit team for their amazing framework
