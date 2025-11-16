# 🧬 GeneTalk - AI-Powered Animal Emotion & Species Recognition System

GeneTalk is a comprehensive project that combines cutting-edge AI/ML models with a modern web interface to identify animals, recognize their emotional states, and provide intelligent chatbot assistance. The project leverages computer vision, audio processing, and natural language processing to create a unique animal interaction platform.

---

### 🎬 Quick Links
> **[🔗 View Live Prototype Demo](https://genetalk.netlify.app)** | **[📹 Watch Video Explanation](https://youtu.be/1xKyQouzM-k?si=1uPSYueXVnLEWcHR)**

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Tech Stack](#tech-stack)
4. [Features](#features)
5. [Installation & Setup](#installation--setup)
6. [Backend Services](#backend-services)
7. [Frontend Application](#frontend-application)
8. [AI Models](#ai-models)
9. [API Endpoints](#api-endpoints)
10. [Usage Guide](#usage-guide)
11. [Deployment](#deployment)
12. [Contributing](#contributing)
13. [License](#license)

---

## 🎯 Project Overview

GeneTalk is an intelligent animal recognition and emotion analysis system that processes multiple data modalities:

- **Image Recognition**: Identifies animals from images using deep learning models
- **Audio Analysis**: Recognizes animal species and their emotional states from audio/vocalizations
- **Intelligent Chatbot**: Provides information about animals, health precautions, and emotion indicators using Google's Gemini API
- **Unified Interface**: Modern Next.js frontend that combines all services into a seamless user experience

### Key Objectives
- Accurate animal species identification from images and audio
- Emotion and intent recognition from animal sounds
- Educational chatbot with comprehensive animal knowledge base
- User-friendly interface for uploading and analyzing media



## 📁 Project Structure

```
GeneTalk/
├── AI_model/                          # Machine Learning Models & Training
│   ├── Audio_model/                   # Audio processing & emotion detection
│   │   ├── model.ipynb               # Audio model training notebook
│   │   ├── combined_data.csv         # Training dataset
│   │   └── audio_df/                 # Audio files for training
│   │
│   └── Image_model/                   # Computer vision models
│       ├── Model_training.ipynb      # CNN model training
│       ├── model.ipynb               # Main model notebook
│       ├── dogs.ipynb                # Dog breed identification
│       ├── best_animal_emotion_model.h5  # Trained emotion model
│       ├── best_custom_cnn_model.h5  # Trained species model
│       ├── animals/                  # Animal images dataset
│       ├── dog/                      # Dog images dataset
│       └── processed_data/           # Preprocessed training data
│
├── Backend/                           # FastAPI Backend Services
│   ├── Audio_Model/                   # Audio prediction service
│   │   ├── main.py                  # FastAPI server for audio analysis
│   │   ├── requirements.txt         # Python dependencies
│   │   ├── DockerFile               # Docker container config
│   │   ├── render.yaml              # Render deployment config
│   │   ├── ModelFile/               # Trained model files
│   │   ├── uploads/                 # Temporary audio file storage
│   │   └── test_audio_model.py      # Testing script
│   │
│   ├── Chatbot/                       # Gemini Chatbot Service
│   │   ├── main.py                  # FastAPI chatbot server
│   │   ├── requirements.txt         # Python dependencies
│   │   └── render.yaml              # Render deployment config
│   │
│   └── Image_model/                   # Image prediction service
│       ├── run.py                   # FastAPI server startup
│       ├── requirements.txt         # Python dependencies
│       ├── README.md                # Service documentation
│       ├── app/                     # Application code
│       └── models/                  # Model storage
│
└── Frontend/                          # Next.js Frontend Application
    └── yaya/
        ├── app/                     # Next.js app directory
        ├── components/              # Reusable React components
        ├── hooks/                   # Custom React hooks
        ├── lib/                     # Utility functions
        ├── package.json            # Node.js dependencies
        ├── tsconfig.json           # TypeScript configuration
        ├── next.config.mjs         # Next.js configuration
        └── postcss.config.mjs       # PostCSS configuration

```

---

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI (Python)
- **ML/DL**: TensorFlow, Keras
- **Audio Processing**: librosa, soundfile
- **NLP**: Google Generative AI (Gemini)
- **ML Utils**: scikit-learn, numpy, pandas
- **Deployment**: Docker, Render
- **Server**: Uvicorn, Starlette

### Frontend
- **Framework**: Next.js 16.0.0
- **UI Library**: React 19.2.0 with TypeScript
- **UI Components**: Radix UI
- **Styling**: Tailwind CSS with custom animations
- **State Management**: React Hook Form
- **HTTP Client**: Axios
- **Charts**: Recharts
- **Notifications**: Sonner

### AI/ML
- **Vision Models**: Custom CNN for animal classification and emotion detection
- **Audio Models**: Mel-spectrogram based neural networks for species and emotion recognition
- **NLP**: Google Gemini API for conversational AI
- **Feature Extraction**: TF-IDF for semantic retrieval

---

## ✨ Features

### 🖼️ Image Analysis
- Upload animal images for species identification
- Multi-class animal classification (Dogs, Cats, Horses, Cows, etc.)
- Emotion detection from animal facial expressions
- Confidence scores for predictions
- Support for multiple image formats

### 🔊 Audio Analysis
- Upload animal vocalizations (WAV, MP3, AIF formats)
- Species identification from sounds
- Emotion/intent recognition (Attack, Care, Play)
- Real-time audio preprocessing and feature extraction
- Mel-spectrogram visualization and analysis

### 🤖 Intelligent Chatbot
- Knowledge base with 16+ animal species
- Context-aware responses using chat history
- Health precautions and care information
- Emotion indicators and behavioral signs
- Powered by Google Gemini API for natural conversations
- Semantic retrieval for relevant information

### 🌐 Web Interface
- Modern, responsive UI built with Next.js and Radix UI
- Real-time file uploads and processing
- Results visualization and confidence indicators
- Chat interface with conversation history
- Dark/Light mode support
- Mobile-friendly design

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9+ (for backend)
- Node.js 18+ (for frontend)
- npm or pnpm (Node package manager)
- Google API Key (for Chatbot service)
- Git

### Backend Setup

#### 1. Audio Model Service
```bash
cd Backend/Audio_Model
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8001
```

#### 2. Chatbot Service
```bash
cd Backend/Chatbot
pip install -r requirements.txt
# Set your Google API key
export GOOGLE_API_KEY="your_api_key_here"
uvicorn main:app --host 0.0.0.0 --port 8002
```

#### 3. Image Model Service
```bash
cd Backend/Image_model
pip install -r requirements.txt
python run.py
```

### Frontend Setup
```bash
cd Frontend/yaya

# Install dependencies
pnpm install
# or
npm install

# Run development server
pnpm dev
# or
npm run dev

# Access the application
# Open http://localhost:3000 in your browser
```

---

## 🔧 Backend Services

### 1. Audio Model Service (`Backend/Audio_Model/`)

**Purpose**: Analyze animal vocalizations and extract species and emotion information

**Key Features**:
- Accepts audio file uploads (AIF, WAV, MP3)
- Preprocesses audio into Mel-spectrograms
- Predicts species and emotional intent
- Returns confidence scores

**API Endpoint**:
- `POST /audio_predict/` - Upload audio file for prediction

**Configuration**:
```python
MODEL_PATH = "ModelFile/final_audio_model.h5"
SAMPLE_RATE = 22050
N_MELS = 128
DURATION = 3.0
```

**Dependencies**: TensorFlow, librosa, FastAPI, numpy

---

### 2. Chatbot Service (`Backend/Chatbot/`)

**Purpose**: Provide intelligent conversation about animals and their care

**Key Features**:
- 16+ animal species in knowledge base
- TF-IDF based semantic retrieval
- Conversation history (last 10 messages)
- Google Gemini integration for natural responses
- Health precautions and emotion indicators

**API Endpoints**:
- `GET /` - Health check
- `POST /chat` - Send message and receive response

**Knowledge Base Includes**:
- Horse, Frog, Goat, Housefly, Monkey, Dog, Cat, Cow, Buffalo, Mosquito, Bee, Peacock, Crow, Parrot, Sparrow, Elephant

**Dependencies**: FastAPI, google-generativeai, scikit-learn

---

### 3. Image Model Service (`Backend/Image_model/`)

**Purpose**: Classify animals from uploaded images

**Key Features**:
- Multi-class animal classification
- Emotion detection from facial features
- High accuracy CNN models
- Fast inference

**Pre-trained Models**:
- `best_custom_cnn_model.h5` - Species classification
- `best_animal_emotion_model.h5` - Emotion recognition

**Dependencies**: TensorFlow, FastAPI, Pillow, numpy

---

## 🧠 AI Models

### Image Models

#### Custom CNN Model
- **Purpose**: Animal species classification
- **Input**: Preprocessed images (224×224×3)
- **Output**: Species probability distribution
- **Classes**: Multiple animal classes (Dogs, Cats, Horses, Cows, etc.)
- **Location**: `AI_model/Image_model/best_custom_cnn_model.h5`

#### Emotion Detection Model
- **Purpose**: Recognize emotions from animal facial expressions
- **Input**: Animal images
- **Output**: Emotion classification
- **Location**: `AI_model/Image_model/best_animal_emotion_model.h5`

### Audio Models

#### Species & Emotion Recognition Model
- **Purpose**: Identify species and emotional intent from vocalizations
- **Input**: Mel-spectrograms (128×N)
- **Output**: 
  - Species classification
  - Intent/emotion classification (Attack, Care, Play)
- **Processing**: 
  - Audio → Mel-spectrogram transformation
  - Normalization and resizing
  - Neural network inference
- **Location**: `Backend/Audio_Model/ModelFile/final_audio_model.h5`

### Training Notebooks

- **`AI_model/Audio_model/model.ipynb`**: Training pipeline for audio models
- **`AI_model/Image_model/Model_training.ipynb`**: CNN training for species classification
- **`AI_model/Image_model/model.ipynb`**: General image model experiments
- **`AI_model/Image_model/dogs.ipynb`**: Dog breed specific training

---

## 🔌 API Endpoints

### Audio Analysis API
**Base URL**: `http://localhost:8001`

```
POST /audio_predict/
Description: Upload an audio file and get species/emotion predictions
Request: multipart/form-data with 'file' field
Response:
{
    "species": "Horse",
    "species_confidence": 0.95,
    "emotion": "Play",
    "emotion_confidence": 0.87
}
```

### Chatbot API
**Base URL**: `http://localhost:8002`

```
GET /
Description: Health check endpoint
Response: {"message": "🐾 Gemini Species Chatbot API is running!"}

POST /chat
Description: Send a message and receive an intelligent response
Request: {"message": "What do horses eat?"}
Response: {"reply": "Horses are herbivores that primarily eat..."}
```

### Image Analysis API
**Base URL**: `http://localhost:8000`

```
POST /predict
Description: Upload an image for species and emotion prediction
Request: multipart/form-data with 'file' field
Response: 
{
    "species": "Dog",
    "emotion": "Happy",
    "confidence": 0.92
}
```

---

## 💻 Usage Guide

### Using the Web Interface

1. **Open the Application**
   - Navigate to `http://localhost:3000`

2. **Image Upload & Analysis**
   - Click on "Upload Image" section
   - Select an animal image
   - View species identification and emotion analysis results
   - Check confidence scores

3. **Audio Upload & Analysis**
   - Click on "Upload Audio" section
   - Select an audio file (AIF, WAV, MP3)
   - Get species and emotion predictions
   - View detailed confidence metrics

4. **Chat with the Chatbot**
   - Click on "Chat" section
   - Type questions about animals
   - Ask about care, health, behaviors
   - View conversation history

### Example Queries

**For the Chatbot**:
- "What do horses eat?"
- "How do I care for a dog?"
- "What are the signs of stress in cats?"
- "Tell me about elephants"
- "How long do frogs live?"

---

## 🚀 Deployment

### Docker Deployment

#### Audio Model Service
```dockerfile
# Build
docker build -f Backend/Audio_Model/DockerFile -t audio-model .

# Run
docker run -p 8001:8000 audio-model
```

#### Using Render
Each backend service includes a `render.yaml` configuration:

```yaml
services:
  - type: web
    name: audio-model-service
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Frontend Deployment

**Vercel (Recommended for Next.js)**
```bash
npm install -g vercel
vercel
```

**Other Platforms**
```bash
npm run build
npm run start
```

---

## 📊 Data Formats

### Audio Files Supported
- `.aif` / `.aiff` - Audio Interchange File Format
- `.wav` - Waveform Audio Format
- `.mp3` - MPEG Audio Layer III

### Image Files Supported
- `.jpg` / `.jpeg` - JPEG format
- `.png` - Portable Network Graphics
- `.gif` - Graphics Interchange Format

### Audio Processing Pipeline
1. Load audio with librosa (sample rate: 22050 Hz)
2. Trim silence (top_db=30)
3. Pad or trim to 3 seconds
4. Compute Mel-spectrogram (128 bins, FFT size: 2048)
5. Convert to dB scale and normalize
6. Resize to model input shape
7. Expand dimensions for batch processing

---

## 🎓 Training Data

### Audio Dataset
- **Location**: `AI_model/Audio_model/audio_df/`
- **Format**: AIF files with naming convention `[Species]-[ID]-[Type]-[Subtype].aif`
- **Species**: Farley (Horse), Freid (various)
- **Types**: A (Audio), C (Call), P (Play)
- **CSV Labels**: `combined_data.csv`

### Image Dataset
- **Location**: `AI_model/Image_model/`
- **Directories**: 
  - `animals/` - General animal images
  - `dog/` - Dog-specific images
  - `dataset_small/` - Smaller subset for testing
- **Processed**: `processed_data/` contains preprocessed versions

---

## 🔐 Environment Configuration

### Chatbot Service
```bash
# Set Google API Key
export GOOGLE_API_KEY="your_google_api_key_here"

# Or create .env file
GOOGLE_API_KEY=your_key_here
```

### CORS Configuration
All services are configured with CORS enabled for cross-origin requests:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 🛠️ Development

### Adding New Animal Species

1. **Update Chatbot Knowledge Base** (`Backend/Chatbot/main.py`)
```python
KB = {
    "NewAnimal": {
        "info": "Description...",
        "health_precautions": "Care tips...",
        "emotion_signs": "Behavior indicators..."
    }
}
```

2. **Train Image Model** with new images in `AI_model/Image_model/animals/`

3. **Train Audio Model** with vocalizations in `AI_model/Audio_model/audio_df/`

### Adding New Features

- **New ML Models**: Add training notebooks in `AI_model/`
- **New API Endpoints**: Extend FastAPI services in `Backend/`
- **New UI Components**: Create React components in `Frontend/yaya/components/`

---

## 📝 Project Configuration Files

### Backend Configuration
- **`Backend/Audio_Model/requirements.txt`** - Audio service dependencies
- **`Backend/Chatbot/requirements.txt`** - Chatbot dependencies
- **`Backend/Image_model/requirements.txt`** - Image service dependencies

### Frontend Configuration
- **`Frontend/yaya/package.json`** - Node.js dependencies and scripts
- **`Frontend/yaya/tsconfig.json`** - TypeScript settings
- **`Frontend/yaya/next.config.mjs`** - Next.js configuration
- **`Frontend/yaya/tailwind.config.js`** - Tailwind CSS customization

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is part of a hackathon initiative. Please check the repository for specific license information.

---

## 📞 Support & Contact

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing documentation in service README files
- Review model training notebooks for implementation details

---

## 🎯 Future Enhancements

- [ ] Real-time audio streaming analysis
- [ ] Video-based animal detection and tracking
- [ ] Multi-species simultaneous detection
- [ ] Advanced emotion nuances (fear, joy, aggression distinctions)
- [ ] Augmented Reality (AR) animal overlays
- [ ] Mobile application
- [ ] Cloud-based model serving (AWS, GCP, Azure)
- [ ] Advanced audio features (spectrogram visualization)
- [ ] User authentication and history
- [ ] Model performance analytics dashboard

---

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Next.js Documentation](https://nextjs.org/docs)
- [Google Generative AI API](https://ai.google.dev/)
- [Librosa Audio Processing](https://librosa.org/)

---

**Made with ❤️ for the hackathon - Street Crew Team**

Last Updated: November 16, 2025


