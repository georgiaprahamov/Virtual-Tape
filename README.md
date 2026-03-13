# Virtual Tape Measure

A Python and OpenCV-based tool to measure objects in real-time using a webcam. The tool uses a standard bank card as a reference to calibrate lengths and measure objects detected in the video feed.

## Requirements

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy (`numpy`)

## Installation

1. Create a virtual environment (optional but recommended):
```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the `main.py` script:
```bash
python main.py
```

2. Controls:
- **C**: Calibrate the camera by placing a standard bank card on the surface.
- **R**: Reset calibration to default.
- **ESC**: Exit the application.

## Files

- `main.py`: The main program logic for the virtual tape measure.
- `Technical_documentary.pdf`: Technical documentation for the application.
