# Face Matching Project

This project identifies faces in photos and copies matching photos from the `@all_photos` folder into the corresponding folders containing `_SELFIE` images.

## Features

- Detects faces in `_SELFIE` photos.
- Matches detected faces with photos in the `@all_photos` folder.
- Copies matching photos to the respective `_SELFIE` folders.

## Requirements

- Python 3.7+
- Virtual environment (`.venv`)
- Libraries: `face_recognition`, `opencv-python`, `dlib`

## Setup Instructions

### Step 1: Clone the Project

```bash
git clone <repository_url>
cd recognition
```

### Step 2: Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # For macOS/Linux
.venv\Scripts\activate   # For Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Program

Place the `images` folder in the project root directory and execute:

```bash
python face_matching.py
```

### Step 5: Deactivate Virtual Environment

After running the script, deactivate the virtual environment:

```bash
deactivate
```

## Project Structure

```
face_matching_project/
│
├── .venv/                 # Virtual environment
├── face_matching.py      # Main script
├── requirements.txt      # Dependencies
└── README.md             # Documentation
```

## Notes

- Ensure all photos are in JPEG format.
- Place the `images` folder in the project root directory before running the script.
- Install `dlib` dependencies if installation fails. Refer to [dlib documentation](http://dlib.net/).

## License

This project is under a private license, use not approved by the author is not allowed.
