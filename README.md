# Face Matching Project

This project identifies faces in photos and copies matching photos from the `@all_photos` folder into the corresponding folders containing `_SELFIE` images.

## Features

- Detects faces in `_SELFIE` photos.
- Matches detected faces with photos in the `@all_photos` folder.
- Copies matching photos to the respective `_SELFIE` folders.

## Requirements

- Python 3.13+
- Virtual environment (`.venv`)
- Libraries: `face_recognition`, `opencv-python`, `dlib`, etc...

## Setup Instructions

### Step 1: Clone the Project

```bash
git clone https://github.com/pavelveter/recognition.git
cd recognition
```

### Step 2: Create a Virtual Environment

```bash
uv venv
source .venv/bin/activate  # For macOS/Linux
.venv\Scripts\activate   # For Windows
```

### Step 3: Install Dependencies

```bash
uv sync
```

### Step 4: Run the Program

Place the `images` folder in the project root directory and execute:

```bash
uv run recognition.py
```

### Step 5: Deactivate Virtual Environment

After running the script, deactivate the virtual environment:

```bash
deactivate
```

## Notes

- Ensure all photos are in JPEG format.
- Place the `images` folder in the project root directory before running the script.

## License

This project is under a private license, use not approved by the author is not allowed.
