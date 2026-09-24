"""Run OrthoGraph in the current Lightning Studio."""
import os
from pathlib import Path
import uvicorn

if __name__ == '__main__':
    uvicorn.run('app:create_app',factory=True,
                host=os.getenv('ORTHOGRAPH_HOST','0.0.0.0'),
                port=int(os.getenv('PORT','7860')),workers=1)
