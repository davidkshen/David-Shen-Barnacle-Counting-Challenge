"""Write a portable, offline HTML review tool; no server or external services."""
import base64
import io
import json
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image
from .audit import tile_counts, random_tiles


def create_review(image, predictions, path, name='Barnacle review'):
    buf = io.BytesIO(); Image.fromarray(image).save(buf, format='JPEG', quality=93)
    data = dict(name=name, image='data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode(),
        predictions=np.round(predictions, 4).tolist(), model_counts=tile_counts(predictions).tolist(),
        random_sample=random_tiles().tolist(), size=1024, grid=8,
        key=hashlib.sha256(buf.getvalue()+np.asarray(predictions).tobytes()).hexdigest()[:16])
    template = Path(__file__).with_name('review_template.html').read_text()
    Path(path).write_text(template.replace('__DATA__', json.dumps(data)))
