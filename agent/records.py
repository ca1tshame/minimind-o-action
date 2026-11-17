"""Fail-closed GUI training record validation; no dataset-wide reads or downloads."""
import hashlib, io
from pathlib import Path
from agent.action import parse_action
from PIL import Image, UnidentifiedImageError

class RecordError(ValueError):
    pass

def validate_training_record(row, *, data_root, approved_sources):
    """Validate an already-labelled record; never synthesize missing targets.

    Caller must supply source names whose training licences were independently
    established. Evaluation/test records cannot be used for training.
    """
    if not isinstance(row, dict) or row.get('split') != 'train':
        raise RecordError('only explicitly designated training records are permitted')
    source = row.get('source')
    if not isinstance(source, str) or source not in approved_sources:
        raise RecordError('source licence not approved')
    if not isinstance(row.get('task'), str) or not row['task'].strip():
        raise RecordError('missing task')
    images = row.get('images')
    if not isinstance(images, list) or not images:
        raise RecordError('missing observation images')
    base = Path(data_root).resolve(); validated=[]
    for i, im in enumerate(images):
        if not isinstance(im, dict) or set(im) != {'path','sha256','turn'} or type(im['turn']) is not int or im['turn'] < 0:
            raise RecordError('invalid image binding')
        if i and im['turn'] < images[i-1]['turn']:
            raise RecordError('image turns not in temporal order')
        path = (base / im['path']).resolve()
        if not path.is_relative_to(base) or not path.is_file():
            raise RecordError('missing image or path outside data root')
        blob=path.read_bytes(); digest = hashlib.sha256(blob).hexdigest()
        if digest != im['sha256']:
            raise RecordError('image digest mismatch')
        try:
            with Image.open(io.BytesIO(blob)) as img:
                img.verify()
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            raise RecordError('invalid image') from exc
        validated.append({'path':str(path),'sha256':digest,'turn':im['turn']})
    viewport=row.get('viewport')
    if not isinstance(viewport,dict): raise RecordError('missing viewport')
    raw=row.get('action_json')
    try:
        action=parse_action(raw,visible_bids=set(row.get('visible_bids',[])),width=viewport.get('width'),height=viewport.get('height'))
    except (ValueError, TypeError) as exc:
        raise RecordError('invalid or incomplete action label') from exc
    return {'source':source,'task':row['task'],'images':validated,'action_json':raw,
            'browser_action':action.code,'model_stop':action.stop_status}
