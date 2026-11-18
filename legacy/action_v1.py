"""Fail-closed GUI action protocol and BrowserGym translation.

Only translates a model-supplied action. It never consults task answers.
BrowserGym integration must enable the relevant bid/coord/keyboard/misc subsets.
"""
import json, math, re
from dataclasses import dataclass

class ActionError(ValueError):
    pass

@dataclass(frozen=True)
class BrowserAction:
    code: str | None
    stop_status: str | None = None

def _unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ActionError('duplicate JSON key')
        out[key] = value
    return out

def parse_action(raw, *, visible_bids, width, height):
    if not isinstance(raw, str) or len(raw) > 8192:
        raise ActionError('action must be a short JSON string')
    try:
        obj = json.loads(raw, object_pairs_hook=_unique,
                         parse_constant=lambda _: (_ for _ in ()).throw(ActionError('nonfinite number')))
    except (ValueError, TypeError) as exc:
        raise ActionError('invalid JSON action') from exc
    if not isinstance(obj, dict) or not isinstance(obj.get('action'), str):
        raise ActionError('object and action are required')
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ActionError('invalid viewport')
    typ = obj['action']
    def fields(required):
        if set(obj) != required | {'action'}:
            raise ActionError('unexpected or missing action fields')
    def bid():
        v = obj.get('target')
        if not isinstance(v, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', v) or v not in visible_bids:
            raise ActionError('target is absent from current observation')
        return v
    if typ == 'click':
        if set(obj) == {'action','target'}:
            return BrowserAction('click('+repr(bid())+')')
        fields({'x','y'})
        x, y = obj['x'], obj['y']
        if (type(x) not in (int, float) or type(y) not in (int, float)
                or not math.isfinite(x) or not math.isfinite(y)
                or not 0 <= x < width or not 0 <= y < height):
            raise ActionError('coordinates outside viewport')
        return BrowserAction(f'mouse_click({x!r}, {y!r})')
    if typ == 'type':
        fields({'target','text'})
        target = bid(); text = obj['text']
        if not isinstance(text, str) or len(text) > 1000 or '\x00' in text:
            raise ActionError('invalid text')
        return BrowserAction('fill('+repr(target)+', '+repr(text)+')')
    if typ == 'scroll':
        fields({'direction','amount'})
        direction, amount = obj['direction'], obj['amount']
        if direction not in ('up','down','left','right') or type(amount) is not int or not 1 <= amount <= 2000:
            raise ActionError('invalid scroll')
        dx = amount * (1 if direction == 'right' else -1 if direction == 'left' else 0)
        dy = amount * (1 if direction == 'down' else -1 if direction == 'up' else 0)
        return BrowserAction(f'scroll({dx}, {dy})')
    if typ == 'key':
        fields({'key'})
        key = obj['key']
        if key not in ('Enter','Tab','Escape','Backspace','ArrowUp','ArrowDown','ArrowLeft','ArrowRight'):
            raise ActionError('key is not allowed')
        return BrowserAction('keyboard_press('+repr(key)+')')
    if typ == 'wait':
        fields(set())
        return BrowserAction('noop()')
    if typ == 'stop':
        fields({'status'})
        if obj['status'] not in ('done','failed'):
            raise ActionError('invalid stop status')
        return BrowserAction(None, obj['status'])
    raise ActionError('unsupported action')

def step_environment(env, raw, *, visible_bids, width, height):
    """Return the environment result unchanged; model stop is never task success."""
    action = parse_action(raw, visible_bids=visible_bids, width=width, height=height)
    if action.stop_status is not None:
        return {'model_stop': action.stop_status, 'environment_result': None}
    return {'model_stop': None, 'environment_result': env.step(action.code)}
