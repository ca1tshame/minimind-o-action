"""V12 context management: token-aware history packing shared by RL and eval.

Keeps, in priority order: instruction (always), last error feedback, the most
recent K actions verbatim, then older actions as compact "kind-only" stubs.
Never exceeds the model context and always reserves an output budget.
"""
def compact_history(history, tok, budget_tokens, reserve_out=32, kind_only_after=2):
    """Return (text, stats) fitting budget_tokens (excluding the action target)."""
    if not history:
        return '', {'kept': 0, 'stubbed': 0, 'dropped': 0}
    def ntok(s):
        return len(tok(s, add_special_tokens=False)['input_ids'])
    def kind_of(a):
        for k in ('click', 'type', 'scroll', 'stop', 'longpress', 'scroll_at'):
            if k in a:
                return '<%s>' % k
        return '<act>'
    recent = list(history[-kind_only_after:])
    older = list(history[:-kind_only_after])
    stubbed = 0
    dropped = 0
    while True:
        parts = []
        if older:
            parts.append('Earlier: ' + ';'.join(kind_of(a) for a in older))
        if recent:
            parts.append('Previous actions: ' + '; '.join(recent))
        text = ('\n'.join(parts) + '\n') if parts else ''
        if ntok(text) <= max(0, budget_tokens - reserve_out):
            return text, {'kept': len(recent), 'stubbed': stubbed, 'dropped': dropped}
        if recent and len(recent) > 1:
            recent.pop(0); dropped += 1
        elif recent:
            older.append(recent.pop(0)); stubbed += 1
        elif older:
            older.pop(0); dropped += 1
        else:
            return '', {'kept': 0, 'stubbed': 0, 'dropped': dropped}
