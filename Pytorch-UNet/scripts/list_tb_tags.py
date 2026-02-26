from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import sys

root = Path(__file__).resolve().parents[1]
logdir = root / 'runs' / 'exp_local'
files = sorted(logdir.glob('events*'))
if not files:
    print('No event files found in', logdir)
    sys.exit(1)

for f in files:
    print('File:', f)
    try:
        ea = EventAccumulator(str(f))
        ea.Reload()
        tags = ea.Tags()
        print('Tags:', list(tags.keys()))
        for k in tags:
            v = tags.get(k)
            if isinstance(v, bool):
                print('  ', k, ':', v)
            else:
                try:
                    print('  ', k, ':', len(v))
                except Exception:
                    print('  ', k, ':', type(v))

        # show last scalar values (if any)
        scalar_tags = tags.get('scalars') or []
        for tag in scalar_tags:
            try:
                evs = ea.Scalars(tag)
                if evs:
                    last = evs[-1]
                    print(f"    scalar {tag}: last step={last.step}, value={last.value}")
                else:
                    print(f"    scalar {tag}: no events")
            except Exception as e:
                print(f"    scalar {tag}: failed to read: {e}")
    except Exception as e:
        print('  Failed to read event file:', e)
    print()
