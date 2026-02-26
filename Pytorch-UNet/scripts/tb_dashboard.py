#!/usr/bin/env python3
"""Simple Flask dashboard to visualize TensorBoard scalar tags from event files.

Usage:
  conda activate AI
  pip install flask tensorboard
  python scripts/tb_dashboard.py --logdir runs/exp_local --port 5000

Open http://localhost:5000 in your browser.
"""
from pathlib import Path
import argparse
import json
from flask import Flask, jsonify, render_template

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception:
    EventAccumulator = None

app = Flask(__name__, template_folder=Path(__file__).parent / 'templates')

LOGDIR = None


def load_scalars(logdir: Path, max_points=500):
    """Load scalar data from the latest event files under `logdir`.
    Returns dict: {tag: [{"step": int, "value": float, "wall_time": float}, ...]}
    """
    if EventAccumulator is None:
        raise RuntimeError('tensorboard package not installed')
    result = {}
    files = sorted(logdir.glob('events*'))
    if not files:
        return result

    # prefer the most recent event file
    for f in files[::-1]:
        try:
            ea = EventAccumulator(str(f))
            ea.Reload()
            tags = ea.Tags().get('scalars', [])
            for tag in tags:
                try:
                    evs = ea.Scalars(tag)
                    # reduce to last `max_points`
                    evs = evs[-max_points:]
                    result[tag] = [
                        {'step': int(e.step), 'value': float(e.value), 'wall_time': float(e.wall_time)}
                        for e in evs
                    ]
                except Exception:
                    continue
            # if we found any scalar tags from this file, stop (prefer latest file)
            if result:
                break
        except Exception:
            continue
    return result


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/data')
def data():
    try:
        d = load_scalars(Path(LOGDIR))
        return jsonify({'ok': True, 'scalars': d})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


def main():
    global LOGDIR
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='runs/exp_local')
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()

    LOGDIR = args.logdir
    logpath = Path(LOGDIR)
    if not logpath.exists():
        print(f'Logdir does not exist: {LOGDIR}')
        return

    print('Serving TensorBoard scalars from', LOGDIR)
    print(f'Open http://{args.host}:{args.port} in your browser')
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
