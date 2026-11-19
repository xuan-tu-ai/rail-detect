# rail-detect

Railway track segmentation and on-track obstacle detection from UAV images, trained on the
[UAV-RSOD](https://zenodo.org/records/12606374) dataset with Ultralytics YOLO11.

- Track model: YOLO11-seg, classes `rail_inside`, `rail_lines` (dataset V1, 315 frames).
- Obstacle model: YOLO11 detect, classes Branch, IronRod, Barrel, Boulder, Jerrycan, Person (dataset V2).
- Demo: overlays the track region and reports only obstacles that intersect it or lie within 0.5 m of it.

## Setup

```bash
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

## Pipeline

| Step | Script |
|---|---|
| Download dataset | `scripts/01_download.py` |

## Data and license

UAV-RSOD: Rampriya R.S. et al., "An unmanned aerial vehicle captured dataset for railroad segmentation
and obstacle detection", *Scientific Data* 11:1315 (2024), doi:10.1038/s41597-024-03952-3. CC BY 4.0.

Ultralytics YOLO is AGPL-3.0.
