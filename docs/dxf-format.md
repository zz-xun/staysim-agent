# DXF Input Contract

The importer is intentionally narrow so the same drawing produces the same YAML scene.

## Required entities

| Layer | Entity | Meaning |
| --- | --- | --- |
| `STAYSIM_BOUNDARY` | closed axis-aligned `LWPOLYLINE` | rectangular simulation area |
| `STAYSIM_ENTRANCE` | `CIRCLE` | entry/exit point or range |
| `STAYSIM_BENCH`, `STAYSIM_TREE`, ... | `CIRCLE` | facility influence range |
| `STAYSIM_NOTE` | text | human-readable annotation only |

The importer supports meters and millimeters, records scale and origin metadata, and publishes a content-addressed YAML file. Default facility values such as attractiveness and capacity are explicit assumptions in `source_import.assumptions`; they are not inferred from geometry.

## Deliberate limits

DWG, images, blocks, arbitrary polygons, curves, 3D entities, and obstacle-aware pathfinding are not part of this version. A circle is an activity range, not a wall or collision obstacle. Convert DWG to the supported DXF convention before upload.
