# DXF Input Contract / DXF 输入契约

The importer is intentionally narrow so the same drawing produces the same YAML scene.

导入器有意保持输入范围窄化，以保证同一张图纸生成同一份 YAML 场景。

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

DWG、图片、块、任意多边形、曲线、三维实体和障碍物寻路不属于当前版本。圆表示活动作用范围，不是墙体或碰撞障碍。上传前请将 DWG 转换为支持的 DXF 约定。
