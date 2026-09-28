"""Imported rectangle scenes share one validated runtime/history contract."""

import copy
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from staysim import domain, results, simulation


def scene_data():
    return {
        'name': 'DXF small site', 'width': 12, 'height': 7,
        'elements': [
            {'type': 'entrance', 'x': 0, 'y': 3, 'radius': 1},
            {'type': 'bench', 'x': 11, 'y': 6, 'radius': 0.5},
        ],
        'source_import': {'format': 'dxf', 'source_sha256': 'example',
                          'origin': [1000, 2000], 'scale_to_m': 0.001},
    }


@pytest.mark.parametrize('changes', [
    {'width': 0}, {'width': -1}, {'height': float('inf')}, {'width': True},
    {'height': '7'}, {'elements': {}}, {'source_import': []},
    {'source_import': {'scale': float('nan')}},
])
def test_scene_rejects_invalid_dimensions_and_metadata(tmp_path, changes):
    data = scene_data()
    data.update(changes)
    path = tmp_path / 'scene.yaml'
    path.write_text(yaml.safe_dump(data), encoding='utf-8')
    with pytest.raises(ValueError):
        domain.load_scenario(str(path))


@pytest.mark.parametrize('changes', [
    {'type': 'wall'}, {'type': ['bench']}, {'x': -1}, {'y': 8},
    {'x': float('nan')}, {'radius': 0}, {'radius': float('inf')},
    {'capacity': 1.5}, {'capacity': 0}, {'capacity': True},
    {'shade': 'false'}, {'attractiveness': 1.1}, {'attractiveness': -0.1},
])
def test_scene_rejects_invalid_facilities(changes):
    data = scene_data()
    data['elements'][0].update(changes)
    with pytest.raises(ValueError):
        domain.environment_from_dict(data)


def test_non_leaving_agent_stays_inside_small_imported_site():
    env = domain.environment_from_dict(scene_data())
    persona = domain.get_default_personas()['通勤青年']
    agent = domain.Agent(id=1, persona=persona, x=12, y=7)
    # No target means random wander; the engine is never needed on this branch.
    for _ in range(30):
        simulation.update_agent(agent, 20, env, None, 15)
        assert 0 <= agent.x <= 12
        assert 0 <= agent.y <= 7
    assert all(0 <= x <= 12 and 0 <= y <= 7 for x, y in agent.trajectory)


def test_leaving_direction_uses_actual_midpoint():
    env = domain.environment_from_dict(scene_data())
    persona = domain.get_default_personas()['通勤青年']
    agent = domain.Agent(id=1, persona=persona, x=10, y=3,
                         current_action=domain.Action.LEAVE)
    simulation.update_agent(agent, 10, env, None, 15)
    assert agent.x > env.width + 5


def test_imported_runtime_and_history_survive_source_removal(tmp_path, monkeypatch):
    data = scene_data()
    path = tmp_path / 'scene.yaml'
    path.write_text(yaml.safe_dump(data), encoding='utf-8')
    env = domain.load_scenario(str(path))
    personas = domain.get_default_personas()
    counts = {'买菜老人': 2}
    original = simulation.run_sim(env, personas, counts, duration=30, seed=42)
    assert original.source_import == data['source_import']
    assert 'source_import' not in asdict(original)
    monkeypatch.setattr(results, 'RUNS_DIR', tmp_path / 'runs')
    folder = Path(results.save_run_history(env, original, {'scenario_file': path.name}, personas, counts))
    path.unlink()
    restored, restored_env, _ = results.load_history_result(folder.name)
    assert restored_env.width == 12 and restored_env.height == 7
    assert [asdict(obj) for obj in restored_env.objects] == [asdict(obj) for obj in env.objects]
    assert restored_env.source_import == data['source_import']
    assert restored.source_import == data['source_import']
    assert json.loads(json.dumps(asdict(restored))) == json.loads(json.dumps(asdict(original)))
    fig = results.plot_trajectories(restored_env, restored)
    assert fig.axes[0].get_xlim() == (0, 12)
    assert fig.axes[0].get_ylim() == (0, 7)
    results._plot_modules()[0].close(fig)


def test_imported_metrics_exclude_offsite_points():
    env = domain.environment_from_dict(scene_data())
    agent = domain.Agent(id=1, persona=domain.get_default_personas()['买菜老人'], x=1, y=1)
    agent.trajectory = [(1, 1), (-20, 1), (30, 1)]
    metric = results._compute_spatial_metrics([agent], env)
    agent.trajectory = [(1, 1)]
    assert metric == results._compute_spatial_metrics([agent], env)


def test_scene_metadata_does_not_alias_input():
    data = scene_data()
    expected = copy.deepcopy(data['source_import'])
    env = domain.environment_from_dict(data)
    data['source_import']['origin'][0] = 0
    assert env.source_import == expected
