"""Simulation time loop and movement. No UI or Agent framework imports."""

import logging
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from .cognition import CognitiveEngine
from .domain import Action, Agent, PersonaProfile, SimResult, SpaceElement, SpaceObject
from .results import _compute_spatial_metrics

_logger = logging.getLogger("StaySim")

def update_agent(agent, dt, environment, engine, time_of_day):
    # Imported scenes opt into boundary enforcement. Legacy scenes retain their
    # established movement behavior so existing seeded experiments stay comparable.
    _constrain_imported_agent(agent, environment)
    _update_agent(agent, dt, environment, engine, time_of_day)
    _constrain_imported_agent(agent, environment)


def _constrain_imported_agent(agent, environment):
    if hasattr(environment, 'source_import') and agent.current_action != Action.LEAVE:
        agent.x = max(0, min(environment.width, agent.x))
        agent.y = max(0, min(environment.height, agent.y))


def _update_agent(agent, dt, environment, engine, time_of_day):
    agent.trajectory.append((agent.x, agent.y))
    agent.energy = max(0, agent.energy - 0.002 * dt)

    if agent.current_action == Action.STAY:
        agent.stay_remaining -= dt
        if agent.stay_remaining <= 0:
            agent.energy = min(1.0, agent.energy + 0.15)
            agent.stay_points.append((agent.x, agent.y, dt))
            d = engine.decide((agent.x, agent.y), agent.energy, agent.mood,
                            agent.persona, environment, time_of_day)
            apply_decision(agent, d)
        else:
            agent.x += random.uniform(-0.02, 0.02)
            agent.y += random.uniform(-0.02, 0.02)
        return

    if agent.current_action == Action.WALK_TO and agent.current_target:
        tx, ty = agent.current_target.x, agent.current_target.y
        dx, dy = tx - agent.x, ty - agent.y
        dist = math.sqrt(dx**2 + dy**2)
        if dist < 1.5:
            d = engine.decide((agent.x, agent.y), agent.energy, agent.mood,
                            agent.persona, environment, time_of_day)
            apply_decision(agent, d)
        else:
            speed = agent.persona.speed
            agent.x += (dx/dist) * speed * dt
            agent.y += (dy/dist) * speed * dt
        return

    if agent.current_action == Action.WANDER:
        if agent.current_target:
            tx, ty = agent.current_target.x, agent.current_target.y
            dx, dy = tx - agent.x, ty - agent.y
            if math.sqrt(dx**2+dy**2) < 2.0:
                d = engine.decide((agent.x, agent.y), agent.energy, agent.mood,
                                agent.persona, environment, time_of_day)
                apply_decision(agent, d)
            else:
                s = agent.persona.speed * 0.6
                agent.x += (dx/math.sqrt(dx**2+dy**2)) * s * dt
                agent.y += (dy/math.sqrt(dx**2+dy**2)) * s * dt
        else:
            angle = random.uniform(0, 2*math.pi)
            agent.x += math.cos(angle) * agent.persona.speed * 0.5 * dt
            agent.y += math.sin(angle) * agent.persona.speed * 0.5 * dt
        return

    if agent.current_action == Action.LEAVE:
        if agent.x < environment.width / 2: agent.x -= agent.persona.speed * dt
        else: agent.x += agent.persona.speed * dt


def apply_decision(agent, decision):
    agent.current_action = decision.action
    agent.current_target = decision.target
    agent.stay_remaining = decision.stay_duration
    agent.decisions_log.append(decision.reason)
    if decision.action == Action.STAY:
        target_label = ""
        target_type = ""
        if decision.target:
            target_label = decision.target.label or decision.target.element.value
            target_type = decision.target.element.name
        agent.stay_events.append({
            "x": round(agent.x, 3),
            "y": round(agent.y, 3),
            "duration": round(float(decision.stay_duration or 0), 3),
            "target_label": target_label,
            "target_type": target_type,
            "reason": decision.reason,
        })


def run_sim(env, personas_dict, counts_dict, duration=600, dt=2.0, seed=None,
            engine_mode="rule", api_key=None, api_base=None, model=None, progress=None):
    """运行模拟，返回 SimResult（支持进度回调 progress(ratio, msg)）"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    engine = CognitiveEngine(mode=engine_mode, api_key=api_key, api_base=api_base,
                             model=model, seed=seed)
    # 日志：记录运行开始
    total_agents = sum(counts_dict.values())
    _logger.info(
        f"模拟开始 | 场景={env.name} | 人数={total_agents} | 时长={duration}s | "
        f"引擎={engine_mode} | 配比={counts_dict}"
    )
    # ── 从 YAML 读取入口位置，智能体从入口±3m范围内生成 ──
    entrances = [obj for obj in env.objects if obj.element == SpaceElement.ENTRANCE]
    if not entrances:
        # 场景无入口定义 → 默认左下角区域
        entrances = [SpaceObject(x=env.width * 0.1, y=env.height * 0.5,
                                radius=3, element=SpaceElement.ENTRANCE, attractiveness=0)]

    agents = []
    agent_id = 0
    for pname, persona in personas_dict.items():
        count = counts_dict.get(pname, 0)
        for _ in range(count):
            # ── Persona 个体差异：activity_interests 加 ±0.1 随机噪声 ──
            perturbed_interests = {}
            for elem, val in persona.activity_interests.items():
                noise = random.uniform(-0.1, 0.1)
                perturbed_interests[elem] = max(0.0, min(1.0, val + noise))
            individual_persona = PersonaProfile(
                name=persona.name,
                age_range=persona.age_range,
                speed=persona.speed,
                stay_preference=persona.stay_preference,
                social_preference=persona.social_preference,
                activity_interests=perturbed_interests,
                description=persona.description,
            )

            entrance = random.choice(entrances)
            x = entrance.x + random.uniform(-3, 3)
            y = entrance.y + random.uniform(-3, 3)
            # 确保不超出场景边界
            x_margin, y_margin = min(0.5, env.width / 2), min(0.5, env.height / 2)
            x = max(x_margin, min(env.width - x_margin, x))
            y = max(y_margin, min(env.height - y_margin, y))
            agents.append(Agent(id=agent_id, persona=individual_persona, x=x, y=y,
                              energy=random.uniform(0.3, 1.0), mood=random.uniform(0.5, 1.0)))
            agent_id += 1

    n_steps = int(duration / dt)
    time_of_day = 15.0
    start = time.time()

    # LLM 模式下降低决策频率（每30步=60秒才决策一次，减少API调用）
    decide_interval = 30 if engine_mode == "llm" else 10
    total_decide_rounds = n_steps // decide_interval
    decide_round = 0

    for step in range(n_steps):
        # 总体进度：运动更新占70%，决策占30%
        base_progress = step / n_steps * 0.7

        sim_time = step * dt
        time_of_day = 15.0 + sim_time / 3600

        # 需要决策的智能体
        deciding_agents = []
        for agent in agents:
            if agent.current_action == Action.LEAVE and (agent.x < -5 or agent.x > env.width + 5):
                continue
            if step % decide_interval == 0 and agent.current_action != Action.STAY:
                deciding_agents.append(agent)

        # LLM 模式：并发调用（加速）
        if deciding_agents and engine_mode == "llm" and engine.api_key:
            decide_round += 1
            msg = (f"🧠 LLM决策轮 {decide_round}/{total_decide_rounds} "
                   f"({len(deciding_agents)}个智能体并发)")
            print(f"  {msg}", flush=True)
            if progress:
                progress(base_progress + 0.05, msg)

            def decide_one(agent):
                d = engine.decide((agent.x, agent.y), agent.energy, agent.mood,
                                  agent.persona, env, time_of_day, all_agents=agents)
                return agent, d

            completed = 0
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(decide_one, a) for a in deciding_agents]
                wait_timeout = max(180, (engine.llm_timeout * (engine.llm_retries + 1) + 15) * math.ceil(len(futures) / 8))
                for future in as_completed(futures, timeout=wait_timeout):
                    try:
                        agent, d = future.result()
                        apply_decision(agent, d)
                        completed += 1
                        if progress and completed % 5 == 0:
                            sub_ratio = base_progress + 0.05 + 0.25 * (completed / len(deciding_agents))
                            progress(sub_ratio, f"{msg} — {completed}/{len(deciding_agents)} 完成")
                    except Exception as e:
                        completed += 1

            print(f"    ✅ 完成 (累计LLM调用: {engine.llm_call_count}, "
                  f"Token: {engine.total_tokens})", flush=True)

        elif deciding_agents:
            # 规则模式：直接串行（本来就快）
            for agent in deciding_agents:
                d = engine.decide((agent.x, agent.y), agent.energy, agent.mood,
                                  agent.persona, env, time_of_day, all_agents=agents)
                apply_decision(agent, d)

        # 所有智能体运动更新
        for agent in agents:
            if agent.current_action == Action.LEAVE and (agent.x < -5 or agent.x > env.width + 5):
                continue
            update_agent(agent, dt, env, engine, time_of_day)

        # ── 社会力排斥：智能体之间保持距离 ──
        active_agents = [a for a in agents
                        if not (a.current_action == Action.LEAVE and (a.x < -5 or a.x > env.width + 5))]
        for i, a1 in enumerate(active_agents):
            for a2 in active_agents[i+1:]:
                dx = a1.x - a2.x
                dy = a1.y - a2.y
                dist = math.sqrt(dx**2 + dy**2)
                if dist < 2.0 and dist > 0.01:  # 2m内产生排斥
                    # 排斥力随距离减小而增大
                    force = (2.0 - dist) / 2.0 * 0.1 * dt
                    nx = dx / dist
                    ny = dy / dist
                    a1.x += nx * force
                    a1.y += ny * force
                    a2.x -= nx * force
                    a2.y -= ny * force
                    # 不推出边界
                    if hasattr(env, 'source_import'):
                        _constrain_imported_agent(a1, env)
                        _constrain_imported_agent(a2, env)
                    else:
                        a1.x = max(0.5, min(env.width - 0.5, a1.x))
                        a1.y = max(0.5, min(env.height - 0.5, a1.y))
                        a2.x = max(0.5, min(env.width - 0.5, a2.x))
                        a2.y = max(0.5, min(env.height - 0.5, a2.y))

        if progress and step % max(1, n_steps // 20) == 0:
            progress(base_progress, f"⏱ 模拟进行中... {step*dt:.0f}s / {duration:.0f}s")

    elapsed = time.time() - start

    # ── 计算量化指标 ──
    metrics = _compute_spatial_metrics(agents, env)

    # 汇总
    effective_engine_mode = engine_mode
    if engine_mode == "llm" and engine.fallback_count:
        effective_engine_mode = "llm_with_fallback"
    result = SimResult(env_name=env.name, n_agents=len(agents), elapsed=elapsed,
                       engine_mode=engine_mode, llm_calls=engine.llm_call_count,
                       total_tokens=engine.total_tokens,
                       spatial_entropy=metrics['entropy'],
                       facility_utilization=metrics['facility_util'],
                       cold_zone_ratio=metrics['cold_ratio'],
                       stay_pass_ratio=metrics['stay_pass_ratio'],
                       mean_stay_duration=metrics.get('mean_stay', 0),
                       total_stay_events=metrics.get('total_stay_events', 0),
                       top_facility=metrics.get('top_facility', ""))
    # Keep legacy SimResult serialization stable while attaching audit metadata.
    result.requested_engine_mode = engine_mode
    result.effective_engine_mode = effective_engine_mode
    result.llm_failure_count = engine.llm_failure_count
    result.fallback_count = engine.fallback_count
    result.fallback_reasons = list(engine.fallback_reasons)
    result.simulation_quality = "degraded" if engine.fallback_count else "ok"
    if hasattr(env, 'source_import'):
        from copy import deepcopy
        result.source_import = deepcopy(env.source_import)

    # 清空 LLM 缓存（不跨模拟保留）
    engine.clear_cache()

    for agent in agents:
        pname = agent.persona.name
        result.trajectories.setdefault(pname, []).append(agent.trajectory)
        result.decision_logs.setdefault(pname, []).extend(agent.decisions_log[-10:])
        result.stay_events.setdefault(pname, []).extend(agent.stay_events)
        if agent.current_action == Action.STAY and agent.current_target:
            result.stay_points.setdefault(pname, []).append((agent.x, agent.y, agent.stay_remaining))
        result.final_positions.append((agent.x, agent.y, pname))

    # 日志：记录运行完成
    cost_est = result.total_tokens / 1000000 * 0.15 if result.total_tokens > 0 else 0
    _logger.info(
        f"模拟完成 | 耗时={elapsed:.2f}s | Token={result.total_tokens} | "
        f"LLM调用={result.llm_calls} | 费用≈¥{cost_est*7.2:.3f}"
    )

    return result
