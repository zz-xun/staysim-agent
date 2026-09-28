"""Rule and LLM pedestrian decisions; movement is owned by simulation.py."""

import json
import math
import random
import re

import requests

from .domain import Action, Decision, SpaceElement

class CognitiveEngine:
    """认知决策引擎 —— 支持规则模式和 LLM 模式"""

    def __init__(self, mode="rule", api_key=None, api_base=None, model=None, seed=None):
        self.mode = mode          # "rule" 或 "llm"
        self.api_key = api_key
        self.api_base = api_base  # API 地址（DeepSeek: https://api.deepseek.com）
        self.model = model        # 模型名（gpt-4o-mini / deepseek-chat）
        self.llm_call_count = 0   # 统计 LLM 调用次数
        self.total_tokens = 0     # 统计 token 消耗
        self.llm_failure_count = 0
        self.fallback_count = 0
        self.fallback_reasons = []
        self._cache = {}          # LLM 结果缓存：key → Decision，不跨模拟
        self.llm_timeout = 45
        self.llm_retries = 2
        if seed is not None:
            random.seed(seed)

    def clear_cache(self):
        """清空 LLM 缓存（每次模拟结束后调用）"""
        self._cache.clear()

    def _record_llm_failure(self, reason):
        """Record a degraded LLM decision without exposing credentials or payloads."""
        self.llm_failure_count += 1
        self.fallback_count += 1
        reason = str(reason).replace("\n", " ").strip()
        if reason and reason not in self.fallback_reasons and len(self.fallback_reasons) < 20:
            self.fallback_reasons.append(reason[:160])

    def _llm_failed_decision(self, pos, reason):
        self._record_llm_failure(reason)
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(3, 8)
        return Decision(
            action=Action.WANDER,
            target_pos=(pos[0] + distance * math.cos(angle), pos[1] + distance * math.sin(angle)),
            reason=f"LLM失败: {reason}; 保底随机游走",
        )

    def _llm_failed_rule_decision(self, pos, energy, mood, persona, nearby, time_of_day, reason):
        self._record_llm_failure(reason)
        decision = self._rule_decide(pos, energy, mood, persona, nearby, time_of_day)
        decision.reason = f"LLM失败: {reason}; 使用规则兜底 -> {decision.reason}"
        return decision

    def _parse_llm_json(self, text):
        text = (text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        decoder = json.JSONDecoder()
        for idx, char in enumerate(text):
            if char != "{":
                continue
            try:
                data, _ = decoder.raw_decode(text[idx:])
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                continue
        raise json.JSONDecodeError("No valid JSON object found", text, 0)

    def _local_failed_decision(self, pos, reason):
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(3, 8)
        return Decision(
            action=Action.WANDER,
            target_pos=(pos[0] + distance * math.cos(angle), pos[1] + distance * math.sin(angle)),
            reason=f"本地模型失败: {reason}; 保底随机游走",
        )

    def decide(self, pos, energy, mood, persona, environment, time_of_day, all_agents=None):
        nearby = environment.get_nearby(*pos, 35.0)

        if self.mode == "rule":
            return self._rule_decide(pos, energy, mood, persona, nearby, time_of_day, all_agents)
        elif self.mode == "local":
            return self._local_decide(pos, energy, mood, persona, nearby, time_of_day)
        elif self.mode == "llm":
            # 先查缓存：相同 persona + 近似位置 + 相似状态 → 复用决策
            cache_key = (
                persona.name,
                round(pos[0], -1),      # 四舍五入到10m
                round(pos[1], -1),
                int(energy * 10),       # 能量离散化为 0-10
                int(mood * 10),         # 心情离散化为 0-10
            )
            if cache_key in self._cache:
                return self._cache[cache_key]
            # 缓存未命中 → 调用 LLM 并缓存结果
            decision = self._llm_decide(pos, energy, mood, persona, environment, nearby, time_of_day)
            self._cache[cache_key] = decision
            return decision
        else:
            return self._rule_decide(pos, energy, mood, persona, nearby, time_of_day)

    def describe_surroundings(self, pos, nearby, time_of_day):
        """将空间信息转换为自然语言，供 LLM 理解（增强感官维度）"""
        hour = int(time_of_day)
        minute = int((time_of_day % 1) * 60)
        time_str = f"{hour}:{minute:02d}"

        # 感知半径因设施类型而异：视觉可达15m，听觉5m（棋桌/社交设施需要近）
        _large_rad = 25.0  # 可见范围
        _small_rad = 8.0   # 可听/可闻范围

        visible = []
        audible = []
        for obj in nearby:
            dist = math.sqrt((pos[0]-obj.x)**2 + (pos[1]-obj.y)**2)
            if dist <= _large_rad:
                visible.append((obj, dist))
            if dist <= _small_rad and obj.attractiveness > 0.5:
                audible.append((obj, dist))

        if not visible and not audible:
            return f"当前时间{time_str}。周围是一片空地，没有特别值得停留的东西。"

        lines = [f"当前时间{time_str}。"]
        if visible:
            lines.append("你目光所及可以看到：")
            for i, (obj, dist) in enumerate(visible[:6]):
                shade = "浓密树荫下" if obj.shade else "阳光下"
                attr = "很吸引人" if obj.attractiveness > 0.6 else "一般" if obj.attractiveness > 0.3 else "不太起眼"
                lines.append(f"  {i+1}. {obj.element.value}（{obj.label or ''}），在{shade}，{dist:.0f}米外，看起来{attr}")
        if audible:
            lines.append("你隐约听到/注意到：")
            for obj, dist in audible[:3]:
                lines.append(f"  · {obj.label or obj.element.value}方向有动静（{dist:.0f}米外）")
        return "\n".join(lines)

    def _local_decide(self, pos, energy, mood, persona, nearby, time_of_day):
        """本地模型决策 —— 调用 Ollama API (http://localhost:11434/api/generate)"""
        space_desc = self.describe_surroundings(pos, nearby, time_of_day)
        energy_str = "精力充沛" if energy > 0.7 else "有点累了" if energy > 0.3 else "非常疲惫"

        prompt = f"""你是一个{persona.name}（{persona.age_range[0]}-{persona.age_range[1]}岁，{persona.description}），当前{energy_str}。
    请根据周围环境做出真实行为决策。严格只输出JSON：
    {{"action":"stay|walk_to|wander|leave","target_index":数字,"stay_minutes":数字,"thought":"简短想法"}}

    {space_desc}

    作为{persona.name}，你会怎么做？JSON:"""

        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": self.model or "llama3", "prompt": prompt, "stream": False},
                timeout=30,
            )
            self.llm_call_count += 1

            if resp.status_code != 200:
                print(f"  ⚠️ Ollama API 错误 (HTTP {resp.status_code})")
                return self._local_failed_decision(pos, f"HTTP {resp.status_code}")

            result = resp.json()
            text = result.get("response", "").strip()

            # 解析 JSON
            try:
                data = self._parse_llm_json(text)
            except json.JSONDecodeError:
                import re
                match = re.search(r'\{[^{}]*\}', text)
                if match:
                    data = json.loads(match.group())
                else:
                    return self._local_failed_decision(pos, "返回内容不是JSON")

            action_str = data.get("action", "wander")
            action_map = {
                "stay": Action.STAY, "walk_to": Action.WALK_TO,
                "wander": Action.WANDER, "leave": Action.LEAVE
            }
            action = action_map.get(action_str, Action.WANDER)

            target_obj = None
            target_idx = data.get("target_index", -1)
            if isinstance(target_idx, int) and 1 <= target_idx <= len(nearby):
                target_obj = nearby[target_idx - 1]

            stay_minutes = data.get("stay_minutes", 5)
            if not isinstance(stay_minutes, (int, float)):
                stay_minutes = 5
            stay_seconds = min(stay_minutes * 60, 1800)

            thought = data.get("thought", "本地模型决策")

            if action == Action.WALK_TO and target_obj:
                return Decision(action=Action.WALK_TO, target=target_obj,
                    target_pos=(target_obj.x, target_obj.y),
                    reason=f"🖥本地: {thought}")
            elif action == Action.STAY:
                for obj in nearby:
                    if math.sqrt((pos[0]-obj.x)**2 + (pos[1]-obj.y)**2) < 3.0:
                        target_obj = obj
                        break
                return Decision(action=Action.STAY, target=target_obj,
                    stay_duration=stay_seconds,
                    reason=f"🖥本地: {thought}")
            elif action == Action.LEAVE:
                return Decision(action=Action.LEAVE, reason=f"🖥本地: {thought}")
            else:
                angle = random.uniform(0, 2*math.pi)
                return Decision(action=Action.WANDER,
                    target_pos=(pos[0]+random.uniform(3,10)*math.cos(angle),
                               pos[1]+random.uniform(3,10)*math.sin(angle)),
                    reason=f"🖥本地: {thought}")

        except requests.exceptions.Timeout:
            print(f"  ⚠️ Ollama 超时，记录为本地模型失败")
            return self._local_failed_decision(pos, "API超时")
        except Exception as e:
            print(f"  ⚠️ Ollama 异常({e})，记录为本地模型失败")
            return self._local_failed_decision(pos, str(e))

    def _llm_decide(self, pos, energy, mood, persona, environment, nearby, time_of_day):
        """LLM 驱动的决策 —— 调用大语言模型做认知判断"""

        if not self.api_key:
            # 无 API key → 记录为 LLM 失败，避免伪装成规则决策
            return self._llm_failed_decision(pos, "未设置API Key")

        # 构建 Prompt
        space_desc = self.describe_surroundings(pos, nearby, time_of_day)
        energy_str = "精力充沛" if energy > 0.7 else "有点累了" if energy > 0.3 else "非常疲惫"
        mood_str = "心情很好" if mood > 0.7 else "心情一般" if mood > 0.5 else "心情不太好"

        system_prompt = f"""你是一个{persona.name}（{persona.age_range[0]}-{persona.age_range[1]}岁，{persona.description}），当前{energy_str}。请按三步推理：①观察-你看到了什么 ②判断-这对你意味着什么 ③行动-你决定怎么做。严格只输出JSON：

{{"step1_observe":"看到了什么关键信息","step2_judge":"这对你意味着什么","action":"stay|walk_to|wander|leave","target_index":数字,"stay_minutes":数字,"thought":"简短想法"}}"""

        user_prompt = f"""{space_desc}

作为{persona.name}，你会怎么做？JSON:"""

        try:
            # 构建 API 请求
            api_base = self.api_base or "https://api.openai.com/v1"
            url = f"{api_base}/chat/completions"

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model or "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 220,
            }

            resp = None
            for attempt in range(self.llm_retries + 1):
                try:
                    resp = requests.post(url, headers=headers, json=payload, timeout=self.llm_timeout)
                    self.llm_call_count += 1
                    break
                except requests.exceptions.Timeout:
                    print(f"  LLM API timeout, retry {attempt + 1}/{self.llm_retries + 1}")
            if resp is None:
                return self._llm_failed_rule_decision(
                    pos, energy, mood, persona, nearby, time_of_day,
                    f"API超时 {self.llm_timeout}s x {self.llm_retries + 1}",
                )

            if resp.status_code != 200:
                print(f"  ⚠️ LLM API 错误 (HTTP {resp.status_code}): {resp.text[:100]}")
                return self._llm_failed_rule_decision(
                    pos, energy, mood, persona, nearby, time_of_day,
                    f"HTTP {resp.status_code}",
                )

            result = resp.json()

            # 统计 token
            usage = result.get('usage', {})
            self.total_tokens += usage.get('total_tokens', 0)

            text = result['choices'][0]['message']['content'].strip()

            # 解析 JSON
            try:
                data = self._parse_llm_json(text)
            except json.JSONDecodeError:
                # 尝试提取 JSON 块
                import re
                match = re.search(r'\{[^{}]*\}', text)
                if match:
                    data = json.loads(match.group())
                else:
                    return self._llm_failed_rule_decision(
                        pos, energy, mood, persona, nearby, time_of_day,
                        f"返回内容不是JSON: {text[:80]}",
                    )

            # 映射行动
            action_str = data.get("action", "wander")
            action_map = {
                "stay": Action.STAY, "walk_to": Action.WALK_TO,
                "wander": Action.WANDER, "leave": Action.LEAVE
            }
            action = action_map.get(action_str, Action.WANDER)

            # 找目标设施
            target_obj = None
            target_idx = data.get("target_index", -1)
            if isinstance(target_idx, int) and 1 <= target_idx <= len(nearby):
                target_obj = nearby[target_idx - 1]

            stay_minutes = data.get("stay_minutes", 5)
            if not isinstance(stay_minutes, (int, float)):
                stay_minutes = 5
            stay_seconds = min(stay_minutes * 60, 1800)  # 最多30分钟

            thought = data.get("thought", "LLM决策")
            reasoning = f"观察:{data.get('step1_observe','')} | 判断:{data.get('step2_judge','')}"

            if action == Action.WALK_TO and target_obj:
                return Decision(action=Action.WALK_TO, target=target_obj,
                    target_pos=(target_obj.x, target_obj.y),
                    reason=f"🧠LLM: {thought} [{reasoning}]")
            elif action == Action.STAY:
                for obj in nearby:
                    if math.sqrt((pos[0]-obj.x)**2 + (pos[1]-obj.y)**2) < 3.0:
                        target_obj = obj
                        break
                if target_obj is None:
                    # 附近没有可停留的设施 → 改为漫游探索
                    angle = random.uniform(0, 2*math.pi)
                    return Decision(action=Action.WANDER,
                        target_pos=(pos[0]+random.uniform(8,20)*math.cos(angle),
                                   pos[1]+random.uniform(8,20)*math.sin(angle)),
                        reason=f"🧠LLM: 想歇但附近没设施，四处看看 [{reasoning}]")
                return Decision(action=Action.STAY, target=target_obj,
                    stay_duration=stay_seconds,
                    reason=f"🧠LLM: {thought} [{reasoning}]")
            elif action == Action.LEAVE:
                return Decision(action=Action.LEAVE, reason=f"🧠LLM: {thought} [{reasoning}]")
            else:
                angle = random.uniform(0, 2*math.pi)
                return Decision(action=Action.WANDER,
                    target_pos=(pos[0]+random.uniform(3,10)*math.cos(angle),
                               pos[1]+random.uniform(3,10)*math.sin(angle)),
                    reason=f"🧠LLM: {thought} [{reasoning}]")

        except requests.exceptions.Timeout:
            print(f"  ⚠️ LLM API 超时，记录为 LLM失败")
            return self._llm_failed_rule_decision(pos, energy, mood, persona, nearby, time_of_day, "API超时")
        except Exception as e:
            print(f"  ⚠️ LLM 调用异常({e})，记录为 LLM失败")
            return self._llm_failed_rule_decision(pos, energy, mood, persona, nearby, time_of_day, str(e))

    def _rule_decide(self, pos, energy, mood, persona, nearby, time_of_day, all_agents=None):
        is_tired = energy < 0.3

        # ── 社交互动：计算周围5m内同类型智能体数量 ──
        same_type_nearby = 0
        if all_agents:
            for other in all_agents:
                if other.persona.name == persona.name:
                    dist = math.sqrt((pos[0] - other.x)**2 + (pos[1] - other.y)**2)
                    if dist < 5.0:
                        same_type_nearby += 1

        best_target, best_score = None, -1

        for obj in nearby:
            dist = math.sqrt((pos[0]-obj.x)**2 + (pos[1]-obj.y)**2)
            if dist < 1.0: continue
            interest = persona.activity_interests.get(obj.element, 0.3)
            dist_factor = max(0, 1 - dist/20.0)
            fatigue_bonus = 0.3 if (is_tired and obj.element in [SpaceElement.BENCH, SpaceElement.PAVILION]) else 0
            social_bonus = 0.1 * persona.social_preference * obj.attractiveness
            # 社交互动加成：社交倾向>0.6 且 附近≥2个同类 → +0.15
            social_attraction = 0.0
            if persona.social_preference > 0.6 and same_type_nearby >= 2:
                social_attraction = 0.15
            score = (interest*0.5 + dist_factor*0.2 + obj.attractiveness*0.15
                    + fatigue_bonus + social_bonus + social_attraction)
            if score > best_score:
                best_score, best_target = score, obj

        for obj in nearby:
            if math.sqrt((pos[0]-obj.x)**2 + (pos[1]-obj.y)**2) < 2.0:
                interest = persona.activity_interests.get(obj.element, 0.3)
                if interest > 0.5 and persona.stay_preference > 0.3:
                    stay_time = random.uniform(5, 30) * persona.stay_preference * interest
                    social_note = ""
                    if persona.social_preference > 0.6 and same_type_nearby >= 2:
                        social_note = f"，看到{same_type_nearby}个同类在此"
                    return Decision(Action.STAY, target=obj,
                        target_pos=(obj.x+random.uniform(-1,1), obj.y+random.uniform(-1,1)),
                        stay_duration=stay_time,
                        reason=f"⚡规则: 到达{obj.element.value}，{persona.name}停留{stay_time:.0f}分钟{social_note}")

        if best_target and best_score > 0.3:
            social_note = ""
            if persona.social_preference > 0.6 and same_type_nearby >= 2:
                social_note = f"，社交引力+0.15"
            return Decision(Action.WALK_TO, target=best_target,
                target_pos=(best_target.x, best_target.y),
                reason=f"⚡规则: 被{best_target.element.value}吸引(得分{best_score:.2f}{social_note})")

        if is_tired and persona.stay_preference > 0.5:
            return Decision(Action.WANDER, reason="⚡规则: 累了，找地方坐")

        if persona.stay_preference < 0.3 and random.random() < 0.4:
            return Decision(Action.LEAVE, reason="⚡规则: 没什么兴趣，离开")

        angle = random.uniform(0, 2*math.pi)
        return Decision(Action.WANDER,
            target_pos=(pos[0]+random.uniform(3,10)*math.cos(angle),
                       pos[1]+random.uniform(3,10)*math.sin(angle)),
            reason="⚡规则: 随便逛逛")
