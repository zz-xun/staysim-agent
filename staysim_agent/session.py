"""Durable task sessions: SQLite checkpoints, user resumes and execution receipts."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import sqlite3
import time
from datetime import datetime, timezone
from threading import Lock
from uuid import UUID, uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from langchain_core.callbacks import BaseCallbackHandler

from .graph import build_simulation_graph
from .tools import run_experiment, plan_fingerprint, read_execution_receipt


DEFAULT_DB = Path(__file__).resolve().parents[1] / "output" / "agent" / "checkpoints.sqlite"


class UsageObserver(BaseCallbackHandler):
    """Count provider usage without retaining prompts, outputs or credentials."""

    def __init__(self):
        self.calls = 0
        self.failures = 0
        self.usage_missing = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self._lock = Lock()

    def on_chat_model_start(self, serialized, messages, **kwargs):
        with self._lock:
            self.calls += 1

    def on_llm_end(self, response, **kwargs):
        with self._lock:
            for generations in response.generations:
                usage = getattr(generations[0].message, "usage_metadata", None) if generations else None
                if usage:
                    self.input_tokens += usage.get("input_tokens", 0)
                    self.output_tokens += usage.get("output_tokens", 0)
                else:
                    self.usage_missing += 1

    def on_llm_error(self, error, **kwargs):
        with self._lock:
            self.failures += 1

    def summary(self):
        return {
            "model_calls": self.calls, "model_failures": self.failures,
            "usage_missing": self.usage_missing,
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "cost_estimate": None,
            "cost_note": "未配置模型价格；Token 为供应商返回值，不代表账单金额。",
        }


@contextmanager
def _exclusive_session(path: Path):
    """OS lock is released on process exit, including crashes; no stale lock flags."""
    with path.open("a+b") as stream:
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError("另一个 Agent 命令正在使用此数据库，请等待其完成。") from None
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class AgentSession:
    """One command at a time per local database; open a new session after restart."""

    def __init__(self, db_path=DEFAULT_DB, *, model=None):
        self.db_path = Path(db_path)
        self.model = model

    def __enter__(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._stack = ExitStack()
        try:
            self._stack.enter_context(_exclusive_session(self.db_path.with_suffix(".lock")))
            self.ledger = sqlite3.connect(str(self.db_path), timeout=30, isolation_level=None)
            self._stack.callback(self.ledger.close)
            self.ledger.execute("""CREATE TABLE IF NOT EXISTS executions (
                run_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, fingerprint TEXT NOT NULL,
                status TEXT NOT NULL, result_json TEXT, error TEXT
            )""")
            self.ledger.execute("""CREATE TABLE IF NOT EXISTS observations (
                task_id TEXT NOT NULL, operation TEXT NOT NULL,
                created_at TEXT NOT NULL, data_json TEXT NOT NULL
            )""")
            saver = self._stack.enter_context(SqliteSaver.from_conn_string(str(self.db_path)))
            saver.setup()
            self.graph = build_simulation_graph(self.model, checkpointer=saver, executor=self._execute_once)
            return self
        except BaseException:
            self._stack.close()
            raise

    def __exit__(self, *exc):
        return self._stack.__exit__(*exc)

    def _config(self, task_id):
        try:
            valid = str(UUID(task_id)) == task_id
        except (ValueError, TypeError, AttributeError):
            valid = False
        if not valid:
            raise ValueError("任务编号必须是完整 UUID")
        self.task_id = task_id
        return {"configurable": {"thread_id": task_id}}

    def start(self, *, user_input=None, request=None, selected_scenario=None):
        if (user_input is None) == (request is None):
            raise ValueError("只提供自然语言或结构化请求中的一种")
        task_id = str(uuid4())
        state = {"user_input": user_input} if user_input is not None else {"request": request}
        if selected_scenario:
            state["selected_scenario"] = selected_scenario
        self._invoke(state, task_id, "start")
        return self.show(task_id)

    def _invoke(self, state, task_id, operation):
        observer = UsageObserver()
        config = {**self._config(task_id), "callbacks": [observer]}
        started = time.perf_counter()
        completed = False
        try:
            self.graph.invoke(state, config)
            completed = True
        finally:
            observation = {
                **observer.summary(), "elapsed_seconds": round(time.perf_counter() - started, 4),
                "operation_returned": completed,
            }
            self.ledger.execute(
                "INSERT INTO observations VALUES (?,?,?,?)",
                (task_id, operation, datetime.now(timezone.utc).isoformat(),
                 json.dumps(observation, ensure_ascii=False)),
            )

    def show(self, task_id):
        snapshot = self.graph.get_state(self._config(task_id))
        if not snapshot.values:
            raise ValueError("没有找到此任务；旧版本终端输出不能作为检查点恢复")
        public = {key: value for key, value in snapshot.values.items() if key != "messages"}
        public["task_id"] = task_id
        public["pending"] = [item.value for task in snapshot.tasks for item in task.interrupts]
        public["next_nodes"] = list(snapshot.next)
        public["observability"] = [
            {"operation": operation, "created_at": created_at, **json.loads(data)}
            for operation, created_at, data in self.ledger.execute(
                "SELECT operation,created_at,data_json FROM observations WHERE task_id=? ORDER BY rowid",
                (task_id,),
            )
        ]
        if public.get("run_id"):
            row = self.ledger.execute(
                "SELECT status,error FROM executions WHERE run_id=?", (public["run_id"],),
            ).fetchone()
            public["execution"] = {"status": row[0], "error": row[1]} if row else {"status": "not_started"}
        return public

    def resume(self, task_id, *, action, text=None):
        state = self.show(task_id)
        if not state["pending"]:
            if state["next_nodes"]:
                raise ValueError("任务在节点执行中中断；请使用 --recover 检查恢复")
            return state  # A duplicate confirmation of a terminal task is a read.
        waiting = state["pending"][0]
        if action == "approve":
            if waiting["kind"] != "approval":
                raise ValueError("任务正在等待补充信息，不能直接确认")
            reply = {"action": "approve", "plan_id": waiting["plan_id"]}
        elif action == "reply":
            if not isinstance(text, str) or not text.strip():
                raise ValueError("请通过 --text 提供补充或修改内容")
            reply = {"action": "reply", "text": text}
        elif action == "cancel":
            reply = {"action": "cancel"}
        else:
            raise ValueError("未知恢复操作")
        self._invoke(Command(resume=reply), task_id, action)
        return self.show(task_id)

    def recover(self, task_id):
        state = self.show(task_id)
        if state["pending"] or not state["next_nodes"]:
            return state
        self._invoke(None, task_id, "recover")
        return self.show(task_id)

    def _execute_once(self, config, run_id, fingerprint):
        row = self.ledger.execute(
            "SELECT task_id,fingerprint,status FROM executions WHERE run_id=?", (run_id,),
        ).fetchone()
        if row:
            if row[:2] != (self.task_id, fingerprint):
                raise ValueError("运行编号不属于当前任务或配置")
            # After a crash, artifacts may be complete even if graph/ledger isn't.
            try:
                receipt = read_execution_receipt(run_id, fingerprint)
            except (ValueError, KeyError, OSError):
                self.ledger.execute(
                    "UPDATE executions SET status='uncertain',error=? WHERE run_id=?",
                    ("结果回执或文件校验失败，未自动重复执行。", run_id),
                )
                raise
            if receipt is not None:
                self._commit_result(run_id, receipt)
                return receipt
            self.ledger.execute(
                "UPDATE executions SET status='uncertain',error=? WHERE run_id=?",
                ("先前执行未留下完整回执，不能确认完成；未自动重复执行。", run_id),
            )
            raise RuntimeError("先前执行状态不确定")
        # Recheck mutable inputs after the user returns, before claiming execution.
        if plan_fingerprint(config) != fingerprint:
            self.ledger.execute(
                "INSERT INTO executions VALUES (?,?,?,?,?,?)",
                (run_id, self.task_id, fingerprint, "blocked", None, "场景或模型设置已改变，请创建新任务并重新确认。"),
            )
            raise ValueError("确认后场景或模型设置改变")
        self.ledger.execute(
            "INSERT INTO executions VALUES (?,?,?,?,?,?)",
            (run_id, self.task_id, fingerprint, "running", None, None),
        )
        try:
            result = run_experiment.invoke({"experiment_config": config, "run_id": run_id})
            persisted_validation = result.get("result_validation") or {}
            if persisted_validation.get("valid") is not True:
                message = "结果未通过执行后完整性校验，未标记为成功。"
                self.ledger.execute(
                    "UPDATE executions SET status='failed',error=? WHERE run_id=?",
                    (message, run_id),
                )
                raise RuntimeError(message)
        except Exception as exc:
            self.ledger.execute(
                "UPDATE executions SET status='failed',error=? WHERE run_id=?",
                (f"执行失败（{type(exc).__name__}），不会自动重跑。", run_id),
            )
            raise
        self._commit_result(run_id, result)
        return result

    def _commit_result(self, run_id, result):
        self.ledger.execute(
            "UPDATE executions SET status='completed',result_json=?,error=NULL WHERE run_id=?",
            (json.dumps(result, ensure_ascii=False), run_id),
        )
