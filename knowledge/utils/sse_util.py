import asyncio
import json
import logging
import queue
from typing import Dict, Any, Optional, AsyncGenerator
from fastapi import Request



class SSEEvent:
    PROGRESS = "progress"   # 任务节点进度
    DELTA = "delta"         # LLM 流式输出增量
    FINAL = "final"         # 最终完整答案


# 全局 SSE 任务队列存储
# Key: task_id, Value: queue.Queue
_task_stream: Dict[str, queue.Queue] = {}


def get_sse_queue(task_id: str) -> Optional[queue.Queue]:
    """获取指定任务的队列"""
    return _task_stream.get(task_id)


def create_sse_queue(task_id: str) -> queue.Queue:
    """创建并注册一个新的 SSE 队列"""
    q = queue.Queue()
    _task_stream[task_id] = q
    return q

def remove_sse_queue(task_id: str):
    """移除指定任务的队列
    不存在 key 默认返回 None
    """
    _task_stream.pop(task_id, None)


def _sse_pack(event: str, data: Dict[str, Any]) -> str:
    """打包 SSE 消息格式"""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def push_sse_event (task_id: str, event: str, data: Dict[str, Any]):
    """
    通过 task_id 推送事件到 SSE 队列
    """
    # 1. 获取 SSE 队列
    stream_queue = get_sse_queue(task_id)

    # 2. 队列存在
    if stream_queue:
        # 3. 将事件推送到队列
        stream_queue.put({"event": event, "data": data})

async def sse_generator(task_id: str, request: Request) -> AsyncGenerator:
    """
    消费sse队列中数据,取出东西封装成包yield出去
    Args:
        task_id: 任务id
    Returns:
        AsyncGenerator:异步生成器对象
    """
    sse_queue = _task_stream.get(task_id)
    if sse_queue is None:
        return
    loop = asyncio.get_event_loop()
    try:
        while True:
            #判断当前会话是否还在连接状态
            if await request.is_disconnected():
                return
            #run_in_executor阻塞当前线程池中函数的运行
            #从队列中获取数据,阻塞等待,等待时间是1秒
            try:
                msg = await loop.run_in_executor(None, sse_queue.get, True, 1)
                event_type = msg.get('event')
                event_data = msg.get('data')
                #打包成字符串通过yield返回
                #yield,暂停函数将值返回给调用者
                yield _sse_pack(event_type, event_data)
            except queue.Empty:
                logging.info(f"队列为空...请稍等")
                continue
    except (ConnectionResetError, BrokenPipeError) as e:
        return
    except asyncio.CancelledError:
        # 服务端中断 协程被取消 重新抛出，让外层知道它被成功取消()
        raise
    finally:
        #移除任务队列
        remove_sse_queue(task_id)

