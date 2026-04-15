import asyncio

import uvicorn, os
from typing import Union
from fastapi import FastAPI, UploadFile, Depends, BackgroundTasks, Request,HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from knowledge.core.paths import get_front_page_dir
from knowledge.schema.query_schema import QueryRequest, StreamSubmitResponse, QueryResponse
from knowledge.core.deps import get_query_service
from knowledge.service.query_service import QueryService
from knowledge.utils.sse_util import create_sse_queue, sse_generator


def create_app():
    app = FastAPI(title="掌柜智库的查询应用", version="v1.0")
    # 配置跨域
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    front_page_dir = get_front_page_dir()
    app.mount("/front", StaticFiles(directory=front_page_dir))

    register_router(app)

    return app

def register_router(app: FastAPI):

    @app.get('/')
    def hello_world():
        return {"flag": "success"}

    @app.post('/query')
    async def query_endpoint(request: QueryRequest, background_tasks: BackgroundTasks, service: QueryService = Depends(get_query_service)) -> Union[StreamSubmitResponse, QueryResponse]:
        """
        处理查询请求
        Args:
            query: 前端请求参数对象
            background_tasks: 后台任务对象
            service: 查询业务组件对象
        Returns:
            流式返回或者正常返回值对象
        """
        session_id = request.session_id or service.generate_session_id()

        task_id = service.generate_task_id()
        print(request.is_stream)
        #流式调用
        if request.is_stream:
            #生成sse队列
            create_sse_queue(task_id)
            #定义后台任务
            background_tasks.add_task(service.run_query_graph, task_id=task_id, session_id=session_id, query=request.query, is_stream=True)
            return StreamSubmitResponse(message='查询请求已经提交', session_id=session_id, task_id=task_id)
        else:
            #同步调用,使用asyncio的event_loop阻塞当前线程
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                service.run_query_graph,
                session_id,
                task_id,
                request.query,
                request.is_stream
            )
            #执行完成后获取answer
            answer = service.get_task_result(task_id)
            return QueryResponse(message='查询请求已经处理完了', session_id=session_id, answer=answer)

    @app.get("/stream/{task_id}")
    async def stream(task_id: str, request: Request) -> StreamingResponse:
        """
        返回sse协议要的数据包：流式+yield使用 最佳搭配
        利用生成器+yield直接返回
        返回包格式"event:自定义\ndata:自定义\n\n"
        Args:
            task_id: 任务id
            request: 前端请求体
        Returns:
            StreamingResponse: 将后端组合的sse协议格式的数据 返回给前端
        """
        return StreamingResponse(content=sse_generator(task_id, request), media_type="text/event-stream")

    @app.get("/history/{session_id}")
    async def get_history(
            session_id: str, limit: int = 50,
            service: QueryService = Depends(get_query_service),
    ):
        try:
            items = service.get_history(session_id, limit)
            return {"session_id": session_id, "items": items}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"history error: {e}")

    @app.delete("/history/{session_id}")
    async def clear_chat_history(
            session_id: str,
            service: QueryService = Depends(get_query_service),
    ):
        count = service.clear_history(session_id)
        return {"message": "History cleared", "deleted_count": count}

    @app.get("/sessions")
    async def list_sessions(
            service: QueryService = Depends(get_query_service),
    ):
        sessions = service.list_sessions()
        return {"sessions": sessions}

    @app.delete("/session/{session_id}")
    async def delete_session(
            session_id: str,
            service: QueryService = Depends(get_query_service),
    ):
        count = service.delete_session(session_id)
        return {"message": "Session deleted", "deleted_count": count}

if __name__ == '__main__':
    uvicorn.run(create_app(), host="0.0.0.0", port=8001)