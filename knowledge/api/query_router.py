import asyncio
import json
import uvicorn
from typing import Union
from fastapi import FastAPI, Depends, BackgroundTasks, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from knowledge.core.paths import get_front_page_dir
from knowledge.schema.query_schema import QueryRequest, StreamSubmitResponse, QueryResponse
from knowledge.core.deps import get_query_service
from knowledge.service.query_service import QueryService
from knowledge.utils.sse_util import create_sse_queue, sse_generator
from knowledge.utils.task_util import get_task_result


def create_app():
    app = FastAPI(title="听书知识库查询应用", version="v1.0")

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
        return {"flag": "success", "service": "听书知识库"}

    @app.post('/query')
    async def query_endpoint(
            request: QueryRequest,
            background_tasks: BackgroundTasks,
            service: QueryService = Depends(get_query_service)
    ) -> Union[StreamSubmitResponse, QueryResponse]:
        """
        处理查询请求（听书平台版本）
        """
        session_id = request.session_id or service.generate_session_id()
        task_id = service.generate_task_id()

        # 流式调用
        if request.is_stream:
            create_sse_queue(task_id)
            background_tasks.add_task(
                service.run_query_graph,
                task_id=task_id,
                session_id=session_id,
                query=request.query,
                is_stream=True,
                enable_evaluation=request.enable_evaluation,
                ground_truth=request.ground_truth or "",
            )
            return StreamSubmitResponse(
                message='查询请求已经提交',
                session_id=session_id,
                task_id=task_id
            )
        else:
            # 同步调用
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                service.run_query_graph,
                session_id,
                task_id,
                request.query,
                request.is_stream,
                request.enable_evaluation,
                request.ground_truth or "",
            )
            answer = service.get_task_result(task_id)

            evaluation_result = None
            if request.enable_evaluation:
                evaluation_result = get_task_result(task_id, key="evaluation")
                if isinstance(evaluation_result, str):
                    try:
                        evaluation_result = json.loads(evaluation_result)
                    except json.JSONDecodeError:
                        evaluation_result = None

            return QueryResponse(
                message='查询请求已经处理完了',
                session_id=session_id,
                answer=answer,
                evaluation_result=evaluation_result,
            )

    @app.get("/stream/{task_id}")
    async def stream(task_id: str, request: Request) -> StreamingResponse:
        """SSE 流式输出"""
        return StreamingResponse(
            content=sse_generator(task_id, request),
            media_type="text/event-stream"
        )

    @app.get("/history/{session_id}")
    async def get_history(
            session_id: str,
            limit: int = 50,
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